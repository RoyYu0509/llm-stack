"""
Reference solution that:
- keeps your "make_block_ptr + advance + tile loop" style
- keeps the same 2-kernel backward logic (row-wise kernel -> reduce kernel)
- fixes tensor SHAPES to match the uploaded reference script:
    x2d:   (M, N)  where N = feature dim
    mean:  (M,)
    rstd:  (M,)
    w,b:   (N,)
    dLdw_intermediate / dLdb_intermediate: (GROUP_SIZE, N)
    locks: (2*GROUP_SIZE,)  (first half = lock state, second half = "first write" flag)
"""

import triton
import triton.language as tl
import torch

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")


# --------------------------
# Forward kernel (block_ptr)
# --------------------------
@triton.jit
def _layernorm_forward(
    X_ptr, OUT_ptr,
    w_ptr, b_ptr,
    mean_ptr, rstd_ptr,
    stride_M: tl.constexpr,  # x2d.stride(0)
    M, N,                    # runtime shapes
    eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)  # row index in [0, M)

    # We keep your block_ptr approach, but fix strides/offsets
    # X2D shape = (M, N) with row-major layout => strides=(stride_M, 1)
    X_ptr0 = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, N),
        strides=(stride_M, 1),
        offsets=(pid, 0),
        block_shape=(1, BLOCK_SIZE_N),
        order=(1, 0),
    )
    OUT_ptr0 = tl.make_block_ptr(
        base=OUT_ptr,
        shape=(M, N),
        strides=(stride_M, 1),
        offsets=(pid, 0),
        block_shape=(1, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # w,b are 1D of length N, offsets MUST start at 0 (not pid)
    w_ptr0 = tl.make_block_ptr(
        base=w_ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    b_ptr0 = tl.make_block_ptr(
        base=b_ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    mean_ptr0 = tl.make_block_ptr(
        base=mean_ptr,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(1,),
        order=(0,),
    )
    rstd_ptr0 = tl.make_block_ptr(
        base=rstd_ptr,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(1,),
        order=(0,),
    )

    # IMPORTANT: since you "advance" pointers, use fresh pointers for each pass
    X_mean_ptr = X_ptr0
    X_var_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, N),
        strides=(stride_M, 1),
        offsets=(pid, 0),
        block_shape=(1, BLOCK_SIZE_N),
        order=(1, 0),
    )
    X_out_ptr = tl.make_block_ptr(
        base=X_ptr,
        shape=(M, N),
        strides=(stride_M, 1),
        offsets=(pid, 0),
        block_shape=(1, BLOCK_SIZE_N),
        order=(1, 0),
    )
    OUT_out_ptr = OUT_ptr0
    w_out_ptr = w_ptr0
    b_out_ptr = b_ptr0

    # With BLOCK_SIZE_N >= N (enforced in wrapper), this loop runs once; we keep it for your "logic"
    acc_sum = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(1):
        x = tl.load(X_mean_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        acc_sum += x
        # keep your style
        X_mean_ptr = X_mean_ptr.advance((0, BLOCK_SIZE_N))

    row_sum = tl.sum(acc_sum, axis=1)        # (1,)
    mean = row_sum / N                       # (1,)

    acc_var = tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(1):
        x = tl.load(X_var_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        cols = tl.arange(0, BLOCK_SIZE_N)[None, :]
        diff = tl.where(cols < N, x - mean, 0.0)
        acc_var += diff * diff
        X_var_ptr = X_var_ptr.advance((0, BLOCK_SIZE_N))

    var = tl.sum(acc_var, axis=1) / N        # (1,)
    rstd = 1.0 / tl.sqrt(var + eps)          # (1,)

    tl.store(mean_ptr0, mean, boundary_check=(0,))
    tl.store(rstd_ptr0, rstd, boundary_check=(0,))

    # Apply affine: y = ((x-mean)*rstd)*w + b
    for _ in range(1):
        x = tl.load(X_out_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        wv = tl.load(w_out_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)[None, :]
        bv = tl.load(b_out_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)[None, :]

        cols = tl.arange(0, BLOCK_SIZE_N)[None, :]
        xhat = tl.where(cols < N, (x - mean) * rstd, 0.0)
        y = xhat * wv + bv

        tl.store(OUT_out_ptr, y, boundary_check=(0, 1))

        X_out_ptr = X_out_ptr.advance((0, BLOCK_SIZE_N))
        OUT_out_ptr = OUT_out_ptr.advance((0, BLOCK_SIZE_N))
        w_out_ptr = w_out_ptr.advance((BLOCK_SIZE_N,))
        b_out_ptr = b_out_ptr.advance((BLOCK_SIZE_N,))


# ------------------------------------------
# Backward row kernel (dLdx + partial dLdw,b)
# ------------------------------------------
@triton.jit
def _layernorm_backward_row(
    X_ptr, dLdx_ptr, dLdy_ptr,
    w_ptr,
    dLdw_inter_ptr, dLdb_inter_ptr,
    mean_ptr, rstd_ptr,
    locks_ptr,                  # int32, shape (2*GROUP_SIZE,)
    stride_M: tl.constexpr,
    M, N,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)  # row in [0, M)

    # block pointers for this row
    X0 = tl.make_block_ptr(
        base=X_ptr, shape=(M, N), strides=(stride_M, 1),
        offsets=(pid, 0), block_shape=(1, BLOCK_SIZE_N), order=(1, 0)
    )
    dLdy0 = tl.make_block_ptr(
        base=dLdy_ptr, shape=(M, N), strides=(stride_M, 1),
        offsets=(pid, 0), block_shape=(1, BLOCK_SIZE_N), order=(1, 0)
    )
    dLdx0 = tl.make_block_ptr(
        base=dLdx_ptr, shape=(M, N), strides=(stride_M, 1),
        offsets=(pid, 0), block_shape=(1, BLOCK_SIZE_N), order=(1, 0)
    )

    w0 = tl.make_block_ptr(
        base=w_ptr, shape=(N,), strides=(1,),
        offsets=(0,), block_shape=(BLOCK_SIZE_N,), order=(0,)
    )

    mean0 = tl.make_block_ptr(base=mean_ptr, shape=(M,), strides=(1,), offsets=(pid,), block_shape=(1,), order=(0,))
    rstd0 = tl.make_block_ptr(base=rstd_ptr, shape=(M,), strides=(1,), offsets=(pid,), block_shape=(1,), order=(0,))

    mean = tl.load(mean0, boundary_check=(0,)).to(tl.float32)  # (1,)
    rstd = tl.load(rstd0, boundary_check=(0,)).to(tl.float32)  # (1,)

    # one-tile loop kept (BLOCK_SIZE_N >= N in wrapper)
    for _ in range(1):
        x = tl.load(X0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)     # (1, BS)
        g = tl.load(dLdy0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # (1, BS)
        w = tl.load(w0, boundary_check=(0,), padding_option="zero").to(tl.float32)[None, :]  # (1, BS)

        cols = tl.arange(0, BLOCK_SIZE_N)[None, :]
        xhat = tl.where(cols < N, (x - mean) * rstd, 0.0)          # (1, BS)
        dydx_normed = tl.where(cols < N, g * w, 0.0)               # (1, BS)

        c1 = tl.sum(xhat * dydx_normed, axis=1) / N                # (1,)
        c2 = tl.sum(dydx_normed, axis=1) / N                       # (1,)

        dLdx = (dydx_normed - (xhat * c1 + c2)) * rstd             # (1, BS)
        tl.store(dLdx0, dLdx, boundary_check=(0, 1))

        # partial contributions
        dLdw_part = (g * xhat).to(tl.float16)  # match template: store partials in x.dtype (often fp16)
        dLdb_part = g.to(tl.float16)

        # lock + accumulate into (GROUP_SIZE, N)
        lock_id = pid % GROUP_SIZE
        lock_ptr = locks_ptr + lock_id
        count_ptr = locks_ptr + GROUP_SIZE + lock_id

        # spin lock
        while tl.atomic_cas(lock_ptr, 0, 1) == 1:
            pass

        # create block_ptrs for the lock row in the intermediate buffers
        dLdw_row = tl.make_block_ptr(
            base=dLdw_inter_ptr, shape=(GROUP_SIZE, N), strides=(N, 1),
            offsets=(lock_id, 0), block_shape=(1, BLOCK_SIZE_N), order=(1, 0)
        )
        dLdb_row = tl.make_block_ptr(
            base=dLdb_inter_ptr, shape=(GROUP_SIZE, N), strides=(N, 1),
            offsets=(lock_id, 0), block_shape=(1, BLOCK_SIZE_N), order=(1, 0)
        )

        used = tl.load(count_ptr)
        if used != 0:
            prev_w = tl.load(dLdw_row, boundary_check=(0, 1), padding_option="zero")
            prev_b = tl.load(dLdb_row, boundary_check=(0, 1), padding_option="zero")
            dLdw_part += prev_w
            dLdb_part += prev_b
        else:
            tl.atomic_xchg(count_ptr, 1)

        tl.store(dLdw_row, dLdw_part, boundary_check=(0, 1))
        tl.store(dLdb_row, dLdb_part, boundary_check=(0, 1))

        tl.atomic_xchg(lock_ptr, 0)

        # keep your style (advance once)
        X0 = X0.advance((0, BLOCK_SIZE_N))
        dLdy0 = dLdy0.advance((0, BLOCK_SIZE_N))
        dLdx0 = dLdx0.advance((0, BLOCK_SIZE_N))
        w0 = w0.advance((BLOCK_SIZE_N,))


# --------------------------
# Reduce kernel (same as ref)
# --------------------------
@triton.jit
def _w_b_reduce_kernel(
    dLdw_inter_ptr, dLdb_inter_ptr,
    dLdw_ptr, dLdb_ptr,
    GROUP_SIZE, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    col_ptrs = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dLdw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dLdb_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, GROUP_SIZE, BLOCK_SIZE_M):
        row_ptrs = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (row_ptrs[:, None] < GROUP_SIZE) & (col_ptrs[None, :] < N)
        offsets = row_ptrs[:, None] * N + col_ptrs[None, :]
        dLdw_acc += tl.load(dLdw_inter_ptr + offsets, mask=mask, other=0.0)
        dLdb_acc += tl.load(dLdb_inter_ptr + offsets, mask=mask, other=0.0)

    tl.store(dLdw_ptr + col_ptrs, tl.sum(dLdw_acc, axis=0), mask=col_ptrs < N)
    tl.store(dLdb_ptr + col_ptrs, tl.sum(dLdb_acc, axis=0), mask=col_ptrs < N)


# --------------------------
# Autograd wrapper
# --------------------------
class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, w, b, eps=1e-5):
        # flatten to (M, N) where N = feature dim
        x_shape = x.shape
        N = x_shape[-1]
        x2d = x.reshape(-1, N)
        M = x2d.shape[0]

        y2d = torch.empty_like(x2d)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        # same 64KB heuristic as reference
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This LayerNorm only supports feature dim < 64KB.")

        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 8)

        _layernorm_forward[(M,)](
            x2d, y2d, w, b, mean, rstd,
            x2d.stride(0),
            M, N,
            eps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x2d, w, b, mean, rstd)
        ctx.x_shape = x_shape
        ctx.M = M
        ctx.N = N
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps

        return y2d.reshape(x_shape)

    @staticmethod
    def backward(ctx, dLdy):
        x2d, w, b, mean, rstd = ctx.saved_tensors
        M, N = ctx.M, ctx.N

        dLdy2d = dLdy.reshape(M, N).contiguous()
        dLdx2d = torch.empty_like(x2d)
        dLdw = torch.empty((N,), dtype=w.dtype, device=w.device)
        dLdb = torch.empty((N,), dtype=w.dtype, device=w.device)

        # same group-size heuristic as reference
        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 96
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256

        dLdw_inter = torch.zeros((GROUP_SIZE, N), dtype=x2d.dtype, device=w.device)
        dLdb_inter = torch.zeros((GROUP_SIZE, N), dtype=x2d.dtype, device=w.device)
        locks = torch.zeros((2 * GROUP_SIZE,), dtype=torch.int32, device=w.device)

        _layernorm_backward_row[(M,)](
            x2d, dLdx2d, dLdy2d,
            w,
            dLdw_inter, dLdb_inter,
            mean, rstd,
            locks,
            x2d.stride(0),
            M, N,
            GROUP_SIZE=GROUP_SIZE,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N,
            num_warps=ctx.num_warps,
        )

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        _w_b_reduce_kernel[grid](
            dLdw_inter, dLdb_inter,
            dLdw, dLdb,
            min(GROUP_SIZE, M), N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128,
        )

        return dLdx2d.reshape(ctx.x_shape), None, dLdw, dLdb, None


layer_norm_triton = LayerNorm.apply


# --------------------------
# Quick sanity test (2D)
# --------------------------
from layer_norm_ref import layernorm
def test_layernorm_kernel(M=256, N=1024, dtype=torch.float16, eps=1e-5, device=DEVICE):
    x = (-2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)).requires_grad_(True)
    w = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    b = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    dLdy = 0.1 * torch.randn_like(x)

    y_tri = layer_norm_triton(x, (N,), w, b, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N,), w, b, eps).to(dtype)
    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0)

    y_tri.backward(dLdy, retain_graph=True)
    dLdx_tri, dLdw_tri, dLdb_tri = [t.grad.clone() for t in (x, w, b)]
    x.grad, w.grad, b.grad = None, None, None

    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [t.grad.clone() for t in (x, w, b)]

    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)
    print("PASS: forward + backward")


# Uncomment to run:
if __name__ == "__main__":
    test_layernorm_kernel()
