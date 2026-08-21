import torch
import torch.distributed as dist


class BucketedOverlapDDP(torch.nn.Module):
    """
    DDP wrapper with gradient bucketing and async all-reduce to overlap
    communication with backward computation.

    Usage:
        model = BucketedOverlapDDP(model, bucket_size_bytes=25 * 1024 * 1024)
        model.add_grad_allreduce_hook()
        ... loss.backward() ...
        model.wait_on_queue()  # ensure all comm complete before optimizer.step()
    """

    def __init__(self, module: torch.nn.Module, bucket_size_bytes: int = 25 * 1024 * 1024):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = bucket_size_bytes
        self._hooks = []
        self._buckets = []
        self._work_handles = []
        self._bucket_param_slices = []
        self._built = False

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _build_buckets(self):
        params = [p for p in self.module.parameters() if p.requires_grad]
        if not params:
            self._built = True
            return

        device = params[0].device
        dtype = params[0].dtype
        current_bucket = []
        current_size = 0

        for p in params:
            numel = p.numel()
            bytes_size = numel * p.element_size()
            if current_bucket and current_size + bytes_size > self.bucket_size_bytes:
                self._finalize_bucket(current_bucket, device, dtype)
                current_bucket = []
                current_size = 0

            current_bucket.append(p)
            current_size += bytes_size

        if current_bucket:
            self._finalize_bucket(current_bucket, device, dtype)

        self._built = True

    def _finalize_bucket(self, params, device, dtype):
        total_numel = sum(p.numel() for p in params)
        bucket_buffer = torch.zeros(total_numel, device=device, dtype=dtype)
        slices = []
        offset = 0
        for p in params:
            n = p.numel()
            slices.append((p, offset, offset + n))
            offset += n

        self._buckets.append(
            {
                "buffer": bucket_buffer,
                "slices": slices,
                "ready": 0,
                "count": len(slices),
            }
        )

    def add_grad_allreduce_hook(self):
        if not self._built:
            self._build_buckets()

        for bucket in self._buckets:
            for p, start, end in bucket["slices"]:
                def _post_hook_fn(grad_tensor, _bucket=bucket, _start=start, _end=end):
                    _bucket["buffer"][_start:_end].copy_(grad_tensor.view(-1))
                    _bucket["ready"] += 1
                    if _bucket["ready"] == _bucket["count"]:
                        work = dist.all_reduce(
                            _bucket["buffer"], op=dist.ReduceOp.AVG, async_op=True
                        )
                        self._work_handles.append((work, _bucket))
                    return grad_tensor

                handle = p.register_post_accumulate_grad_hook(_post_hook_fn)
                self._hooks.append(handle)

    def wait_on_queue(self):
        for work, bucket in self._work_handles:
            work.wait()
            # Scatter reduced bucket back into per-parameter grads
            for p, start, end in bucket["slices"]:
                p.grad.view(-1).copy_(bucket["buffer"][start:end])
            bucket["ready"] = 0

        self._work_handles.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
