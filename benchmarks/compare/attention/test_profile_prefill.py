"""CPU tests for profile_prefill.py: the parsers (nsys CSV layouts), the busy-fraction maths, the grouping.

The Nsight Systems capture itself needs a GPU box; what can go wrong on a laptop is the arithmetic and the CSV
parsing, so that is what is pinned down here (with hand-computed expectations).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import profile_prefill as pp  # noqa: E402

pytest.importorskip("cs336_basics.transfromer.scaled_dot_prod_attention")


def test_merged_busy_ns_overlap_gap_and_span():
    busy, span = pp.merged_busy_ns([(0, 10), (5, 10), (30, 5)])   # [0,15) union, gap, [30,35)
    assert busy == 20 and span == 35
    assert pp.merged_busy_ns([]) == (0.0, 0.0)
    busy, span = pp.merged_busy_ns([(100, 50), (110, 5)])         # nested interval: span uses the MAX end
    assert busy == 50 and span == 50


KERN_CSV = '''Generating SQLite file x.sqlite from x.nsys-rep
Processing [x.sqlite] with [cuda_gpu_kern_sum]...

"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"
"90.0","900000","10","90000.0","90000","80000","95000","1000.0","void flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128>, bool, true>(Flash_fwd_params)"
"10.0","100000","30","3333.3","3000","2000","5000","300.0","void at::native::elementwise_kernel<128, 2>(int)"
'''

TRACE_CSV = '''Processing [x.sqlite] with [cuda_gpu_trace]...
"Start (ns)","Duration (ns)","CorrId","GrdX","GrdY","GrdZ","BlkX","BlkY","BlkZ","Reg/Trd","Name"
"1000","100","1","1","1","1","1","1","1","32","kernel_a"
"1050","100","2","1","1","1","1","1","1","32","kernel_b"
"1400","100","3","1","1","1","1","1","1","32","kernel_c"
"1600","999","4","","","","","","","","[CUDA memcpy Device-to-Host]"
'''


def test_parse_nsys_kernel_summary_handles_banner_and_commas_in_names():
    ks = pp.parse_nsys_kernel_summary(KERN_CSV)
    assert [k["instances"] for k in ks] == [10, 30]
    assert ks[0]["total_ns"] == 900000 and "Flash_fwd_kernel_traits<64, 128, 128>" in ks[0]["name"]


def test_parse_nsys_gpu_trace_busy_fraction_ignores_memcpy():
    t = pp.parse_nsys_gpu_trace(TRACE_CSV)
    # kernels: [1000,1100) U [1050,1150) = 150 ns busy, then [1400,1500) = 100 ns; span = 1000..1500 = 500
    assert t["n_kernels"] == 3 and t["busy_ns"] == 250 and t["span_ns"] == 500
    assert t["busy_frac"] == pytest.approx(0.5)


def test_summarize_kernels_per_call_numbers_and_categories():
    ks = pp.parse_nsys_kernel_summary(KERN_CSV)
    s = pp.summarize_kernels(ks, n_calls=10)
    assert s["gpu_kernel_ms_per_call"] == pytest.approx(1_000_000 / 10 / 1e6)
    assert s["launches_per_call"] == pytest.approx(4.0)                 # (10 + 30) launches / 10 calls
    assert s["top_kernels"][0]["share"] == pytest.approx(0.9)
    assert s["category_share"]["fused_attention"] == pytest.approx(0.9)


@pytest.mark.parametrize("name,cat", [
    ("void flash_fwd_kernel<...>", "fused_attention"),
    ("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nn", "gemm"),
    ("void cutlass::Kernel<...>", "gemm"),
    ("void at::native::(anonymous namespace)::softmax_warp_forward<...>", "softmax_reduce"),
    ("void at::native::elementwise_kernel<128, 2, ...>", "mask_elementwise"),
    ("Memcpy DtoD (Device -> Device)", "memcpy_memset"),
    ("something_unknown", "other"),
])
def test_categorize(name, cat):
    assert pp.categorize(name) == cat


def test_summary_groups_by_system_and_geomean_ratio_is_right():
    def cell(system, seq, base, tool="torch"):
        return dict(system=system, seq=seq, tool=tool, status="ok", baseline_ms=base, gpu_kernel_ms_per_call=base * 0.9,
                    launches_per_call=3.0, gpu_busy_frac=0.9)

    cells = [cell("sdpa_flash", 128, 1.0), cell("sdpa_flash", 256, 2.0),
             cell("triton_fa_mine", 128, 2.0), cell("triton_fa_mine", 256, 8.0)]   # ratios 2 and 4 -> geomean 2.83
    args = pp.argparse.Namespace(systems=["sdpa_flash", "triton_fa_mine"], seqs=[128, 256], tokens=32768,
                                 active_iters=20)
    md = pp.summary_markdown(cells, args)
    row = [ln for ln in md.splitlines() if ln.startswith("| triton_fa_mine")][0]
    assert "2.83x" in row and "2/2" in row
    assert "| sdpa_flash |" in md and "1.00x" in md


def test_cpu_cell_runs_in_a_fresh_process(tmp_path):
    args = pp.argparse.Namespace(dtype="float32", tokens=256, heads=2, head_dim=16, device="cpu",
                                 warmup_calls=1, active_iters=2)
    cell = pp._spawn(args, "sdpa_math", 64, "torch", timeout_s=180)
    assert cell["status"] == "ok" and cell["batch"] == 4 and cell["cpu_top_ops"], cell


def test_nsys_missing_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(pp.shutil, "which", lambda _: None)
    args = pp.argparse.Namespace(nsys_bin=None, dtype="bfloat16", tokens=32768, heads=12, head_dim=64,
                                 device="cuda", warmup_calls=1, active_iters=2)
    cell = pp.nsys_cell(args, "sdpa_flash", 128, tmp_path)
    assert cell["status"] == "unavailable" and "nsys" in cell["note"]
    json.dumps(cell)


def test_truncated_nsys_capture_is_an_error_not_a_ten_times_smaller_number():
    """2026-10-16: nsys 2026.3.2 on a 550 driver captured ONE kernel for 10 calls. That must be refused."""
    assert pp.check_capture(1, 10) is not None and "truncated" in pp.check_capture(1, 10)
    assert pp.check_capture(10, 10) is None
    assert pp.check_capture(30, 10) is None                      # several kernels per call is normal


def test_find_nsys_prefers_explicit_then_env_then_known_path(tmp_path, monkeypatch):
    a, b = tmp_path / "a_nsys", tmp_path / "b_nsys"
    a.write_text(""); b.write_text("")
    monkeypatch.setenv("NSYS_BIN", str(b))
    assert pp.find_nsys(str(a)) == str(a)
    assert pp.find_nsys(None) == str(b)
    monkeypatch.delenv("NSYS_BIN")
    monkeypatch.setattr(pp, "NSYS_CANDIDATES", [])
    monkeypatch.setattr(pp.shutil, "which", lambda _: "/usr/bin/nsys")
    assert pp.find_nsys(None) == "/usr/bin/nsys"
