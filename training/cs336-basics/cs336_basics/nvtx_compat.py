"""A drop-in stand-in for ``torch.cuda.nvtx`` that degrades to a no-op off CUDA.

The model and its submodules annotate their forward passes with ``@nvtx.range(...)`` so
Nsight traces are readable. Those annotations are pure instrumentation, but
``torch.cuda.nvtx`` only *imports* cleanly on a non-CUDA build -- the first actual call
raises::

    RuntimeError: NVTX functions not installed. Are you sure you have a CUDA build?

which means importing and running ``TransformerLM`` at all is impossible on a CPU-only
or MPS machine. That breaks two things this project actually needs: the local
smoke-test-before-you-rent-a-GPU workflow, and serving the trained weights from
``llm-serving``, whose stated workflow is "CPU/MPS functional testing during
development".

So: probe NVTX once at import, and fall back to a no-op ``range`` if it isn't usable.
On a real CUDA box this is exactly ``torch.cuda.nvtx`` and profiling is unaffected.

Usage is identical to the module it replaces::

    from cs336_basics import nvtx_compat as nvtx

    @nvtx.range("MyModule_forward")
    def forward(self, x): ...
"""

import contextlib

import torch.cuda.nvtx as _torch_nvtx


def _nvtx_is_usable() -> bool:
    """Probe rather than infer. ``torch.cuda.is_available()`` is the wrong question --
    it asks about a *device*, while NVTX is about whether the *build* has the symbols,
    and the two come apart (a CUDA build on a machine with no GPU still has NVTX)."""
    try:
        _torch_nvtx.range_push("_nvtx_probe")
        _torch_nvtx.range_pop()
        return True
    except Exception:
        return False


NVTX_AVAILABLE = _nvtx_is_usable()

if NVTX_AVAILABLE:
    range = _torch_nvtx.range
    range_push = _torch_nvtx.range_push
    range_pop = _torch_nvtx.range_pop
    mark = _torch_nvtx.mark
else:

    @contextlib.contextmanager
    def range(msg: str = "", *args, **kwargs):  # noqa: A001 - mirrors torch's name
        """No-op form. ``contextlib.contextmanager`` yields a ContextDecorator, so this
        stays usable both as ``with nvtx.range(...)`` and as ``@nvtx.range(...)``,
        matching torch's own implementation."""
        yield

    def range_push(msg: str = "", *args, **kwargs) -> int:
        return 0

    def range_pop(*args, **kwargs) -> int:
        return 0

    def mark(msg: str = "", *args, **kwargs) -> None:
        return None


__all__ = ["NVTX_AVAILABLE", "range", "range_push", "range_pop", "mark"]
