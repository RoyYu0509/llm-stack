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

import functools

import torch
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

class range:  # noqa: A001 - mirrors torch's name
    """``with nvtx.range(msg):`` and ``@nvtx.range(msg)``; a no-op off CUDA, and a no-op while
    ``torch.compile`` is tracing.

    This used to be ``torch.cuda.nvtx.range`` itself (a ``contextlib.contextmanager`` generator, i.e. a
    ``ContextDecorator``). TorchDynamo cannot inline ``ContextDecorator.__call__.<locals>.inner`` (it lives in
    the stdlib's skipfiles), so every decorated ``forward`` was a graph break and ``fullgraph=True`` failed on
    the first ``Embedding.forward``. A plain wrapper that checks ``torch.compiler.is_compiling()`` is traced
    straight through (the branch is constant-folded), and in eager mode it pushes / pops exactly as before."""

    def __init__(self, msg: str = "", *args, **kwargs):
        self._msg, self._args, self._kwargs = msg, args, kwargs

    def __enter__(self):
        if NVTX_AVAILABLE and not torch.compiler.is_compiling():
            _torch_nvtx.range_push(self._msg.format(*self._args, **self._kwargs))
        return self

    def __exit__(self, *exc) -> bool:
        if NVTX_AVAILABLE and not torch.compiler.is_compiling():
            _torch_nvtx.range_pop()
        return False

    def __call__(self, fn):
        msg, args, kwargs = self._msg, self._args, self._kwargs

        @functools.wraps(fn)
        def wrapper(*a, **k):
            if torch.compiler.is_compiling():
                return fn(*a, **k)
            with range(msg, *args, **kwargs):
                return fn(*a, **k)

        return wrapper


if NVTX_AVAILABLE:
    range_push = _torch_nvtx.range_push
    range_pop = _torch_nvtx.range_pop
    mark = _torch_nvtx.mark
else:

    def range_push(msg: str = "", *args, **kwargs) -> int:
        return 0

    def range_pop(*args, **kwargs) -> int:
        return 0

    def mark(msg: str = "", *args, **kwargs) -> None:
        return None


__all__ = ["NVTX_AVAILABLE", "range", "range_push", "range_pop", "mark"]
