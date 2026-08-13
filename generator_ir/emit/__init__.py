"""Source emission backends."""

from generator_ir.emit.nvidia import CudaEmitter, translate_cuda
from generator_ir.emit.amd import HipEmitter, translate_hip


def emitter_for(backend: str):
    """Return the emitter class for 'cuda' or 'hip'."""
    if backend == "hip":
        return HipEmitter
    if backend == "cuda":
        return CudaEmitter
    raise ValueError(f"unknown backend: {backend!r}")


def rendersrc(program, backend: str) -> str:
    return emitter_for(backend)().rendersrc(program)


__all__ = ["CudaEmitter", "HipEmitter", "translate_cuda", "translate_hip",
           "emitter_for", "rendersrc"]
