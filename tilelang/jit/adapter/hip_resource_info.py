"""Query AMD GPU kernel resource usage (VGPRs, scratch, max threads) by
loading a HSACO module through libamdhip64 directly.

Mirrors what triton's amd backend does in driver.c::loadBinary, but as a
standalone helper so we can attach the numbers to a tilelang JITKernel
without having to dig the loaded hipFunction_t out of TVM's runtime.

Also exposes a small thread-local recorder used by
``tilelang_callback_hip_compile`` to expose the HSACO it produced to the
JITKernel that triggered the lowering — there is no other clean handle on
those bytes once TVM packs them into the host .so.
"""
from __future__ import annotations

import ctypes
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)

# Match a __global__ kernel header up to the parameter list, optionally
# skipping any number of attribute-style invocations like
# __launch_bounds__(256, 1) or __amdgpu_flat_work_group_size__(...) that
# appear between the return type and the kernel name.
_KERNEL_HEAD_RE = re.compile(
    r"__global__\b\s+\w[\w\s\*&]*?\s"  # `__global__ void` (or similar return type)
    r"((?:__\w+__\s*\([^)]*\)\s*)*)"  # zero or more __attr__(...) clauses
    r"(\w+)\s*\(",  # the actual kernel name + '('
)
_RESERVED_KERNEL_NAMES = {"__launch_bounds__", "__amdgpu_flat_work_group_size__"}


_RECORDER = threading.local()


def reset_recorder() -> None:
    """Start a fresh recording window. Safe to call repeatedly."""
    _RECORDER.items = []


def record_compiled_hsaco(code: str, hsaco: bytes | bytearray) -> None:
    """Called by ``tilelang_callback_hip_compile`` to expose the HSACO it
    just produced. No-op if no recorder is active on this thread."""
    items = getattr(_RECORDER, "items", None)
    if items is None:
        return
    items.append((code, bytes(hsaco)))


def pop_recorded_hsacos() -> list[tuple[str, bytes]]:
    """Return everything recorded since the last ``reset_recorder`` and
    clear the buffer."""
    items = getattr(_RECORDER, "items", [])
    _RECORDER.items = []
    return list(items)


def extract_kernel_names(hip_source: str) -> list[str]:
    """Best-effort extraction of __global__ kernel names from generated HIP
    source. Tilelang's codegen emits headers like ``__global__ void
    __launch_bounds__(N) kernel(...)`` so we skip any leading attribute
    clauses before the kernel name."""
    seen: list[str] = []
    for _attrs, name in _KERNEL_HEAD_RE.findall(hip_source):
        if name in _RESERVED_KERNEL_NAMES:
            continue
        if name not in seen:
            seen.append(name)
    return seen

# hipFunction_attribute values, copied from <hip/hip_runtime_api.h>.
_HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
_HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
_HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
_HIP_FUNC_ATTRIBUTE_NUM_REGS = 4

_HIP_LIB_CANDIDATES: tuple[str, ...] = (
    "libamdhip64.so",
    "libamdhip64.so.7",
    "libamdhip64.so.6",
)


@dataclass
class KernelResourceUsage:
    """Per-kernel resource numbers as reported by the HIP runtime."""

    n_regs: int  # VGPRs (HIP_FUNC_ATTRIBUTE_NUM_REGS)
    n_spills: int  # scratch / spill bytes / 4 (matches triton's accounting)
    n_max_threads: int  # max threads/block the kernel may legally launch with
    shared_bytes: int  # static LDS bytes


class _HipFunctionsUnavailable(RuntimeError):
    pass


def _candidate_paths() -> Iterable[str]:
    yield from _HIP_LIB_CANDIDATES
    rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    if rocm_home:
        for cand in _HIP_LIB_CANDIDATES:
            yield os.path.join(rocm_home, "lib", cand)
    for cand in _HIP_LIB_CANDIDATES:
        yield os.path.join("/opt/rocm/lib", cand)


_hip_lib: ctypes.CDLL | None = None


def _load_hip() -> ctypes.CDLL:
    global _hip_lib
    if _hip_lib is not None:
        return _hip_lib
    last_err: OSError | None = None
    for cand in _candidate_paths():
        try:
            _hip_lib = ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            return _hip_lib
        except OSError as e:
            last_err = e
            continue
    raise _HipFunctionsUnavailable(
        f"libamdhip64.so not found; last error: {last_err}"
    )


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"HIP call {what} failed with code {rc}")


def query_kernel_resources(
    hsaco: bytes | bytearray, kernel_name: str
) -> KernelResourceUsage:
    """Load `hsaco` into a transient hipModule_t, look up `kernel_name`,
    and read the per-function attributes via hipFuncGetAttribute.

    The module is unloaded before returning, so this leaves no GPU state
    behind.
    """
    lib = _load_hip()

    # Signatures. hipModule_t / hipFunction_t are opaque pointers.
    lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipModuleLoadData.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    lib.hipFuncGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.hipFuncGetAttribute.restype = ctypes.c_int
    lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
    lib.hipModuleUnload.restype = ctypes.c_int

    buf = (ctypes.c_ubyte * len(hsaco)).from_buffer_copy(bytes(hsaco))
    module = ctypes.c_void_p()
    _check(
        lib.hipModuleLoadData(ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)),
        "hipModuleLoadData",
    )
    try:
        func = ctypes.c_void_p()
        _check(
            lib.hipModuleGetFunction(
                ctypes.byref(func), module, kernel_name.encode("utf-8")
            ),
            f"hipModuleGetFunction({kernel_name!r})",
        )

        def _attr(code: int) -> int:
            out = ctypes.c_int(0)
            _check(
                lib.hipFuncGetAttribute(ctypes.byref(out), code, func),
                f"hipFuncGetAttribute({code})",
            )
            return int(out.value)

        n_regs = _attr(_HIP_FUNC_ATTRIBUTE_NUM_REGS)
        local_bytes = _attr(_HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
        n_max_threads = _attr(_HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        shared_bytes = _attr(_HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    finally:
        unload_rc = lib.hipModuleUnload(module)
        if unload_rc != 0:
            logger.warning("hipModuleUnload failed with code %d", unload_rc)

    # Match triton's convention so n_spills lines up with what users expect.
    return KernelResourceUsage(
        n_regs=n_regs,
        n_spills=local_bytes // 4,
        n_max_threads=n_max_threads,
        shared_bytes=shared_bytes,
    )


def is_available() -> bool:
    try:
        _load_hip()
        return True
    except _HipFunctionsUnavailable:
        return False
