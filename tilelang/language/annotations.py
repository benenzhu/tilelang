"""Annotation helpers exposed on the TileLang language surface."""

from typing import Callable

from tilelang.layout import Fragment, Layout
from tilelang.utils.language import is_fragment
from tvm.script.parser.tir import attr, block_attr
from tvm.tir import FloatImm

__all__ = [
    "use_swizzle",
    "annotate_layout",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
    "annotate_restrict_buffers",
]


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True, num_xcds: int = 0):
    """Annotate a kernel to use a specific threadblock swizzle pattern.

    Parameters
    ----------
    panel_size : int
        Width of the panel (group-M size) for the L2 cache swizzle.
    order : str
        "row" or "column" — selects Row-major or Column-major panel layout.
    enable : bool
        Set to False to disable swizzle entirely.
    num_xcds : int
        Number of XCDs (chiplets) for AMD MI-series GPUs.
        When > 0, an XCD remap is applied before the panel swizzle so that
        ``panel_size²`` consecutive workgroups land on the same XCD,
        reducing cross-chiplet traffic.
        Typical values: 8 for MI300X, 2 for MI250X.
        0 (default) disables XCD remap.
    """
    if not enable:
        return None
    if num_xcds > 0:
        suffix = "Xcd"
        tparams = f"<{panel_size}, {num_xcds}>"
    else:
        suffix = ""
        tparams = f"<{panel_size}>"
    device_func = ("rasterization2DRow" if order == "row" else "rasterization2DColumn") + suffix
    return attr(None, "threadblock_swizzle_pattern", f"tl::{device_func}{tparams}")


def annotate_layout(layout_map: dict):
    """Annotate the layout of the buffer."""
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if is_fragment(buffer):
            assert isinstance(layout, Fragment), f"for Fragment {buffer}, layout must be a Fragment, but got {type(layout)}"
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")

    return block_attr({"layout_map": _layout_map})


def annotate_safe_value(safe_value_map: dict):
    """Annotate the safe value of the buffer."""
    _safe_value_map = {}
    for buffer, safe_value in safe_value_map.items():
        _safe_value_map[buffer.data] = safe_value
    return block_attr({"safe_value_map": _safe_value_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: dict):
    """Annotate the L2 hit ratio of the buffer."""
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = FloatImm("float32", float(hit_ratio))
    return block_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})


def annotate_restrict_buffers(*buffers):
    """Mark the given buffer parameters as non-restrict.

    This annotation tells codegen to omit the `__restrict__` qualifier for the
    specified kernel buffer parameters. Use this when two (or more) buffers may
    alias, for example overlapping slices from the same base tensor.

    Example
    -------
    >>> @T.prim_func
    ... def buggy_kernel(x: T.Tensor((N,), T.float32),
    ...                  y: T.Tensor((N,), T.float32)):
    ...     T.annotate_restrict_buffers(x, y)
    ...     with T.Kernel(N, threads=32) as pid:
    ...         y[pid] = x[pid] + 1
    """
    if not buffers:
        return None
    data_vars = []
    for buf in buffers:
        try:
            data_vars.append(buf.data)
        except Exception as e:
            raise TypeError(f"annotate_restrict_buffers expects Buffer arguments, got {type(buf)}") from e
    # Also return as block attribute (root block exists by default) for readability/tools.
    return block_attr({"tl.non_restrict_params": data_vars})
