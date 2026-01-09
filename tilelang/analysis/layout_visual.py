import tilelang.language as T
from tvm import tir
from tvm.tir import PyStmtExprVisitor

from tvm.tir.transform import prim_func_pass
from tilelang.tools.plot_layout import plot_layout
from tilelang.tools.plot_shared_layout import plot_shared_layout, print_shared_layout_format


def print_fragment_format(layout: T.Fragment) -> str:
    """
    Format fragment layout information into a human-readable string.

    Parameters
    ----------
    layout : T.Fragment
        The fragment layout to format

    Returns
    -------
    str
        Formatted string showing shape, thread mapping, and index mapping
    """
    if isinstance(layout, T.Fragment):
        input_shape = layout.get_input_shape()
        output_shape = layout.get_output_shape()
        lines = [f"  Shape: {input_shape} -> {output_shape}", f"  Thread: {layout.forward_thread}", f"  Index:  {layout.forward_index}"]
        print("\n".join(lines))
    else:
        raise ValueError(f"Expected T.Fragment, but got {type(layout).__name__}")


@tir.functor.visitor
class _LayoutVisualVisitor(PyStmtExprVisitor):
    """
    User-friendly pass which visualizes fragment and shared memory layouts
    inferred during compilation.

    In TileLang:
    - Fragment layouts describe how logical indices map to thread IDs and
      register file locations (for local/register buffers like C_local)
    - Shared memory layouts describe swizzle patterns for bank conflict
      avoidance (for shared buffers like A_shared, B_shared)

    This pass generates two types of output:
    1. Textual output: A human-readable description printed to console
    2. Visual diagrams: Color-coded plots saved to files (PDF, PNG, SVG formats)

    Configuration:
    The pass is controlled by the TL_ENABLE_LAYOUT_VISUALIZATION configuration option.
    The configuration accepts string values:

    - Empty string or not set: Pass does nothing (default, disabled)
    - "png": Generate PNG format only (recommended for quick inspection)
    - "pdf": Generate PDF format only (recommended for documentation)
    - "svg": Generate SVG format only (recommended for web/vector graphics)
    - "all": Generate all formats (PDF, PNG, SVG)
    - "png,svg": Generate multiple formats (comma-separated)
    """

    def __init__(self, formats: list[str] = ""):
        super().__init__()
        self.layout_found = []
        self.processed_layouts = set()
        self.formats_list = [f for f in formats if f != "txt"]

    def visit_block_(self, op: tir.Block) -> None:
        if "layout_map" in op.annotations:
            layout_map = op.annotations["layout_map"]

            for key, layout in layout_map.items():
                # Get buffer name from key
                if hasattr(key, "name"):
                    key_name = key.name
                else:
                    key_name = str(key)

                # Use buffer name for dedup (not layout content)
                # This ensures each buffer gets its own visualization even if layouts are identical
                if key_name in self.processed_layouts:
                    continue

                if isinstance(layout, T.Fragment):
                    continue
                    # Handle Fragment layout (register/local buffers)
                    print(f"{key_name} inferenced layout:")
                    print_fragment_format(layout)
                    for fmt in self.formats_list:
                        plot_layout(layout, name=f"{key_name}_layout", formats=fmt)
                    self.processed_layouts.add(key_name)

                elif isinstance(layout, T.Layout):
                    # Handle Layout (shared memory buffers with swizzle)
                    # Get element size from buffer dtype
                    element_bits = 16  # default to bfloat16/float16
                    if hasattr(key, "dtype"):
                        import tvm

                        dtype = tvm.DataType(key.dtype)
                        element_bits = dtype.bits

                    print(f"{key_name} shared memory layout:")
                    print_shared_layout_format(layout, name=key_name)
                    for fmt in self.formats_list:
                        plot_shared_layout(layout, name=f"{key_name}_layout", formats=fmt, element_bits=element_bits)
                    self.processed_layouts.add(key_name)

        # super().visit_block_(op)


def LayoutVisual(formats: str = ""):
    def pass_fn(func: tir.PrimFunc, mod, ctx):
        _LayoutVisualVisitor(formats=formats).visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
