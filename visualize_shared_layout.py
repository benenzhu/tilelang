#!/usr/bin/env python3
"""
Visualize Shared Memory Layout (Swizzle) for TileLang

This script can:
1. Extract A_shared/B_shared layouts from TileLang compilation
2. Visualize the swizzle pattern used for bank conflict avoidance
"""
# %%
# %config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tilelang.language as T
from tilelang.layout import Layout


def plot_shared_layout(
    layout: Layout,
    name: str = "shared_layout",
    save_directory: str = "./tmp",
    formats: str = "svg",
    font_size: int = 8,
    num_banks: int = 32,
):
    """
    Visualize a T.Layout (shared memory swizzle layout).

    Unlike Fragment which has thread mapping, Layout only has index mapping.
    This visualizes how logical (row, col) maps to physical address.

    Args:
        layout: T.Layout object
        name: Output file name
        save_directory: Directory to save figures
        formats: Output format (png, svg, pdf)
        font_size: Font size for cell labels
        num_banks: Number of shared memory banks (typically 32)
    """
    import os
    import pathlib

    # Get the input shape of the layout
    input_shape = layout.get_input_shape()
    input_shape = [int(var) for var in input_shape]

    print(f"Layout input shape: {input_shape}")
    print(f"Layout forward_index: {layout.get_forward_index()}")

    # For shared memory, typically 2D: (rows, cols)
    if len(input_shape) == 2:
        nrows, ncols = input_shape
    elif len(input_shape) == 3:
        # Handle (batch, rows, cols) by taking the last 2D slice
        _, nrows, ncols = input_shape
        print(f"Taking 2D slice: ({nrows}, {ncols})")
    else:
        raise ValueError(f"Expected 2D or 3D layout, got shape {input_shape}")

    # Build the physical address map
    addr_map = np.zeros((nrows, ncols), dtype=int)

    for row in range(nrows):
        for col in range(ncols):
            if len(input_shape) == 2:
                indices = [row, col]
            else:
                indices = [0, row, col]  # batch=0

            # Get the physical address from layout
            # Output may be multi-dimensional, take the last element (actual address)
            phys_addr = layout.map_forward_index(indices)
            if isinstance(phys_addr, (list, tuple)):
                phys_addr = phys_addr[-1]  # Take last dimension (the actual swizzled address)
            addr_map[row, col] = int(phys_addr)

    # Create figure
    cell_size = 0.6
    fig, ax = plt.subplots(figsize=(cell_size * ncols + 2, cell_size * nrows + 2))

    # Color by bank assignment (assuming 4-byte elements)
    cmap = plt.get_cmap("hsv", num_banks + 1)

    for row in range(nrows):
        for col in range(ncols):
            addr = addr_map[row, col]
            bank = addr % num_banks
            color = cmap(bank)

            rect = patches.Rectangle((col, row), 1, 1, linewidth=0.5, edgecolor="black", facecolor=color)
            ax.add_patch(rect)

            # Show physical address
            ax.text(col + 0.5, row + 0.5, str(addr), ha="center", va="center", fontsize=font_size)

    # Row/col labels
    for row in range(nrows):
        ax.text(-0.3, row + 0.5, str(row), ha="center", va="center", fontsize=font_size)
    for col in range(ncols):
        ax.text(col + 0.5, -0.3, str(col), ha="center", va="center", fontsize=font_size)

    ax.set_xlim(-0.5, ncols + 0.5)
    ax.set_ylim(-0.5, nrows + 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"{name}\nShape: {input_shape}, Forward: {layout.get_forward_index()}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Save
    tmp_directory = pathlib.Path(save_directory)
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    if formats in ["svg", "all"]:
        plt.savefig(tmp_directory / f"{name}.svg", bbox_inches="tight")
        print(f"Saved: {tmp_directory / f'{name}.svg'}")
    if formats in ["png", "all"]:
        plt.savefig(tmp_directory / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {tmp_directory / f'{name}.png'}")

    return fig


def extract_layouts_from_kernel(kernel):
    """
    Extract A_shared and B_shared layouts from a compiled TileLang kernel.

    This inspects the IR metadata to find layout_map annotations.
    """
    # Access the underlying module
    if hasattr(kernel, "mod"):
        mod = kernel.mod
    elif hasattr(kernel, "_mod"):
        mod = kernel._mod
    else:
        print("Cannot find module in kernel object")
        return None

    from tvm import tir

    layouts = {}

    def extract_from_block(block):
        if hasattr(block, "annotations") and "layout_map" in block.annotations:
            layout_map = block.annotations["layout_map"]
            for key, layout in layout_map.items():
                key_name = str(key) if hasattr(key, "name") else str(key)
                if "shared" in key_name.lower():
                    layouts[key_name] = layout
                    print(f"Found layout for {key_name}: {type(layout)}")

    # Visit all blocks
    class LayoutExtractor(tir.stmt_functor.StmtVisitor):
        def visit_block(self, op):
            extract_from_block(op)
            self.visit_stmt(op.body)

    for gvar, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            try:
                extractor = LayoutExtractor()
                extractor.visit_stmt(func.body)
            except Exception as e:
                print(f"Error extracting from {gvar}: {e}")

    return layouts


def demo_swizzle_layout():
    """
    Demo: create and visualize a swizzle layout manually.
    """
    from tilelang.layout.swizzle import make_full_bank_swizzled_layout

    # Create a swizzled layout for a 8x64 bfloat16 buffer (one swizzle period)
    # stride=8, continuous=64, element_size=16 bits
    # This shows the 128B XOR swizzle pattern
    layout = make_full_bank_swizzled_layout(8, 64, 16)

    print("Swizzle layout:")
    print(f"  Input shape: {layout.get_input_shape()}")
    print(f"  Output shape: {layout.get_output_shape()}")
    print(f"  Forward index: {layout.get_forward_index()}")

    # Visualize (use 8 banks for cleaner display, actual GPU has 32 banks)
    plot_shared_layout(layout, name="demo_swizzle_8x64", font_size=5, num_banks=8)
    plt.show()

    return layout


def demo_from_gemm():
    """
    Demo: extract layouts from a GEMM kernel.
    """
    import tilelang

    @tilelang.jit(out_idx=[-1], verbose=False)
    def matmul_nt(M, N, K, block_M, block_N, block_K, dtype=T.bfloat16, accum_dtype=T.float32):
        @T.prim_func
        def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    # Compile the kernel
    kernel = matmul_nt(256, 256, 128, 64, 64, 32)

    # Try to extract layouts
    layouts = extract_layouts_from_kernel(kernel)

    if layouts:
        for name, layout in layouts.items():
            if isinstance(layout, Layout):
                plot_shared_layout(layout, name=name, font_size=6)


def visualize_cuda_swizzle(rows=8, cols=64, element_size=16):
    """
    Visualize the swizzle pattern from CUDA code like:

    addr = i_1 * 4096 + (threadIdx.x >> 3) * 64 +
           (((threadIdx.x & 63) >> 5) + ((threadIdx.x & 7) >> 2) & 1) * 32 +
           (((threadIdx.x & 31) >> 4) + ((threadIdx.x & 3) >> 1) & 1) * 16 +
           (((threadIdx.x & 15) >> 3) + (threadIdx.x & 1) & 1) * 8

    This is a 128B XOR swizzle pattern.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Build address map
    addr_map = np.zeros((rows, cols), dtype=int)

    for row in range(rows):
        for col in range(cols):
            # 128B XOR swizzle formula (simplified for visualization)
            # XOR the row with groups of 8, 16, 32 columns
            addr = row * cols + col
            # XOR with row for each 8-column group
            swizzled_col = col
            swizzled_col = ((col // 8) ^ (row % 2)) * 8 + (col % 8)
            swizzled_col = ((swizzled_col // 16) ^ ((row // 2) % 2)) * 16 + (swizzled_col % 16)
            swizzled_col = ((swizzled_col // 32) ^ ((row // 4) % 2)) * 32 + (swizzled_col % 32)
            addr_map[row, col] = row * cols + swizzled_col

    # Plot
    fig, ax = plt.subplots(figsize=(cols * 0.4 + 1, rows * 0.5 + 1))

    num_banks = 8
    cmap = plt.get_cmap("hsv", num_banks + 1)

    for row in range(rows):
        for col in range(cols):
            addr = addr_map[row, col]
            # Color by address mod num_banks to show bank distribution
            bank = (addr // 8) % num_banks  # 8 elements per bank (16-bit elements, 4 bytes per bank)
            color = cmap(bank)

            rect = patches.Rectangle((col, row), 1, 1, linewidth=0.5, edgecolor="black", facecolor=color)
            ax.add_patch(rect)
            ax.text(col + 0.5, row + 0.5, str(addr), ha="center", va="center", fontsize=4)

    # Labels
    for row in range(rows):
        ax.text(-0.3, row + 0.5, str(row), ha="center", va="center", fontsize=8)
    for col in range(0, cols, 8):
        ax.text(col + 4, -0.4, f"{col}-{col + 7}", ha="center", va="center", fontsize=7)

    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("128B XOR Swizzle Pattern\n(row XOR (col // 8, col // 16, col // 32))", fontsize=10)
    ax.axis("off")
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Demo: Swizzle Layout Visualization")
    print("=" * 60)
    demo_swizzle_layout()

    print("\n" + "=" * 60)
    print("Demo: CUDA-style Swizzle Pattern")
    print("=" * 60)
    visualize_cuda_swizzle()
    plt.show()

# %%
