"""
Plot shared memory layout (swizzle pattern) for TileLang.

Unlike Fragment layouts which have thread mappings, shared memory Layouts
only have index mappings (swizzle patterns for bank conflict avoidance).
"""

from __future__ import annotations

import tilelang.language as T


def _detect_swizzle_period(layout: T.Layout, nrows: int, ncols: int) -> tuple[int, int]:
    """
    Detect the minimal repeating tile size for a swizzle layout.

    For XOR-based swizzle patterns, we find the smallest (row_period, col_period) where:
    - addr(r + row_period, c) - addr(r, c) is constant for all c
    - addr(r, c + col_period) - addr(r, c) = col_period (linear offset)

    Returns:
        (row_period, col_period): Size of the minimal repeating tile
    """

    def get_addr(row, col):
        if len(layout.get_input_shape()) == 2:
            indices = [row, col]
        else:
            indices = [0, row, col]
        phys = layout.map_forward_index(indices)
        if hasattr(phys, "__len__") and len(phys) > 0:
            phys = phys[-1]
        return int(phys)

    # Detect row period: find smallest p where addr(r+p, c) - addr(r, c) is constant for all c
    # This means the XOR pattern repeats every p rows
    row_period = nrows
    for period in [1, 2, 4, 8, 16, 32]:
        if period >= nrows:
            break
        is_periodic = True

        # Check all rows in [0, period) to see if pattern repeats
        for row in range(min(period, nrows - period)):
            # Sample multiple columns to verify
            num_cols_to_check = min(32, ncols)
            offsets = []
            for col in range(num_cols_to_check):
                addr1 = get_addr(row, col)
                addr2 = get_addr(row + period, col)
                offsets.append(addr2 - addr1)

            # All offsets should be the same for a valid period
            if len(set(offsets)) != 1:
                is_periodic = False
                break

        if is_periodic:
            row_period = period
            break

    # Detect column period: find smallest p where addr(r, c+p) - addr(r, c) = p
    # This means columns beyond p are just linear translations of columns 0 to p-1
    col_period = ncols
    for period in [8, 16, 32, 64, 128]:
        if period >= ncols:
            break
        is_periodic = True

        # Check if swizzle pattern becomes linear after `period` columns
        # i.e., addr(r, c + period) = addr(r, c) + period for all r, c
        num_rows_to_check = min(row_period, nrows)
        for row in range(num_rows_to_check):
            for col in range(min(period, ncols - period)):
                addr1 = get_addr(row, col)
                addr2 = get_addr(row, col + period)
                if addr2 - addr1 != period:
                    is_periodic = False
                    break
            if not is_periodic:
                break

        if is_periodic:
            col_period = period
            break

    return row_period, col_period


def _get_cutlass_colors():
    """
    Get CUTLASS-style color palette for bank visualization.
    Uses soft, distinguishable colors similar to CUTLASS documentation.
    32 colors for 32 banks.
    """
    colors = [
        "#E8E8E8",  # 0: light gray
        "#FF6B6B",  # 1: soft red
        "#FFA94D",  # 2: soft orange
        "#FFE066",  # 3: soft yellow
        "#69DB7C",  # 4: soft green
        "#38D9A9",  # 5: soft teal
        "#4DABF7",  # 6: soft blue
        "#9775FA",  # 7: soft purple
        "#F783AC",  # 8: soft pink
        "#FF8787",  # 9: coral
        "#FFD43B",  # 10: gold
        "#8CE99A",  # 11: mint
        "#66D9E8",  # 12: cyan
        "#748FFC",  # 13: indigo
        "#DA77F2",  # 14: violet
        "#FCC2D7",  # 15: light pink
        "#D0D0D0",  # 16: medium gray
        "#E03131",  # 17: darker red
        "#F76707",  # 18: darker orange
        "#F59F00",  # 19: amber
        "#37B24D",  # 20: darker green
        "#0CA678",  # 21: darker teal
        "#1C7ED6",  # 22: darker blue
        "#7048E8",  # 23: darker purple
        "#E64980",  # 24: magenta
        "#FA5252",  # 25: bright red
        "#FAB005",  # 26: bright yellow
        "#40C057",  # 27: bright green
        "#15AABF",  # 28: bright cyan
        "#4C6EF5",  # 29: bright blue
        "#AE3EC9",  # 30: bright purple
        "#F06595",  # 31: bright pink
    ]
    return colors


def plot_shared_layout(
    layout: T.Layout,
    save_directory: str = "./tmp",
    name: str = "shared_layout",
    colormap: str = "",
    verbose: bool = False,
    formats: str | list[str] = "png",
    num_banks: int = 32,
    show_base_tile_only: bool = True,
    element_bits: int = 16,
) -> None:
    """
    Plot a shared memory layout (T.Layout) showing swizzle pattern.

    Unlike Fragment which maps (i,j) -> (thread, local_idx),
    Layout maps (i,j) -> physical_address for shared memory.

    Parameters
    ----------
    layout : T.Layout
        The layout object that describes index swizzle mapping.
    save_directory : str, optional
        Directory where output images will be saved (default "./tmp").
    name : str, optional
        Base name of output files (default "shared_layout").
    colormap : str, optional
        Colormap for visualization (default "hsv").
    verbose : bool, optional
        Print additional mapping info (default False).
    formats : str | list[str], optional
        Output format(s): "png", "svg", "pdf", "all" (default "png").
    num_banks : int, optional
        Number of shared memory banks for coloring (default 32).
    show_base_tile_only : bool, optional
        If True, only show the minimal repeating tile (default True).
        For 128B swizzle, this is typically 8 rows x 64 cols.
    element_bits : int, optional
        Size of each element in bits (default 16 for bfloat16/float16).
        Used to calculate bank assignment correctly.
        - float32: 32 bits (1 element per bank)
        - bfloat16/float16: 16 bits (2 elements per bank)
        - int8: 8 bits (4 elements per bank)
    """
    import os
    import pathlib
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Get input shape
    input_shape = layout.get_input_shape()
    input_shape = [int(var) for var in input_shape]

    if verbose:
        print(f"Layout input shape: {input_shape}")
        print(f"Layout forward_index: {layout.get_forward_index()}")

    # Handle 2D or 3D layouts
    if len(input_shape) == 2:
        nrows, ncols = input_shape
        batch_size = 1
    elif len(input_shape) == 3:
        batch_size, nrows, ncols = input_shape
    else:
        raise ValueError(f"Expected 2D or 3D layout, got shape {input_shape}")

    # Calculate how many elements fit in one bank (each bank is 32 bits = 4 bytes)
    elements_per_bank = 32 // element_bits

    # Detect the base repeating tile size
    if show_base_tile_only:
        base_rows, base_cols = _detect_swizzle_period(layout, nrows, ncols)
        max_rows = min(base_rows, nrows)
        # Ensure we show at least num_banks columns (in bank units)
        # So we need at least num_banks * elements_per_bank element columns
        min_cols_for_banks = num_banks * elements_per_bank
        max_cols = max(min(base_cols, ncols), min(min_cols_for_banks, ncols))
        if verbose:
            print(f"Detected base tile: {base_rows} x {base_cols} elements")
            print(f"Adjusted to show {num_banks} banks: {max_rows} x {max_cols} elements")
    else:
        # Limit visualization size for very large layouts
        max_rows = min(nrows, 64)
        max_cols = min(ncols, 128)

    # Adjust columns: display by bank, not by element
    # For bfloat16 (16 bits): 2 elements per bank, so display_cols = max_cols // 2
    display_cols = max_cols // elements_per_bank
    display_rows = max_rows

    if verbose:
        print(f"Element size: {element_bits} bits, {elements_per_bank} elements per bank")
        print(f"Display shape: {display_rows} x {display_cols} (in bank units)")

    # Build physical address map (using first element of each bank group)
    addr_map = np.zeros((display_rows, display_cols), dtype=int)

    for row in range(display_rows):
        for col in range(display_cols):
            # Map to the first element in this bank group
            elem_col = col * elements_per_bank
            if len(input_shape) == 2:
                indices = [row, elem_col]
            else:
                indices = [0, row, elem_col]  # batch=0

            # Get physical address (take last dimension for multi-output layouts)
            phys_addr = layout.map_forward_index(indices)
            # Handle Array, list, or tuple - take the last element
            if hasattr(phys_addr, "__len__") and len(phys_addr) > 0:
                phys_addr = phys_addr[-1]
            # Convert element address to bank address
            addr_map[row, col] = int(phys_addr) // elements_per_bank

    # Calculate font size based on matrix size
    if display_cols > 64:
        font_size = 4
    elif display_cols > 32:
        font_size = 6
    elif display_cols > 16:
        font_size = 8
    else:
        font_size = 10

    # Create figure
    cell_size = 0.5
    fig_width = max(cell_size * display_cols + 2, 8)
    fig_height = max(cell_size * display_rows + 2, 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Color by bank assignment
    if colormap == "cutlass":
        cutlass_colors = _get_cutlass_colors()

        def get_color(bank):
            return cutlass_colors[bank % len(cutlass_colors)]
    else:
        cmap = plt.get_cmap("RdYlBu", lut=num_banks + 8)

        def get_color(bank):
            return cmap(bank)

    for row in range(display_rows):
        for col in range(display_cols):
            addr = addr_map[row, col]
            # addr is already in bank units
            bank = addr % num_banks
            color = get_color(bank)

            rect = patches.Rectangle((col, row), 1, 1, linewidth=0.3, edgecolor="gray", facecolor=color)
            ax.add_patch(rect)

            # Show physical address in cell
            ax.text(col + 0.5, row + 0.5, str(addr), ha="center", va="center", fontsize=font_size, color="black")

    # Row/col labels
    label_fontsize = min(font_size + 2, 10)
    for row in range(display_rows):
        ax.text(-0.4, row + 0.5, str(row), ha="center", va="center", fontsize=label_fontsize)
    for col in range(0, display_cols, max(1, display_cols // 8)):
        ax.text(col + 0.5, -0.4, str(col), ha="center", va="center", fontsize=label_fontsize)

    ax.set_xlim(-0.6, display_cols + 0.5)
    ax.set_ylim(-0.6, display_rows + 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Title with info
    title = f"{name}\nElement shape: {input_shape} ({element_bits}-bit)"
    title += f"\nBank units: {display_rows}×{display_cols} (each cell = {elements_per_bank} elements)"
    if show_base_tile_only and (nrows > max_rows or ncols > max_cols):
        title += f"\n(Base tile, repeats {nrows // max_rows}×{ncols // max_cols})"
    ax.set_title(title, fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend_patches = [
        patches.Patch(color="gray", label=f"Color = addr mod {num_banks} (bank)"),
        patches.Patch(color="white", label="Text = physical address"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(1.0, 1.15),
        ncols=2,
    )

    plt.tight_layout()

    # Save
    tmp_directory = pathlib.Path(save_directory)
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)

    # Parse formats
    if isinstance(formats, str):
        formats_str = formats.strip().lower()
        if formats_str == "all":
            formats_list = ["pdf", "png", "svg"]
        elif "," in formats_str:
            formats_list = [f.strip() for f in formats_str.split(",")]
        else:
            formats_list = [formats_str]
    else:
        formats_list = formats

    # Save figures
    if "pdf" in formats_list:
        pdf_path = tmp_directory / f"{name}.pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved shared layout pdf: {pdf_path}")

    if "png" in formats_list:
        png_path = tmp_directory / f"{name}.png"
        plt.savefig(png_path, bbox_inches="tight", transparent=False, dpi=150)
        print(f"Saved shared layout png: {png_path}")

    if "svg" in formats_list:
        svg_path = tmp_directory / f"{name}.svg"
        plt.savefig(svg_path, bbox_inches="tight", format="svg")
        print(f"Saved shared layout svg: {svg_path}")

    plt.close(fig)


def print_shared_layout_format(layout: T.Layout, name: str = "shared") -> None:
    """
    Print shared memory layout information in human-readable format.

    Parameters
    ----------
    layout : T.Layout
        The shared memory layout to format
    name : str
        Name of the buffer (e.g., "A_shared")
    """
    input_shape = layout.get_input_shape()
    output_shape = layout.get_output_shape()
    forward_idx = layout.get_forward_index()

    lines = [
        f"  Shape: {input_shape} -> {output_shape}",
        f"  Swizzle: {forward_idx}",
    ]
    print("\n".join(lines))
