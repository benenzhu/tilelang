#!/usr/bin/env python3
"""
Visualize Swizzle Layout for Shared Memory Bank Conflict Avoidance
"""
# %%
# Display as SVG in Jupyter (sharper, scalable)
# %config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def swizzle(addr: int, S: int, B: int, M: int) -> int:
    """
    XOR-based swizzle function.

    Args:
        addr: Linear address (row * stride + col)
        S: Right shift amount (typically log2(stride))
        B: Number of bits to swizzle (log2(max_phase))
        M: Base position of bits to XOR (log2(vec_size))

    Returns:
        Swizzled address
    """
    Bmask = ((1 << B) - 1) << M
    return ((addr >> S) & Bmask) ^ addr


def get_bank_color(bank: int, num_banks: int = 32):
    """
    Get color based on bank assignment.
    Supports up to 32 banks with distinct colors.
    """
    import matplotlib.cm as cm

    # Use a colormap that provides good distinction for 32 colors
    # 'hsv' gives rainbow colors, 'tab20' + 'tab20b' can combine for more
    # if num_banks <= 20:
    #     cmap = cm.get_cmap('tab20', num_banks)
    #     return cmap(bank % num_banks)
    # else:
    # For 32 banks, use HSV colormap for rainbow effect
    # Shift hue slightly to avoid red-red wrap
    cmap = cm.get_cmap("hsv", num_banks + 1)
    return cmap(bank % num_banks)


def draw_grid(
    ax, data: np.ndarray, title: str, show_bank_colors: bool = True, stride: int = 8, vec_size: int = 2, num_banks: int = 32, font_size=9
):
    """
    Draw a grid visualization.

    Args:
        ax: Matplotlib axis
        data: 2D array of values to display
        title: Title for the plot
        show_bank_colors: Whether to color by bank
        stride: Number of columns (for bank calculation)
        vec_size: Vector size for bank grouping
    """
    rows, cols = data.shape

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Draw column headers
    for col in range(cols):
        ax.text(col + 0.5, -0.3, str(col), ha="center", va="center", fontsize=10)

    # Draw row headers
    for row in range(rows):
        ax.text(-0.3, row + 0.5, str(row), ha="center", va="center", fontsize=10)

    # Draw cells
    for row in range(rows):
        for col in range(cols):
            value = data[row, col]

            # Determine bank color (group by vec_size columns)
            if show_bank_colors:
                bank = (col // vec_size) % num_banks
                bank = value % num_banks
                color = get_bank_color(bank, num_banks)
            else:
                color = "white"

            # Draw rectangle
            rect = patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor="black", facecolor=color)
            ax.add_patch(rect)

            # Draw value
            ax.text(col + 0.5, row + 0.5, str(int(value)), ha="center", va="center", fontsize=font_size)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def create_swizzle_visualization(
    rows: int = 8, cols: int = 8, S: int = 3, B: int = 2, M: int = 1, vec_size: int = 2, num_banks: int = 32, font_size=12
):
    """
    Create before/after swizzle visualization.

    Args:
        rows: Number of rows
        cols: Number of columns
        S: Swizzle parameter (log2(stride))
        B: Swizzle parameter (bits to XOR)
        M: Swizzle parameter (base position)
        vec_size: Vector size for coloring
    """
    # Create original layout (linear indices)
    original = np.arange(rows * cols).reshape(rows, cols)

    # Create swizzled layout
    swizzled = np.zeros((rows, cols), dtype=int)
    for row in range(rows):
        for col in range(cols):
            addr = row * cols + col
            swizzled_addr = swizzle(addr, S, B, M)
            # The swizzled address determines where this element goes
            # Or we can show what element ends up at each position
            swizzled[row, col] = swizzle(addr, S, B, M)

    # Alternative: show what original index maps to each swizzled position
    swizzled_content = np.zeros((rows, cols), dtype=int)
    for row in range(rows):
        for col in range(cols):
            addr = row * cols + col
            swizzled_addr = swizzle(addr, S, B, M)
            new_row = swizzled_addr // cols
            new_col = swizzled_addr % cols
            if 0 <= new_row < rows and 0 <= new_col < cols:
                swizzled_content[new_row, new_col] = addr

    # Create figure
    if cols > 16:
        fig, axes = plt.subplots(2, 1, figsize=(7, 14))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    draw_grid(
        axes[0], original, f"Original Layout\n(Linear: row * {cols} + col)", vec_size=vec_size, num_banks=num_banks, font_size=font_size
    )
    # Use swizzled_content: shows what original value is at each swizzled position
    # This correctly demonstrates M parameter: M=1 -> 2 consecutive, M=2 -> 4 consecutive
    draw_grid(
        axes[1],
        swizzled,
        f"Swizzled Layout\nswizzle<B={B}, M={M}, S={S}>(addr)",
        vec_size=vec_size,
        num_banks=num_banks,
        font_size=font_size,
    )

    plt.suptitle(f"XOR Swizzle Visualization ({rows}×{cols} matrix)\nBmask = ((1<<{B})-1)<<{M} = {((1 << B) - 1) << M:#06b}", fontsize=15)
    plt.tight_layout()

    return fig


def create_bank_conflict_demo(rows: int = 8, cols: int = 8):
    """
    Demonstrate bank conflict avoidance with swizzle.
    Shows which bank each element maps to.
    """
    num_banks = 32
    bank_width = 4  # 4 bytes per bank
    elem_size = 4  # float = 4 bytes

    # Banks for original layout
    original_banks = np.zeros((rows, cols), dtype=int)
    for row in range(rows):
        for col in range(cols):
            addr = row * cols + col
            byte_addr = addr * elem_size
            bank = (byte_addr // bank_width) % num_banks
            original_banks[row, col] = bank

    # Banks for swizzled layout (S=3, B=3, M=0 for 8-wide)
    S, B, M = 3, 3, 0
    swizzled_banks = np.zeros((rows, cols), dtype=int)
    for row in range(rows):
        for col in range(cols):
            addr = row * cols + col
            swizzled_addr = swizzle(addr, S, B, M)
            byte_addr = swizzled_addr * elem_size
            bank = (byte_addr // bank_width) % num_banks
            swizzled_banks[row, col] = bank

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Custom drawing for bank visualization
    for ax_idx, (data, title) in enumerate(
        [
            (original_banks, "Original: Bank Assignment\n(Same column = Same bank = CONFLICT!)"),
            (swizzled_banks, "Swizzled: Bank Assignment\n(Same column = Different banks = NO CONFLICT!)"),
        ]
    ):
        ax = axes[ax_idx]
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Draw headers
        for col in range(cols):
            ax.text(col + 0.5, -0.3, f"Col {col}", ha="center", va="center", fontsize=9)
        for row in range(rows):
            ax.text(-0.5, row + 0.5, f"Row {row}", ha="center", va="center", fontsize=9)

        # Draw cells with bank-based colors
        cmap = plt.cm.get_cmap("tab20", 32)
        for row in range(rows):
            for col in range(cols):
                bank = data[row, col]
                color = cmap(bank % 20)

                rect = patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor="black", facecolor=color)
                ax.add_patch(rect)
                ax.text(col + 0.5, row + 0.5, f"B{bank}", ha="center", va="center", fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle("Shared Memory Bank Conflict Visualization", fontsize=14)
    plt.tight_layout()

    return fig


"""
template <uint32_t S, uint32_t B, uint32_t M>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
    constexpr auto Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
}
"""
if __name__ == "__main__":
    # Example 1: Simple 8x8 swizzle visualization (like the images)
    print("Creating swizzle visualization...")

    # Parameters for 8x8 matrix with 2-element vectors
    # S=3 (log2(8)), B=2 (2 bits to XOR), M=1 (start at bit 1)
    # fig1 = create_swizzle_visualization(rows=8, cols=8, B=3, M=0,  S=3, vec_size=2, num_banks=8, font_size=25)
    """
    constexpr auto Bmask = ((1 << 3) - 1) << 0 = 7
    return ((addr >> 3) & 7) ^ addr
    """

    # fig1 = create_swizzle_visualization(rows=8, cols=8, B=3, M=1,  S=3, vec_size=2, num_banks=8, font_size=25)
    if False:  # 加宽矩阵, (8, 32) 8个bank.
        # fig1 = create_swizzle_visualization(rows=8, cols=32, B=3, M=0,  S=3, vec_size=2, num_banks=8, font_size=6) # 这个竖着访问会有冲突...
        ### 矩阵加宽了，增加 S... 对应 linear layout里面的 stride.
        ### 但是为啥不用加 M 呢？
        fig1 = create_swizzle_visualization(rows=8, cols=32, B=3, M=0, S=5, vec_size=2, num_banks=8, font_size=6)  # 这个竖着访问会有冲突...
    if False:  # 访问多列.
        # 请求一个 [8,1] -> [4,2] 的矩阵
        fig1 = create_swizzle_visualization(rows=8, cols=8, B=2, M=1, S=2, vec_size=2, num_banks=8, font_size=25)  # 这个竖着访问会有冲突...

    if True:  # 访问多列.
        # 请求一个 [8,1] -> [4,2] 的矩阵
        # fig1 = create_swizzle_visualization(rows=16, cols=8, B=3, M=1,  S=2, vec_size=2, num_banks=8, font_size=9) # 这个竖着访问会有冲突...
        """
        constexpr auto Bmask = ((1 << 3) - 1) << M = 14;
        return ((addr >> 2) & 14) ^ addr.
        """
        # fig1 = create_swizzle_visualization(rows=16, cols=8, B=3, M=1,  S=2, vec_size=2, num_banks=8, font_size=9) # 这个竖着访问会有冲突...
        # fig1 = create_swizzle_visualization(rows=16, cols=8, B=3, M=1,  S=3, vec_size=2, num_banks=8, font_size=9) # 这个竖着访问会有冲突...
        # fig1 = create_swizzle_visualization(rows=32, cols=32, B=3, M=2,  S=5, vec_size=2, num_banks=32, font_size=6) # 这个竖着访问会有冲突...
        # fig1 = create_swizzle_visualization(rows=16, cols=16, B=2, M=2,  S=4, vec_size=2, num_banks=32, font_size=6) # 这个竖着访问会有冲突...
        # fig1 = create_swizzle_visualization(rows=16, cols=16, B=2, M=2,  S=3, vec_size=2, num_banks=32, font_size=6) # 这个竖着访问会有冲突...
        # fig1 = create_swizzle_visualization(rows=16, cols=32, B=4, M=0,  S=4, vec_size=2, num_banks=16, font_size=4) # 这个竖着访问会有冲突...
    if True:
        # fig1 = create_swizzle_visualization(rows=8, cols=8, B=3, M=1,  S=3, vec_size=2, num_banks=16, font_size=25)

        fig1 = create_swizzle_visualization(rows=8, cols=8, B=3, M=0, S=3, vec_size=2, num_banks=8, font_size=25)
        fig1 = create_swizzle_visualization(rows=8, cols=16, B=3, M=0, S=4, vec_size=2, num_banks=8, font_size=16)
    # fig1.savefig('swizzle_8x8.png', dpi=300, bbox_inches='tight')
    # print("Saved: swizzle_8x8.png")

    # Example 2: 32x32 matrix (more realistic for GEMM)
    # S=5 (log2(32)), B=3 (8 phases), M=2 (4-element vectors)
    # fig2 = create_swizzle_visualization(rows=8, cols=32, S=5, B=3, M=2, vec_size=4)
    # fig2.savefig('swizzle_8x32.png', dpi=300, bbox_inches='tight')
    # print("Saved: swizzle_8x32.png")

    # # Example 3: Bank conflict demonstration
    # fig3 = create_bank_conflict_demo()
    # fig3.savefig('bank_conflict_demo.png', dpi=300, bbox_inches='tight')
    # print("Saved: bank_conflict_demo.png")

    # print("\nDone! Check the generated PNG files.")

    # Show plots if running interactively
    plt.show()


# %%

# %%

# %%

# %%

# %%

# %%
#
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
