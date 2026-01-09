#!/usr/bin/env python3
"""
Preview all matplotlib colormaps with 32 discrete colors.
Helps choose a good colormap for bank visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# All qualitative and other useful colormaps for discrete data
colormaps = [
    # Qualitative (best for discrete categories)
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    # Sequential (can work for discrete)
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    # Cyclic (good for wrapping values like bank IDs)
    "hsv",
    "twilight",
    "twilight_shifted",
    # Rainbow-like
    "rainbow",
    "jet",
    "turbo",
    "nipy_spectral",
    "gist_rainbow",
    # Other interesting ones
    "Spectral",
    "coolwarm",
    "RdYlBu",
    "RdYlGn",
    "PiYG",
    "PRGn",
]

num_colors = 32  # Number of banks

# Create figure
n_maps = len(colormaps)
fig, axes = plt.subplots(n_maps, 1, figsize=(14, n_maps * 0.6))

for ax, cmap_name in zip(axes, colormaps):
    # try:
    cmap = plt.get_cmap(cmap_name, lut=num_colors + 8)
    # except:
    #     cmap = plt.get_cmap(cmap_name)

    # Draw color swatches
    for i in range(num_colors):
        color = cmap(i)
        rect = patches.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="none")
        ax.add_patch(rect)

    ax.set_xlim(0, num_colors)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-0.5, 0.5, cmap_name, ha="right", va="center", fontsize=10, fontweight="bold")

plt.suptitle(f"Colormap Preview ({num_colors} discrete colors for bank visualization)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("tmp/colormap_preview.png", dpi=150, bbox_inches="tight")
print("Saved: tmp/colormap_preview.png")
plt.show()
