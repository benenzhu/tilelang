#pragma once

#include "common.h"

namespace tl {
TL_DEVICE unsigned int chiplet_transform_chunked(unsigned int workgroup_id,
                                                 unsigned int num_workgroups,
                                                 int num_xcds, int chunk_size) {
  int xcd = workgroup_id % num_xcds;
  int block = num_xcds * chunk_size;
  int limit = (num_workgroups / block) * block;

  // Tail workgroups that don't fill a complete block are left unchanged
  if (workgroup_id >= (unsigned int)limit)
    return workgroup_id;

  int local_pid = workgroup_id / num_xcds;
  int chunk_idx = local_pid / chunk_size;
  int pos_in_chunk = local_pid % chunk_size;

  return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}

template <int panel_width, bool use_amd_remap = false>
TL_DEVICE dim3 rasterization2DRow() {
  if constexpr (!use_amd_remap) {
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const unsigned int grid_size = gridDim.x * gridDim.y;
    const unsigned int panel_size = panel_width * gridDim.x;
    const unsigned int panel_offset = block_idx % panel_size;
    const unsigned int panel_idx = block_idx / panel_size;
    const unsigned int total_panel = ceil_div(grid_size, panel_size);
    const unsigned int stride =
        panel_idx + 1 < total_panel
            ? panel_width
            : (grid_size - panel_idx * panel_size) / gridDim.x;
    const unsigned int col_idx = (panel_idx & 1)
                                     ? gridDim.x - 1 - panel_offset / stride
                                     : panel_offset / stride;
    const unsigned int row_idx =
        panel_offset % stride + panel_idx * panel_width;
    return {col_idx, row_idx, blockIdx.z};
  } else {
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    const unsigned int grid_size = gridDim.x * gridDim.y;

    // Step 1: XCD remap
    unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    block_idx = chiplet_transform_chunked(block_idx, grid_size, 8,
                                          panel_width * panel_width);

    // Step 2: Panel (L2) swizzle
    const unsigned int panel_size = panel_width * gridDim.x;
    const unsigned int panel_offset = block_idx % panel_size;
    const unsigned int panel_idx = block_idx / panel_size;
    const unsigned int total_panel = ceil_div(grid_size, panel_size);
    const unsigned int stride =
        panel_idx + 1 < total_panel
            ? panel_width
            : (grid_size - panel_idx * panel_size) / gridDim.x;
    const unsigned int col_idx = (panel_idx & 1)
                                     ? gridDim.x - 1 - panel_offset / stride
                                     : panel_offset / stride;
    const unsigned int row_idx =
        panel_offset % stride + panel_idx * panel_width;
    return {col_idx, row_idx, blockIdx.z};
  }
}

template <int panel_width> TL_DEVICE dim3 rasterization2DColumn() {
  auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const unsigned int grid_size = gridDim.x * gridDim.y;
  const unsigned int panel_size = panel_width * gridDim.y;
  const unsigned int panel_offset = block_idx % panel_size;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int total_panel = ceil_div(grid_size, panel_size);
  const unsigned int stride =
      panel_idx + 1 < total_panel
          ? panel_width
          : (grid_size - panel_idx * panel_size) / gridDim.y;
  const unsigned int row_idx = (panel_idx & 1)
                                   ? gridDim.y - 1 - panel_offset / stride
                                   : panel_offset / stride;
  const unsigned int col_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

} // namespace tl
