// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/dprint.h"

void kernel_main() {
    // Read parameters from the kernel arguments
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);

    // The circular buffers to read the tiles into
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (This is most of the cases).
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Create address generators for the input buffers. Consider these the
    // pointers for interleaved buffers
    // Setting the page size to be tile_size_bytes works because we set it up
    // explicitly in host code. This is usually a good idea as it makes coding
    // easy.
    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, in0_addr);
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto in1 = TensorAccessor(in1_args, in1_addr);

    // Create Device 2.0 Noc and CircularBuffer objects
    Noc noc;
    CircularBuffer cb_in0_buf(cb_in0);
    CircularBuffer cb_in1_buf(cb_in1);
    DPRINT("cta: {} {} {} {}  in1 offset={}\n", get_compile_time_arg_val(0), get_compile_time_arg_val(1),
           get_compile_time_arg_val(2), get_compile_time_arg_val(3), in0_args.next_compile_time_args_offset());
    DPRINT("cb0 fifo_addr={:#x} size={} limit={:#x} page={}\n", get_local_cb_interface(cb_in0).fifo_limit - get_local_cb_interface(cb_in0).fifo_size,
           get_local_cb_interface(cb_in0).fifo_size, get_local_cb_interface(cb_in0).fifo_limit, get_local_cb_interface(cb_in0).fifo_page_size);

    // Loop over all the tiles and read them into the circular buffers
    for (uint32_t i = 0; i < n_tiles; i++) {
        // First make sure there is space in the circular buffers to be written to.
        if (i < 5) {
            DPRINT("i={} before reserve: wr={:#x} received={} acked={}\n", i, cb_in0_buf.get_write_ptr(),
                   *get_cb_tiles_received_ptr(cb_in0), *get_cb_tiles_acked_ptr(cb_in0));
        }
        cb_in0_buf.reserve_back(1);
        cb_in1_buf.reserve_back(1);  // wait until we have 1 free slot. This blocks if the
                                     // other kernels cannot consume the tiles fast enough.
                                     // Deciding how large the buffer should be is a tradeoff.
        // read the tile into the circular buffer
        noc.async_read(in0, cb_in0_buf, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});
        // We can overlap async reads and writes to reduce the data movement overhead.
        noc.async_read(in1, cb_in1_buf, tile_size_bytes, {.page_id = i}, {.offset_bytes = 0});

        noc.async_read_barrier();  // Wait until tile reads are done
        cb_in0_buf.push_back(1);
        if (i < 5) {
            DPRINT("i={} after push:     wr={:#x} received={} acked={}\n", i, cb_in0_buf.get_write_ptr(),
                   *get_cb_tiles_received_ptr(cb_in0), *get_cb_tiles_acked_ptr(cb_in0));
        }
        cb_in1_buf.push_back(1);  // mark the tiles as ready. From this point forward kernels
                                  // calling `cb_wait_front` will see this tile
    }
}
