// B06 reader (BRISC, NoC 0): stream tiles i = 0..n-1 of A and B from DRAM into CBs c_0 and c_1.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);   // DRAM address of A (runtime arg 0)
    const uint32_t b_addr = get_arg_val<uint32_t>(1);   // DRAM address of B (runtime arg 1)
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);  // how many tiles to stream

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    // Compile-time args 0.. describe A, then B; chain the offsets (A11).
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_a, 1);                        // block until c_0 has a free page
        cb_reserve_back(cb_b, 1);                        // ... and c_1 too
        noc_async_read_page(i, a, get_write_ptr(cb_a));  // start copying tile i of A into that page
        noc_async_read_page(i, b, get_write_ptr(cb_b));  // both reads are in flight together
        noc_async_read_barrier();                        // the bytes are in L1 now
        cb_push_back(cb_a, 1);                           // publish: received += 1, wr advances
        cb_push_back(cb_b, 1);
    }
}
