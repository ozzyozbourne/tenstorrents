// B10 kernel (BRISC on chip 0): fill one 4 KiB page in L1 with a pattern and send it over TT-Fabric
// (1 hop, 1D) to a DRAM page on the neighbouring chip. Adapted from
// tests/tt_metal/tt_fabric/benchmark/collectives/unicast/kernels/unicast_tx_writer_cb_to_dst.cpp.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"

using namespace tt::tt_fabric;

void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr uint32_t PAGE = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    size_t idx = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t seed = get_arg_val<uint32_t>(idx++);
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // The payload: word i = seed ^ (i * 0x9E3779B9), written by this RISC-V core into a CB page.
    constexpr uint32_t cb = tt::CBIndex::c_0;
    cb_reserve_back(cb, 1);
    const uint32_t l1 = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < PAGE / 4; i++) {
        p[i] = seed ^ (i * 0x9E3779B9u);
    }
    (void)p[PAGE / 4 - 1];

    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
    fabric_set_unicast_route<false>(header, 1);  // 1D route: deliver one hop away
    const auto dst = TensorAccessor(dst_args, dst_addr);
    header->to_noc_unicast_write(NocUnicastCommandHeader{dst.get_noc_addr(0, 0, 0)}, PAGE);

    sender.open<true>();
    sender.wait_for_empty_write_slot();
    sender.send_payload_without_header_non_blocking_from_address(l1, PAGE);
    sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();
    sender.close();
}
