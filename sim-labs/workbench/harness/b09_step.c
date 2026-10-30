// b09_step.c (lab B09): drive libttsim.so with no tt-metal at all.
// Writes an 8-instruction RISC-V program into the L1 of one Tensix core through the PCIe BAR,
// takes that core's BRISC out of reset, steps the clock one cycle at a time and reads the
// result back from L1.
//   gcc -O2 -o b09_step b09_step.c -ldl && ./b09_step ~/sim/wh/libttsim_wh.so [x y] [clocks]
// Wormhole only: the TLB layout below is Wormhole's (ttsim src/pcie.cpp:234-266).
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static void (*sim_init)(void), (*sim_exit)(void), (*sim_clock)(uint32_t);
static uint32_t (*cfg_rd)(uint32_t bdf, uint32_t off);
static void (*cfg_wr)(uint32_t bdf, uint32_t off, uint32_t data);
static void (*mem_rd)(uint64_t paddr, void *p, uint32_t size);
static void (*mem_wr)(uint64_t paddr, const void *p, uint32_t size);
static uint64_t bar0;

#define TLB_CFG_BASE 0x1FC00000u   // BAR0 offset of the TLB config registers, 8 bytes per TLB
#define TLB_1M_SIZE  0x100000u     // TLBs 0-155 are 1 MiB windows at BAR0 offset index * 1 MiB

// Point 1 MiB TLB `index` at NoC endpoint (x, y), local address `addr` (rounded down to 1 MiB).
// Config layout for a 1 MiB TLB: bits 15:0 = addr >> 20, bits 21:16 = x, bits 27:22 = y.
static uint64_t tlb_map(uint32_t index, uint32_t x, uint32_t y, uint64_t addr) {
    uint64_t cfg = (addr >> 20) | ((uint64_t)(x | (y << 6)) << 16);
    mem_wr(bar0 + TLB_CFG_BASE + 8 * index, &cfg, 8);
    return bar0 + (uint64_t)index * TLB_1M_SIZE;   // host address of the window
}
static uint32_t rd32(uint64_t p) { uint32_t v; mem_rd(p, &v, 4); return v; }
static void wr32(uint64_t p, uint32_t v) { mem_wr(p, &v, 4); }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s libttsim.so [x y] [clocks]\n", argv[0]); return 2; }
    uint32_t x = argc > 3 ? atoi(argv[2]) : 18, y = argc > 3 ? atoi(argv[3]) : 18;  // virtual coords of worker (0,0)
    uint32_t clocks = argc > 4 ? atoi(argv[4]) : 30;
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
    sim_init = dlsym(h, "libttsim_init");  sim_exit = dlsym(h, "libttsim_exit");
    sim_clock = dlsym(h, "libttsim_clock");
    cfg_rd = dlsym(h, "libttsim_pci_config_rd32"); cfg_wr = dlsym(h, "libttsim_pci_config_wr32");
    mem_rd = dlsym(h, "libttsim_pci_mem_rd_bytes"); mem_wr = dlsym(h, "libttsim_pci_mem_wr_bytes");

    sim_init();
    // 1. PCI config space: find BAR0 and make sure memory decode (command register bit 1) is on.
    //    ttsim already sets it at reset, "as left by firmware enumeration" (src/pcie.cpp:172).
    bar0 = ((uint64_t)cfg_rd(0, 0x14) << 32) | (cfg_rd(0, 0x10) & ~0xFu);
    cfg_wr(0, 0x04, cfg_rd(0, 0x04) | 2);
    printf("BAR0 at 0x%llx, command=0x%x\n", (unsigned long long)bar0, cfg_rd(0, 0x04) & 0xFFFF);

    // 2. Two TLB windows onto core (x, y): its L1 from address 0, and its register page at 0xFFB00000.
    uint64_t l1 = tlb_map(0, x, y, 0);
    uint64_t regs = tlb_map(1, x, y, 0xFFB00000);
    uint64_t soft_reset = regs + 0x121B0;              // RISCV_DEBUG_REG_SOFT_RESET_0 = 0xFFB121B0

    // 3. The program (assembled with riscv-tt-elf-as -march=rv32i; no compressed instructions,
    //    because ttsim treats a 32-bit word whose low bits are not 0b11 as a Tensix instruction).
    static const uint32_t prog[] = {
        0x10000293,  // 0x00  li   t0, 0x100      result address
        0x00000513,  // 0x04  li   a0, 0          running sum
        0x00500593,  // 0x08  li   a1, 5          counter
        0x00b50533,  // 0x0c  add  a0, a0, a1     loop:
        0x00a2a023,  // 0x10  sw   a0, 0(t0)
        0xfff58593,  // 0x14  addi a1, a1, -1
        0xfe059ae3,  // 0x18  bnez a1, loop
        0x0000006f,  // 0x1c  j    .              park
    };
    mem_wr(l1, prog, sizeof prog);                    // BRISC's reset PC is 0 (ttsim src/tile.cpp:1108)
    wr32(l1 + 0x100, 0xdeadbeef);                     // poison the result word
    printf("L1[0x0..0x1c] written, L1[0x100] = 0x%08x, SOFT_RESET_0 = 0x%05x\n", rd32(l1 + 0x100), rd32(soft_reset));

    // 4. Release BRISC only: 0x800 is BRISC's reset bit; TRISC0-2 (0x1000/0x2000/0x4000) and NCRISC (0x40000) stay held.
    wr32(soft_reset, rd32(soft_reset) & ~0x800u);
    printf("SOFT_RESET_0 = 0x%05x (BRISC running)\n", rd32(soft_reset));

    // 5. Step one clock at a time and watch the result word change.
    uint32_t last = rd32(l1 + 0x100);
    for (uint32_t c = 1; c <= clocks; c++) {
        sim_clock(1);
        uint32_t v = rd32(l1 + 0x100);
        if (v != last) printf("after clock %2u: L1[0x100] = %u\n", c, v);
        last = v;
    }
    printf("final: L1[0x100] = %u (expected 5+4+3+2+1 = 15)\n", last);
    sim_exit();
    return last == 15 ? 0 : 1;
}
