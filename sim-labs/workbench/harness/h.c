#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
    void (*init)(void) = dlsym(h, "libttsim_init");
    void (*fini)(void) = dlsym(h, "libttsim_exit");
    uint32_t (*cfg_rd)(uint32_t, uint32_t) = dlsym(h, "libttsim_pci_config_rd32");
    void (*clk)(uint32_t) = dlsym(h, "libttsim_clock");
    init();
    printf("cfg[0] = 0x%08x (device<<16 | vendor)\n", cfg_rd(0, 0));
    clk(1000);
    fini();
    return 0;
}
