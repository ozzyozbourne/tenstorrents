#!/usr/bin/env python3
"""Chapter A18: a 3x40 bfloat16 tensor through from_torch -> to_layout(TILE) -> to_device, and what it became.

Prints the logical and padded shapes, the device pages, the zero padding and the round trip. Deterministic.
"""
import torch, ttnn
device = ttnn.open_device(device_id=0)
x = (torch.arange(3*40, dtype=torch.float32).reshape(3, 40) + 1).to(torch.bfloat16)   # values 1..120
t = ttnn.from_torch(x, dtype=ttnn.bfloat16)                     # default layout: ROW_MAJOR, on host
print("from_torch    :", t.layout, t.shape, "padded", t.padded_shape)
tt = ttnn.to_layout(t, ttnn.TILE_LAYOUT)                        # tilize (host)
print("to_layout TILE:", tt.layout, t.shape, "padded", tt.padded_shape, "tile", tt.tile)
d = ttnn.to_device(tt, device)                                  # copy to DRAM
print("on device     :", d.memory_config().buffer_type, d.memory_config().memory_layout)
print("pages", d.buffer_num_pages(), "x", d.buffer_page_size(), "bytes")
p = ttnn.from_device(d).to_torch_with_padded_shape()            # read back, keep the padding
print("padded view   :", tuple(p.shape), " nonzero:", int((p != 0).sum()), "of", p.numel())
print("row 0, cols 36..43:", p[0, 36:44].tolist())
print("row 3, cols 0..3  :", p[3, 0:4].tolist())
y = ttnn.to_torch(d)
print("to_torch      :", tuple(y.shape), "bit-identical:", torch.equal(x, y))
ttnn.close_device(device)
