#!/usr/bin/env python3
"""Chapter A18: upload the 3x40 tensor in ROW_MAJOR layout, then tilize and untilize it with device ops.

Shows the page shapes of both layouts (one row per page vs one tile per page). Deterministic.
"""
import torch, ttnn
device = ttnn.open_device(device_id=0)
x = (torch.arange(3*40, dtype=torch.float32).reshape(3, 40) + 1).to(torch.bfloat16)
d_rm = ttnn.from_torch(x, device=device)                     # ROW_MAJOR, on device
print("device RM  :", d_rm.layout, d_rm.shape, "padded", d_rm.padded_shape,
      "pages", d_rm.buffer_num_pages(), "x", d_rm.buffer_page_size(), "B")
d_t = ttnn.to_layout(d_rm, ttnn.TILE_LAYOUT)                 # tilize op on the device
print("device TILE:", d_t.layout, d_t.shape, "padded", d_t.padded_shape,
      "pages", d_t.buffer_num_pages(), "x", d_t.buffer_page_size(), "B")
p = ttnn.from_device(d_t).to_torch_with_padded_shape()
print("padding is zero:", bool((p[3:, :] == 0).all() and (p[:, 40:] == 0).all()), " data intact:", torch.equal(p[:3, :40], x))
back = ttnn.to_layout(d_t, ttnn.ROW_MAJOR_LAYOUT)            # untilize op on the device
print("untilized  :", back.layout, back.shape, "equal:", torch.equal(ttnn.to_torch(back), x))
ttnn.close_device(device)
