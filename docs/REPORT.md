# NVFP4 Ternary TPU — Engineering Report

**Date:** 2026-07-14
**Target:** Tiny Tapeout TTSKY26c, 1×1 tile (~167×108 µm), SkyWater SKY130A
**Top module:** `tt_um_kashif_fp4_ternary_tpu`
**Clock:** 50 MHz
**Result:** 9 cocotb tests passing (RTL + GitHub Actions CI)

---

## 1. Objective

Demonstrate the first open-silicon accelerator using the NVFP4 (E2M1)
4-bit floating-point weight format — the same format used in NVIDIA
Blackwell Tensor Cores — combined with ternary {-1, 0, +1} activations
that eliminate the hardware multiplier entirely.

## 2. Design Metrics

| Metric | Value |
|--------|-------|
| Array size | 4×4 = 16 PEs |
| Weight format | E2M1, 4-bit (element type shared by NVFP4 and MXFP4; they differ in host-side block scaling — NVFP4: 16-elem blocks with E4M3 scales, MXFP4: 32-elem blocks with E8M0) |
| Activation format | Ternary {-1, 0, +1}, 2-bit |
| Multiplier | None (MUX-add) |
| FFs per PE | 14 (4-bit weight + 10-bit accumulator) |
| Gates per PE | ~74 (E2M1 LUT + add/sub + accumulator) |
| Total PEs | 16 |
| Effective MACs/cycle | 16 |
| Accumulator | 10-bit signed (max ±192, 9 bits + margin) |
| Activation functions | Host-side (correct only after cross-tile accumulation) |
| Tile size | 1×1 |
| I/O protocol | Direct pin streaming (no SPI) |
| Total latency | ~57 cycles (8 load + 18 compute + 32 output) |

## 3. Architecture

### Dataflow: Weight-Stationary Broadcast

Weights are loaded once into each PE's 4-bit register and held for the
entire compute block. Activations are broadcast down columns — all PEs
in a column receive the same ternary value each cycle.

```
C[i][j] = W[i][j] × sum_k(act_k[j])
```

This is a matrix-vector multiply. For matrix-matrix, the host chains
multiple calls (see MNIST application in info.md).

### PE Design

```
                    ┌─────────┐
  weight_in[3:0] ──►│ weight  │── weight_code[3:0]
                    │ reg (4b)│
                    └─────────┘
                         │ combinational decode
                    ┌─────────┐
                    │ E2M1    │── w_mag[3:0], w_sign
                    │ LUT     │
                    └─────────┘
                         │
  act_in[1:0] ──────────┤
                    ┌─────────┐
                    │ MUX-add │── product (±w_val or 0)
                    └─────────┘
                         │
                    ┌─────────┐
  acc_clear ────────►│ 10-bit  │── acc_out[9:0]
  act_valid ────────►│ acc reg │
                    └─────────┘
```

Only 2 registers per PE:
- `weight_code` (4-bit, no reset — loaded before use)
- `acc` (10-bit signed, async reset — must be 0 at start of each block)

Weight magnitude decoded combinationally via 8-entry LUT:
`{exp[1:0], mant}` → `|value|×2` ∈ {0, 1, 2, 3, 4, 6, 8, 12}

### Control FSM

4-state FSM encoded in `uio_in[7:6]`:
- **IDLE** (00): hold, status output
- **LOAD** (01): 8 cycles, 2 weights/cycle via `ui_in`
- **COMPUTE** (10): 18 cycles (1 clear+MAC + 16 MAC + 1 drain), then snapshot
- **OUTPUT** (11): 32 cycles, 16 results × 2 bytes

ASIC optimizations (per TT HDL guide /hdl/fpga_vs_asic/):
- `weight_load`, `weight_a/b`, `load_idx` combinational pass-through
  (saves ~11 FFs — mode-gated, not state-dependent)
- `act0_r..act3_r`, `act_valid_r`, `acc_clear_r` registered for timing
- `acc_snap[0:15]` = 160 FFs for latched results
- Activation functions host-side by design

### Output Path

`(* keep = "true" *)` FFs for `uio_oe` and `uio_out` prevent Yosys
from tying them to `conb` cells whose internal pulldown causes magic
LVS to short pins to VGND (pattern adopted from Mini-TPU `tt_um_tpu.v`).

## 4. Comparison with Other TT TPU Designs

| Feature | Mini-TPU (IEEE) | PFW TPU | **This design** |
|---------|-----------------|---------|-----------------|
| Array | 3×3 = 9 PEs | 2×2 = 4 PEs | **4×4 = 16 PEs** |
| Weight format | INT4 | INT8 | **E2M1 (NVFP4)** |
| Activation | INT4 | INT8 | **Ternary {-1,0,+1}** |
| Multiplier | 4×4 hardware | 8×8 hardware | **None (MUX-add)** |
| PE registers | 3 | 3 | **2** |
| PE gates | ~200+ | ~400+ | **~74** |
| I/O | SPI (12-bit instr) | Parallel streaming | **Parallel streaming** |
| Weight load | ~108+ cycles | 8 cycles | **8 cycles** |
| Total latency | ~220+ cycles | ~41 cycles | **~57 cycles** |
| MACs/cycle | 0.04 | 0.10 | **0.28** |
| Tile | 1×1 | 1×2 | **1×1** |
| ReLU | No | Yes | **No (host-side by design)** |
| Numerics | Educational INT4 | INT8 | **NVFP4 (Blackwell)** |

## 5. Verification

### Test Suite (9 tests, all passing)

| Test | What it verifies | Pattern source |
|------|-----------------|---------------|
| `test_basic_mac` | Known weights + column-specific activations | Original |
| `test_varied_weights` | All 8 E2M1 magnitudes | Original |
| `test_int4_mode` | INT4 decode mode + mode latching | Original |
| `test_zero_weights` | Zero weights → zero output | Mini-TPU pattern |
| `test_zero_activations` | Zero activations → zero output | Mini-TPU pattern |
| `test_max_accumulation` | Boundary: weight=6.0, 16×+1 = 192 | Mini-TPU pattern |
| `test_negative_weights` | Negative E2M1 with mixed activations | Original |
| `test_random` | 50 seeded random trials | Mini-TPU pattern |
| `test_back_to_back` | Consecutive matmuls, state reset | Mini-TPU pattern |

### Gate-Level Test Infrastructure

- `gl_preheat()`: runs zero-weight matmul to flush X from pipeline regs
- `_safe_int()`: X-safe LogicArray reads for GL simulation
- `GL_TEST` environment variable reduces random trial count
- Testbench `tb.v` includes `GL_TEST` ifdefs for `VPWR`/`VGND`

### Known Limitations

- Golden model mirrors PE structure (not fully independent — improvement planned)
- Weight register has no reset (requires `gl_preheat()` in GL sim)
- Matrix-vector only (not true GEMM — broadcast architecture)
- No on-chip weight memory (weights re-streamed each call)

## 6. HDL Guide Compliance

Per [tinytapeout.com/hdl/important/](https://tinytapeout.com/hdl/important/):

- ✅ Top module named `tt_um_kashif_fp4_ternary_tpu` (unique, with username)
- ✅ Exact module port definition matching TT template
- ✅ No `initial` blocks; explicit `rst_n` reset
- ✅ All outputs assigned (`uo_out`, `uio_out`, `uio_oe`)
- ✅ `(* keep *)` FFs for unused output pins (LVS safety)
- ✅ `_unused` wire for unused inputs
- ✅ `default_nettype none`
- ✅ No `config.tcl` modifications
- ✅ Design not optimised away (all 16 accumulators read out)

Per [tinytapeout.com/hdl/fpga_vs_asic/](https://tinytapeout.com/hdl/fpga_vs_asic/):

- ✅ Minimal flops (combinational weight decode, only 2 regs/PE)
- ✅ Deeper combinational logic with fewer pipeline registers
- ✅ Async reset only on functionally-required registers (accumulator)

## 7. File Structure

```
src/
  project.v              # Top-level TT module
  pe.v                   # Processing element: E2M1 weight + ternary MAC
  systolic_array_4x4.v   # 4×4 weight-stationary grid
  control_fsm.v          # IDLE/LOAD/COMPUTE/OUTPUT state machine
docs/
  info.md                # Datasheet (protocol, E2M1 table, application)
  REPORT.md              # This file
test/
  tb.v                   # Verilog testbench (GL_TEST compatible)
  test.py                # 9 cocotb tests
  Makefile               # icarus/cocotb build
DESIGN_NOTES.md           # Int7+1 structured sparsity idea
info.yaml                 # TT metadata: 1x1 tile, 50MHz, SKY130A
```

## 8. Design Rationale (from Roune's document)

Key principles from Roune's "Designing AI Chip Software and Hardware" (2026)
that validate our design choices:

- **"FP4 for activations is very challenging"** (Numerics section): We use
  ternary {-1, 0, +1} activations instead of FP4 — avoiding the challenge
  while keeping activations ultra-low-precision.

- **"Integer summation is cheaper than FP summation"** (Numerics nerd box):
  Our 10-bit signed integer accumulator is cheaper than FP32 accumulation.
  Roune notes: "The product of two FP4's can be represented in an int8 using
  fixed point. So it might actually be cheaper and also more accurate to sum
  FP4 products in integer values instead of floating point."

- **"8 bits is always sufficient for inference"** (Numerics section): Our
  E2M1 weights are 4-bit and ternary activations are 2-bit — well within
  the sufficient range when combined with block-scale dequantization.

- **"Systolic array numerics do not need to be the same as scalar and vector
  numerics"** (Numerics section): Our ASIC uses E2M1+ternary while the host
  RP2040 applies E4M3/FP32 scales in software — exactly the split Roune
  recommends.

- **"Not all systolic arrays are made equal"** (Numerics section): By
  eliminating the hardware multiplier entirely (ternary activations), our
  PEs are ~3× smaller than INT4 designs, allowing 16 PEs in a 1×1 tile.

## 9. References

- [NVFP4: NVIDIA Blackwell format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [OCP Microscaling (MX) Formats spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Microscaling Data Formats for Deep Learning (arXiv:2310.10537)](https://arxiv.org/abs/2310.10537)
- [Ternary Weight Networks (Li et al. 2016, arXiv:1605.04711)](https://arxiv.org/abs/1605.04711)
- [Binarized Neural Networks (Hubara et al. 2016, arXiv:1602.02830)](https://arxiv.org/abs/1602.02830)
- [vLLM Qwix: NVFP4 quantization on TPU](https://github.com/vllm-project/tpu-inference)
- [PFW TPU](https://github.com/wangantian/pfw_tpu) — INT8 2×2 systolic, TT SKY26b
- [Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi) — INT4 3×3 systolic, TT SKY26b
- [FP8 MAC Unit](https://github.com/chatelao/ttihp-fp8-mul) — OCP MX streaming MAC, TT IHP26a
- [TT HDL Guide](https://tinytapeout.com/hdl/) — FPGA-to-ASIC considerations
- [TT Tech Specs](https://tinytapeout.com/specs/) — Clock, GPIO, analog, memory constraints
- [Companion design: Int7+1 Sparse TPU](https://github.com/kashif/kashif-int7-sparse-tpu)
