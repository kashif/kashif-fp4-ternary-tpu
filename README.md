![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# NVFP4 Ternary TPU

A 4x4 weight-stationary systolic array for matrix-vector multiplication using **NVFP4 (E2M1) 4-bit floating-point weights** with **ternary {-1, 0, +1} activations** — the first open-silicon NVFP4-format inference accelerator.

- [Read the documentation for project](docs/info.md)
- [Design notes & future ideas](DESIGN_NOTES.md)

## Why This Design Is Innovative

### No hardware multiplier

With ternary activations, the multiply-accumulate (MAC) operation reduces to **add / subtract / no-op** — no multiplier is needed:

| Activation | Operation | Hardware |
|-----------|-----------|----------|
| +1 | `acc += weight` | adder |
| -1 | `acc -= weight` | subtractor |
| 0 | `acc` unchanged | nothing |

This eliminates the most expensive cell in each processing element (PE). A conventional 4-bit INT multiplier (~200+ gates) is replaced by a MUX + adder (~60 gates), allowing 16 PEs (4x4) to fit in a 1x1 tile where a conventional design fits only 9 (3x3).

### NVFP4 format — same as NVIDIA Blackwell

The E2M1 weight format (1 sign + 2 exponent + 1 mantissa) is identical to **NVFP4** and **MXFP4** — the 4-bit floating-point formats used in NVIDIA Blackwell Tensor Cores and the OCP Microscaling (MX) spec. The chip outputs raw integer accumulators; the host RP2040 applies the E4M3 block scale and FP32 tensor scale in software (NVFP4 dequantization).

### Comparison with existing TinyTapeout TPU designs

| Feature | Mini-TPU (IEEE_ttsky_mini_tpu_spi) | PFW TPU | **This design** |
|---------|-------------------------------------|---------|-----------------|
| Array size | 3x3 = 9 PEs | 2x2 = 4 PEs | **4x4 = 16 PEs** |
| Weight format | INT4 | INT8 | **E2M1 / NVFP4** |
| Activation format | INT4 | INT8 | **Ternary {-1,0,+1}** |
| Multiply hardware | 4x4 integer multiplier | 8x8 integer multiplier | **MUX + adder (no multiplier)** |
| PE register count | 3 | 3 | **2** |
| PE gate estimate | ~200+ | ~400+ | **~60** |
| I/O protocol | SPI (12-bit instructions) | Parallel (8-bit streaming) | **Parallel (8-bit streaming)** |
| Weight load time | ~108+ cycles | 8 cycles | **8 cycles** |
| Total matmul latency | ~220+ cycles | ~41 cycles | **~57 cycles** |
| Throughput (MACs/cycle) | 0.04 | 0.10 | **0.28** |
| Tile size | 1x1 | 1x2 | **1x1** |
| ReLU | No | Yes | **Yes** |

### ASIC-optimized design

Following the [TT HDL guide](https://tinytapeout.com/hdl/fpga_vs_asic/):
- **Minimal flops**: Weight magnitude decoded combinationally (only 2 registers per PE: 4-bit weight code + 10-bit accumulator)
- **No `initial` blocks**: Explicit `rst_n` reset on accumulator only
- **`(* keep *)` FFs** on unused output pins to prevent Yosys `conb` cells from causing LVS shorts to VGND (pattern from [Mini-TPU](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi))

## Architecture

```
         act_col0  act_col1  act_col2  act_col3
            |         |         |         |
         +---+     +---+     +---+     +---+
    W[0]  |PE |     |PE |     |PE |     |PE |
         +---+     +---+     +---+     +---+
    W[1]  |PE |     |PE |     |PE |     |PE |
         +---+     +---+     +---+     +---+
    W[2]  |PE |     |PE |     |PE |     |PE |
         +---+     +---+     +---+     +---+
    W[3]  |PE |     |PE |     |PE |     |PE |
         +---+     +---+     +---+     +---+
```

- 16 PEs, each storing one E2M1 weight (4 bits)
- Activations broadcast down columns each cycle
- After 16 cycles (one NVFP4 block): each PE holds `acc = W[i][j] * sum(act[j])`
- Host applies E4M3 block scale + FP32 tensor scale (NVFP4 dequantization)

### E2M1 Format (NVFP4/MXFP4)

4-bit encoding: 1 sign + 2 exponent + 1 mantissa

| Code | Value | x2 (integer) |
|------|-------|-------------|
| 0000 | +0.0  | 0 |
| 0001 | +0.5  | 1 |
| 0010 | +1.0  | 2 |
| 0011 | +1.5  | 3 |
| 0100 | +2.0  | 4 |
| 0101 | +3.0  | 6 |
| 0110 | +4.0  | 8 |
| 0111 | +6.0  | 12 |
| 1000 | -0.0  | 0 |
| 1001 | -0.5  | 1 |
| ...  | ...   | ... |

### Protocol

```
uio_in[7:6] selects mode:

00 = IDLE
01 = LOAD (8 cycles):  ui_in[7:4]=weight_a, ui_in[3:0]=weight_b, uio_in[5:3]=pair_idx
10 = COMPUTE (17 cycles): ui_in = {act3, act2, act1, act0} (2-bit ternary each), uio_in[0]=relu_en
11 = OUTPUT (32 cycles): uo_out = result bytes (2 per accumulator), uio_out[7]=done
```

## File Structure

```
src/
  project.v              # Top-level TT module (tt_um_kashif_fp4_ternary_tpu)
  pe.v                   # Processing element: E2M1 weight + ternary MAC
  systolic_array_4x4.v   # 4x4 weight-stationary grid
  control_fsm.v          # IDLE/LOAD/COMPUTE/OUTPUT state machine
docs/
  info.md                # Datasheet with protocol + E2M1 table
test/
  tb.v                   # Verilog testbench (GL_TEST compatible)
  test.py                # 9 cocotb tests
  Makefile               # icarus/cocotb build
DESIGN_NOTES.md           # Int7+1 structured sparsity idea for future
info.yaml                 # TT metadata: 1x1 tile, 50MHz, SKY130A
```

## Verification

9 cocotb tests, all passing in RTL simulation and GitHub Actions CI:

| Test | Description | Pattern source |
|------|-------------|---------------|
| `test_basic_mac` | Uniform weights, known activations | Original |
| `test_varied_weights` | Different E2M1 value per PE | Original |
| `test_relu` | ReLU clamping on negative outputs | Original |
| `test_zero_weights` | Zero weights -> zero output | Mini-TPU `test_zero` |
| `test_zero_activations` | Zero acts -> zero output | Mini-TPU `test_zero` |
| `test_max_accumulation` | Boundary: weight=6.0, 16x+1=192 | Mini-TPU `test_overflow` |
| `test_negative_weights` | Negative E2M1 with mixed activations | Original |
| `test_random` | 50 seeded random weight/activation trials | Mini-TPU `test_random` |
| `test_back_to_back` | Consecutive matmuls verify state reset | Mini-TPU `test_back_to_back` |

Gate-level test compatibility (from [Mini-TPU](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi)):
- `_safe_int()` for X-propagation-safe LogicArray reads
- `gl_preheat()` runs a zero-weight matmul to flush X from pipeline regs before GL tests
- `VPWR`/`VGND` power pin wiring via `GL_TEST` ifdef in testbench

## Running Tests Locally

```bash
# Install dependencies
sudo pacman -S iverilog          # Arch Linux
pip install cocotb pytest

# Run tests
cd test
make clean && make
```

## Target

- **Shuttle**: TTSKY26c (SkyWater SKY130A)
- **Tile**: 1x1 (~167x108 um)
- **Clock**: 50 MHz
- **Deadline**: 2026-09-07

## References

- [NVFP4: NVIDIA Blackwell format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [OCP Microscaling (MX) Formats spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Ternary Weight Networks (Li et al. 2016)](https://arxiv.org/abs/1605.04711)
- [Binarized Neural Networks (Hubara et al. 2016)](https://arxiv.org/abs/1602.02830)
- [PFW TPU](https://github.com/wangantian/pfw_tpu) — INT8 2x2 systolic, TT SKY26b
- [Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi) — INT4 3x3 systolic, TT SKY26b
- [FP8 MAC Unit](https://github.com/chatelao/ttihp-fp8-mul) — OCP MX streaming MAC, TT IHP26a
- [TT HDL Guide](https://tinytapeout.com/hdl/) — FPGA-to-ASIC considerations
- [TT Tech Specs](https://tinytapeout.com/specs/) — Clock, GPIO, analog, memory constraints

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that aims to make it easier and cheaper than ever to get your digital and analog designs manufactured on a real chip. To learn more and get started, visit https://tinytapeout.com.
