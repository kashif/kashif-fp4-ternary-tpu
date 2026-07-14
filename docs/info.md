# NVFP4 Ternary TPU — Tiny Tapeout Systolic Array Accelerator

This project implements a **4×4 weight-stationary systolic array** for
matrix-vector multiplication using **NVFP4 (E2M1) 4-bit floating-point
weights** and **ternary {-1, 0, +1} activations** — the first
open-silicon accelerator to use the NVFP4 format introduced in NVIDIA
Blackwell Tensor Cores.

Built using **Tiny Tapeout** and **SkyWater 130nm PDK**.
Targets the **TTSKY26c** shuttle, 1×1 tile, 50 MHz.

---

## How it Works

The key innovation: with ternary activations, the multiply-accumulate
(MAC) operation reduces to **add / subtract / no-op** — no hardware
multiplier is needed.

| Activation | Operation | Hardware |
|-----------|-----------|----------|
| +1 | `acc += weight` | adder |
| -1 | `acc -= weight` | subtractor |
| 0 | `acc` unchanged | nothing |

This eliminates the most expensive cell in each processing element (PE).
A conventional 4-bit INT multiplier (~200+ gates) is replaced by a
MUX + adder (~60 gates), allowing 16 PEs (4×4) to fit in a 1×1 tile
where a conventional design fits only 9 (3×3).

### Architecture

```
         act_col0  act_col1  act_col2  act_col3
            |         |         |         |
         +---+     +---+     +---+     +---+
  W[0]    |PE |     |PE |     |PE |     |PE |
         +---+     +---+     +---+     +---+
  W[1]    |PE |     |PE |     |PE |     |PE |
         +---+     +---+     +---+     +---+
  W[2]    |PE |     |PE |     |PE |     |PE |
         +---+     +---+     +---+     +---+
  W[3]    |PE |     |PE |     |PE |     |PE |
         +---+     +---+     +---+     +---+
```

- **16 PEs**, each storing one E2M1 weight (4 bits)
- Activations **broadcast down columns** each cycle
- After 16 cycles (one NVFP4 block): each PE holds
  `acc = W[i][j] × sum(act[j])`
- Host applies **E4M3 block scale + FP32 tensor scale**
  (NVFP4 dequantization) in software on the RP2040
- Built-in **ReLU** on latched accumulator snapshot

### PE Internal Structure

Each PE contains only 2 registers:
- **4-bit weight code** (E2M1, loaded once, held for block)
- **10-bit signed accumulator** (async reset)

Weight magnitude is decoded **combinationally** from the 4-bit code
via an 8-entry LUT — no separate magnitude register needed (per TT
HDL guide: "flops are the most expensive cells in ASIC").

---

## E2M1 Format (NVFP4 / MXFP4)

4-bit encoding: **1 sign + 2 exponent + 1 mantissa**

| Code | Value | ×2 (integer domain) |
|------|-------|---------------------|
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
| 1010 | -1.0  | 2 |
| 1011 | -1.5  | 3 |
| 1100 | -2.0  | 4 |
| 1101 | -3.0  | 6 |
| 1110 | -4.0  | 8 |
| 1111 | -6.0  | 12 |

The PE works in a **scaled-by-2 integer domain** to avoid fractions.
Max accumulation: 16 cycles × 12 = 192, which fits in 9 bits signed
(10 bits used for margin).

### NVFP4 vs MXFP4

| Feature | MXFP4 (OCP MX) | NVFP4 (NVIDIA Blackwell) |
|---------|----------------|--------------------------|
| Element format | E2M1 (4-bit) | E2M1 (4-bit) — **same** |
| Block size | 32 elements | **16 elements** |
| Block scale | E8M0 (power-of-2) | **E4M3 (FP8 multiply)** |
| Tensor scale | None | **FP32 (software)** |

This design uses the E2M1 element format (shared by both). The block
scale and tensor scale are applied by the host RP2040 in software,
keeping the ASIC minimal. The vLLM Qwix framework fuses E4M3 + FP32
scales into a single FP32 blockwise scale — the same approach works
here.

---

## Protocol

```
uio_in[7:6] selects mode:

00 = IDLE
01 = LOAD      (8 cycles for 16 weights)
10 = COMPUTE   (18 cycles for 16 MACs + clear + drain)
11 = OUTPUT    (32 cycles for 16 results × 2 bytes)
```

### LOAD — 8 cycles

| Signal | Field | Bits |
|--------|-------|------|
| `ui_in[7:4]` | weight_a (E2M1) | 4 |
| `ui_in[3:0]` | weight_b (E2M1) | 4 |
| `uio_in[5:3]` | pair index (0–7) | 3 |

Pair index maps to (row, col_pair):
`row = idx[2:1]`, `col_pair = idx[0]` (0 → cols 0,1; 1 → cols 2,3)

### COMPUTE — 18 cycles

| Signal | Field | Bits |
|--------|-------|------|
| `ui_in[7:6]` | act_col0 (ternary) | 2 |
| `ui_in[5:4]` | act_col1 (ternary) | 2 |
| `ui_in[3:2]` | act_col2 (ternary) | 2 |
| `ui_in[1:0]` | act_col3 (ternary) | 2 |
| `uio_in[0]` | relu_en | 1 |

Ternary encoding: `00` = 0, `01` = +1, `10` = -1, `11` = reserved

### OUTPUT — 32 cycles

| Signal | Field |
|--------|-------|
| `uo_out[7:0]` | result byte (high then low per 10-bit accumulator) |
| `uio_out[7]` | done flag (1 when all 32 bytes sent) |
| `uio_out[3:0]` | status (mode + counter for debugging) |

Byte packing: even cycle = `{6'b0, acc[9:8]}`, odd cycle = `acc[7:0]`

---

## How to Test

### Simulation

```bash
cd test
make clean && make
```

9 cocotb tests run on Icarus Verilog:

| Test | Description |
|------|-------------|
| `test_basic_mac` | Uniform weights, column-specific activation patterns |
| `test_varied_weights` | All 8 E2M1 magnitudes, checks `w_scaled × 16` |
| `test_relu` | ReLU clamps negative accumulators to 0 |
| `test_zero_weights` | Zero weights → zero output regardless of activations |
| `test_zero_activations` | Zero activations → zero output regardless of weights |
| `test_max_accumulation` | Max weight (mag=12) × 16 × +1 = 192 (boundary) |
| `test_negative_weights` | Negative E2M1 with mixed activation patterns |
| `test_random` | 50 seeded random weight/activation trials |
| `test_back_to_back` | Consecutive matmuls verify state reset |

### Gate-Level Testing

Gate-level tests run automatically via the GDS GitHub Action. The
testbench includes `GL_TEST` ifdefs for `VPWR`/`VGND` power pins.
`gl_preheat()` flushes X values from pipeline registers before real
tests. `_safe_int()` handles X-valued LogicArray reads.

### Hardware (TT Demo Board)

1. Reset: `rst_n` low for 10+ clock cycles
2. Set `uio_in[7:6] = 01` (LOAD mode)
3. Load 8 weight pairs: `ui_in = {weight_a, weight_b}`, `uio_in[5:3] = index`
4. Set `uio_in[7:6] = 10` (COMPUTE mode)
5. Stream 16+ cycles of ternary activations via `ui_in`, `uio_in[0] = relu_en`
6. Set `uio_in[7:6] = 11` (OUTPUT mode)
7. Read 32 bytes from `uo_out`, check `uio_out[7] = 1` (done)

---

## External Hardware

No external hardware required. All I/O via TT demo board pins.
The RP2040 on the demo board handles:
- Pre-processing (image downsampling, ternarization)
- Weight loading and activation streaming
- Post-processing (E4M3/FP32 scale application, argmax)

---

## Application: MNIST Inference

A 2-layer MLP can run on the ASIC + RP2040:

```
28×28 image
    │
    ▼  RP2040: average 4 quadrants → ternarize
4 ternary inputs {-1,0,+1}
    │
    ▼  ASIC call 1: LOAD W1[4×4], COMPUTE with relu_en=1, OUTPUT
4 ReLU'd values
    │
    ▼  RP2040: apply E4M3 scale, ternarize
4 ternary hidden values
    │
    ├─ ASIC call 2: LOAD W2a[4×4], COMPUTE, OUTPUT → 4 logits
    ├─ ASIC call 3: LOAD W2b[4×4], COMPUTE, OUTPUT → 4 logits
    └─ ASIC call 4: LOAD W2c[4×4], COMPUTE, OUTPUT → 2 logits
         │
    RP2040: apply scales, argmax over 10 → digit 0-9
```

4 ASIC calls × ~57 cycles = ~228 cycles. The RP2040 handles
pre/post-processing and the final classification head in software.
