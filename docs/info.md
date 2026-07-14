<!---
This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.
-->

## How it works

A mini TPU built around a **3x3 output-stationary systolic array** that
computes `C = A x W` with **ternary activations {-1, 0, +1}** and **NVFP4
(E2M1) 4-bit floating-point weights** — the format introduced in NVIDIA
Blackwell Tensor Cores. There is **no hardware multiplier anywhere**: with
ternary activations the multiply-accumulate reduces to a mux-add,

```
+1 -> acc += w      -1 -> acc -= w      0 -> acc unchanged
```

E2M1 weights are decoded combinationally to the x2 integer domain
(0, 1, 2, 3, 4, 6, 8, 12 with sign; -0 = 0), so results are exact 7-bit
signed integers (max |C| = 36). Block scaling happens on the host during
dequantization, which makes the chip element-level format-agnostic: apply
E4M3 scales per 16-element block (+ FP32 tensor scale) for **NVFP4**
semantics, or E8M0 power-of-two scales per 32-element block for **MXFP4**.
NVFP4 is the more accurate of the two (smaller blocks isolate outliers;
FP8 scales are far finer than powers of two), and the exact integer
accumulators let the host apply per-block scales to bit-exact partial sums.

Architecture, SPI protocol, and skewed-wavefront control follow the proven
reference mini-TPU
([MILOUDIAS/IEEE_ttsky_mini_tpu_spi](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi)):
activations flow right, weights flow down, both streams change every cycle
(real dot products), and a full matmul runs in a 7-cycle wavefront. An
optional ReLU (flag in the RUN instruction) clamps negative results at
readout.

### Instruction set (16 bits, sent LSB-first over SPI)

| Instruction | Format (binary)        | Description |
|-------------|------------------------|-------------|
| `LOAD A`    | `10 0 rr 0ee 000000tt` | Ternary activation `t` (00=0, 01=+1, 10=-1) into row `r` (0-2), element `e` (0-2) |
| `LOAD B`    | `10 1 cc 0kk 0000wwww` | E2M1 weight nibble into column `c` (0-2), step `k` (0-2) |
| `RUN`       | `01 r 0000000000000`   | Clear accumulators, run the wavefront (7 cycles); `r`=1 applies ReLU at readout |
| `STORE`     | `11 0 rr cc 000000000` | Drive C[r][c] (sign-extended byte) on `uo_out` |

SCLK must be at most clk/6 (the SPI bit counter crosses clock domains
unsynchronised, as in the reference). The `ready` pin (uio[1]) pulses when a
RUN completes; alternatively just wait 7+ clock cycles. The SPI is
receive-only: all results are read via STORE on `uo_out` (the reference's
MISO readback stream is omitted to save area on the 1x1 tile).

### E2M1 weight encoding (element type of NVFP4 and MXFP4)

| Code | Value | x2 integer |     | Code | Value | x2 integer |
|------|-------|------------|-----|------|-------|------------|
| 0000 | +0.0  | 0  | | 1000 | -0.0  | 0   |
| 0001 | +0.5  | 1  | | 1001 | -0.5  | -1  |
| 0010 | +1.0  | 2  | | 1010 | -1.0  | -2  |
| 0011 | +1.5  | 3  | | 1011 | -1.5  | -3  |
| 0100 | +2.0  | 4  | | 1100 | -2.0  | -4  |
| 0101 | +3.0  | 6  | | 1101 | -3.0  | -6  |
| 0110 | +4.0  | 8  | | 1110 | -4.0  | -8  |
| 0111 | +6.0  | 12 | | 1111 | -6.0  | -12 |

## How to test

Run the cocotb testbench:

```
cd test
make -B
```

The suite drives the SPI interface exactly like an external host and checks
the full `C = A x W` result against an **independent golden model** (E2M1 and
ternary decode from first principles, then a plain matrix multiply). It
includes ternary semantics (+1/-1/0), ReLU on/off, negative-zero handling, a
non-degeneracy test (equal-sum activation matrices must produce different
results), accumulator-clear checks, and randomized full-coverage trials.

## External hardware

None required. Any SPI-capable host (e.g. the demo board's RP2040) drives
MOSI/CS/SCLK and reads result bytes on `uo_out`.
