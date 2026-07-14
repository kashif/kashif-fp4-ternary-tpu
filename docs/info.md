<!---
This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.
-->

## How it works

A mini TPU built around a **4x4 output-stationary systolic array** that
computes `C = A x W` with **ternary activations {-1, 0, +1}** and **NVFP4
(E2M1) 4-bit floating-point weights** — the format introduced in NVIDIA
Blackwell Tensor Cores. There is **no hardware multiplier anywhere**: with
ternary activations the multiply-accumulate reduces to a mux-add,

```
+1 -> acc += w      -1 -> acc -= w      0 -> acc unchanged
```

E2M1 weights are decoded combinationally to the x2 integer domain
(0, 1, 2, 3, 4, 6, 8, 12 with sign; -0 = 0), so results are exact 7-bit
signed integers (max |C| = 48). Block scaling happens on the host during
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
(real dot products), and a full matmul runs in a 10-cycle wavefront.

Activation functions are host-side: they are only correct after cross-tile
partial-sum accumulation and bias, which happen on the host anyway.

### Second mode: ternary/binary weights x INT4 activations (Bonsai-style)

The RUN instruction's `m` flag re-interprets the 4-bit memory-B operand as
plain INT4 two's complement instead of E2M1. Because the systolic dot
product is symmetric, the host simply swaps operand roles: **ternary (or
binary) weights** go into memory A — 2-bit codes, byte-compatible with the
deployed Bonsai ternary layout — and **INT4 activations** into memory B.
FP16 group scales (e.g. one per 128 weights, as in Bonsai/BitNet-style
models) are applied by the host to the chip's exact partial sums. Max
|C| = 32 in this mode, still exact in the 7-bit accumulators. Hardware
cost: one 7-bit 2:1 mux per PE.

### Instruction set (16 bits, sent LSB-first over SPI)

| Instruction | Format (binary)        | Description |
|-------------|------------------------|-------------|
| `LOAD A`    | `10 0 rr 0ee 000000tt` | Ternary activation `t` (00=0, 01=+1, 10=-1) into row `r` (0-3), element `e` (0-3) |
| `LOAD B`    | `10 1 cc 0kk 0000wwww` | E2M1 weight nibble into column `c` (0-3), step `k` (0-3) |
| `RUN`       | `01 m 0000000000000`   | Clear accumulators, run the wavefront (10 cycles); `m`=0 E2M1 weight decode, `m`=1 INT4 decode (Bonsai mode) |
| `STORE`     | `11 0 rr cc 000000000` | Drive C[r][c] (sign-extended byte) on `uo_out` |

After power-up, issue one throwaway RUN before the first real matmul: the
PE pipeline registers are no-reset cells (area optimization from the
reference) and hold random values until a wavefront flushes them.

SCLK must be at most clk/6 (the SPI bit counter crosses clock domains
unsynchronised, as in the reference). The `ready` pin (uio[1]) pulses when a
RUN completes; alternatively just wait 10+ clock cycles. The SPI is
receive-only: all results are read via STORE on `uo_out` (the reference's
MISO readback stream is omitted to save area).

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
includes ternary semantics (+1/-1/0), both decode modes, negative-zero handling, a
non-degeneracy test (equal-sum activation matrices must produce different
results), accumulator-clear checks, and randomized full-coverage trials.

## External hardware

None required. Any SPI-capable host (e.g. the demo board's RP2040) drives
MOSI/CS/SCLK and reads result bytes on `uo_out`.
