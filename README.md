![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# NVFP4 Ternary Mini-TPU

A 3x3 output-stationary systolic array computing `C = A x W` with **NVFP4
(E2M1) 4-bit floating-point weights** and **ternary {-1, 0, +1} activations**
— and **no hardware multiplier anywhere**. Fits a single Tiny Tapeout 1x1
tile.

- [Read the documentation for project](docs/info.md)

## Why it's interesting

### No multiplier

With ternary activations, multiply-accumulate reduces to add / subtract /
skip:

| Activation | Operation |
|-----------|-----------|
| +1 | `acc += weight` |
| -1 | `acc -= weight` |
| 0  | `acc` unchanged |

The most expensive cell in a conventional PE simply doesn't exist here — each
PE is an E2M1 decoder, a mux, and a 7-bit adder.

### NVFP4 elements, same as Blackwell (and how MXFP4 differs)

E2M1 weights (1 sign + 2 exponent + 1 mantissa) represent
±{0, 0.5, 1, 1.5, 2, 3, 4, 6} — the element encoding shared by
[NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
and MXFP4. The two formats differ only in **block scaling**, which in this
architecture lives on the host:

| | NVFP4 | MXFP4 (OCP MX) |
|---|---|---|
| Element type | E2M1 | E2M1 |
| Block size | **16** | 32 |
| Block scale | **E4M3 (FP8)** | E8M0 (power of 2) |
| Tensor scale | FP32 | — |

NVFP4's smaller blocks isolate outliers better and its FP8 scales quantize
far finer than powers of two — which is why it's the more accurate format,
especially for LLMs. The chip computes raw E2M1 products in the x2 integer
domain, exactly; the host applies whichever block-scaling scheme it wants
during dequantization, so the same silicon serves **both** formats. Because
the accumulators are exact, the host can apply per-block scales to bit-exact
partial sums (a K=16 NVFP4 block spans multiple K=3 tiles) with no
accumulated rounding from the hardware.

### A real systolic matmul

The architecture is the silicon-proven
[Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi): operand
memories, skewed 7-cycle wavefront, activations flowing right, weights
flowing down, results accumulating in place. Both operand streams change
every cycle — the array computes true dot products
(`C[i][c] = sum_k A[i][k] * W[k][c]`), verified against an independent golden
model.

```
            W col 0    W col 1    W col 2      (E2M1 nibbles, skewed)
               |          |          |
A row 0 --> [PE 00] -> [PE 01] -> [PE 02]      A rows: ternary codes
               |          |          |
A row 1 --> [PE 10] -> [PE 11] -> [PE 12]
               |          |          |
A row 2 --> [PE 20] -> [PE 21] -> [PE 22]
```

### SPI instruction set (16 bits, LSB-first; SCLK <= clk/6)

| Instruction | Format (binary)        | Description |
|-------------|------------------------|-------------|
| `LOAD A`    | `10 0 rr 0ee 000000tt` | Ternary activation into row `r`, element `e` |
| `LOAD B`    | `10 1 cc 0kk 0000wwww` | E2M1 nibble into column `c`, step `k` |
| `RUN`       | `01 r 0000000000000`   | Clear accumulators, run 7 cycles; `r`=1 = ReLU |
| `STORE`     | `11 0 rr cc 000000000` | C[r][c] as sign-extended byte on `uo_out` |

Pins: `ui[0]`=MOSI, `ui[1]`=CS, `ui[2]`=SCLK; `uo_out`=result byte;
`uio[1]`=ready. The SPI is receive-only — results are read via STORE on
`uo_out`, so the reference's MISO readback stream is omitted (area).

## File structure

```
src/
  project.v     # Top-level TT module (tt_um_kashif_fp4_ternary_tpu)
  tpu.v         # Core: control + memories + array + result mux (ReLU)
  spi.v         # SPI instruction receiver, 16-bit, receive-only
  control.v     # LOAD/RUN/STORE decode, skewed wavefront counter
  memory_a.v    # Activations: 3x3 ternary (2-bit)
  memory_b.v    # Weights: 3x3 E2M1 nibbles
  array.v       # 3x3 systolic array, 7-bit exact accumulators
  pe.v          # E2M1 decode + ternary mux-add MAC (no multiplier)
test/
  tb.v          # Verilog testbench (GL_TEST compatible)
  test.py       # 7 cocotb tests with independent golden model
  Makefile      # icarus/cocotb build
info.yaml       # TT metadata: 1x1 tile, 5 MHz, SKY130A
```

## Verification

7 cocotb tests drive the SPI interface like an external host and compare all
9 results against an independent golden model:

| Test | Description |
|------|-------------|
| `test_known_matmul` | Hand-checked matmul, mixed E2M1 values |
| `test_ternary_semantics` | +1 adds, -1 subtracts, 0 skips |
| `test_relu` | ReLU flag clamps negatives; off passes them |
| `test_not_degenerate` | Equal-sum activations must differ (guards against w*sum collapse) |
| `test_run_clears_accumulators` | Back-to-back RUNs don't double |
| `test_negative_zero` | E2M1 -0 behaves as exact zero |
| `test_random` | 12 randomized full-coverage trials (random ReLU) |

## Target

- **Shuttle**: TTSKY26c (SkyWater SKY130A)
- **Tile**: 1x1 (~167x108 um)
- **Clock**: 5 MHz (SPI SCLK <= 833 kHz)

## References

- [NVFP4: NVIDIA Blackwell format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [OCP Microscaling (MX) Formats spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Ternary Weight Networks (Li et al. 2016)](https://arxiv.org/abs/1605.04711)
- [Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi) — architecture and SPI protocol base
- [Companion design: Int7+1 Sparse Mini-TPU](https://github.com/kashif/kashif-int7-sparse-tpu)
- [TT HDL Guide](https://tinytapeout.com/hdl/) / [TT Tech Specs](https://tinytapeout.com/specs/)

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that aims to make it easier and cheaper than ever to get your digital and analog designs manufactured on a real chip. To learn more and get started, visit https://tinytapeout.com.
