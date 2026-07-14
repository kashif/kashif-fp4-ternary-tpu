# NVFP4 Ternary Mini-TPU — Engineering Report

**Date:** 2026-07-14
**Target:** Tiny Tapeout TTSKY26c, 1x2 tile, SkyWater SKY130A
**Top module:** `tt_um_kashif_fp4_ternary_tpu`
**Clock:** 5 MHz (SPI SCLK <= clk/6)
**Result:** 7 cocotb tests passing (RTL + GL); GDS, precheck, GL test, viewer green in CI

---

## 1. Objective

An open-silicon accelerator for the E2M1 4-bit floating-point element type
(shared by NVFP4 and MXFP4) with ternary activations that eliminate the
hardware multiplier entirely — plus a second decode mode that runs
Bonsai/BitNet-style ternary/binary weights against INT4 activations on the
same PEs.

## 2. Design Metrics

| Metric | Value |
|--------|-------|
| Array | 4x4 = 16 PEs, output-stationary systolic |
| Contraction | K = 4 per pass (= NVFP4 16-element block / 4) |
| Mode 0 | E2M1 weights (x2 integer domain) x ternary activations |
| Mode 1 | ternary/binary weights x INT4 activations (RUN flag) |
| Multiplier | None (mux-add; mode selects E2M1 LUT vs INT4 decode) |
| Accumulator | 7-bit signed, exact (max 48 mode 0 / 32 mode 1) |
| Block scaling | Host-side: NVFP4 (16/E4M3+FP32), MXFP4 (32/E8M0), or FP16 group scales |
| Activation functions | Host-side (correct only after cross-tile accumulation) |
| I/O | SPI, 16-bit instructions, receive-only; results via STORE on uo_out |
| Wavefront | 10 cycles per RUN (skewed, reference mini-TPU pattern) |
| Tile / utilization | 1x2 at ~64% effective (measured in CI) |

## 3. Architecture

Ported from the silicon-proven reference
([MILOUDIAS/IEEE_ttsky_mini_tpu_spi](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi)):
operand memories, skewed wavefront control (line i streams during
t in [i+1, i+4]), activations flowing right, weights flowing down, results
accumulating in place. Both operand streams change every cycle, so the array
computes true dot products — C[i][c] = sum_k A[i][k] * B[k][c].

PE datapath: 2-bit ternary code steers a mux-add (+w / -w / skip) of the
4-bit value operand, decoded per-RUN as E2M1 (LUT to 0,1,2,3,4,6,8,12 with
sign) or INT4 two's complement. One 7-bit adder, no multiplier. Pipeline
registers are no-reset dfxtp cells (area); accumulators have async reset.

See `Architecture.drawio` and `Dataflow.drawio`.

## 4. Key Engineering Decisions (with the data that forced them)

- **1x1 does not close for this class of design with exact accumulators.**
  Three CI attempts at 3x3-on-1x1 landed at 82.7% / 80.6% / 79.0% effective
  utilization and all failed detailed placement (8-50 unplaceable instances
  at or after CTS). The reference fits 1x1 only via 4-bit *truncating*
  accumulators (results mod 16) plus a die-area fix. We kept exactness and
  took 1x2 — then spent the headroom on a 4x4 array (63.9% utilization,
  fully green).
- **SPI is receive-only.** The reference's MISO accumulator readback (63-bit
  mux + counter) duplicated the STORE readout path; removed for area.
- **No activation-function instruction.** Nonlinearities are only correct
  after cross-tile partial-sum accumulation and bias, which happen on the
  host; a per-pass on-chip clamp invites silently wrong tiled inference.
- **Constant pins driven from two shared (* keep *) FFs** (one 0, one 1)
  — avoids conb/VGND LVS merges without per-pin registers.
- **CLOCK_PERIOD relaxed to 100 ns** in config.json (design runs at 5 MHz);
  the template's 20 ns constraint wasted area on needless timing repair.
- **Throwaway first RUN after power-up** (documented in info.md): no-reset
  pipeline registers hold random values until a wavefront flushes them; the
  GL testbench does this via `gl_preheat()`.

## 5. Verification

Golden model is an independent dense matmul built from first principles
(decode E2M1/INT4 nibbles and ternary codes, multiply in plain Python) — it
shares no structure with the RTL.

| Test | What it verifies |
|------|-----------------|
| `test_known_matmul` | Hand-checked matmul, mixed E2M1 values |
| `test_ternary_semantics` | +1 adds, -1 subtracts, 0 skips |
| `test_int4_mode` | INT4 decode incl. discriminators (0x8: -0 vs -8), mode latched per RUN |
| `test_not_degenerate` | Equal-sum activations must differ (guards against w*sum collapse) |
| `test_run_clears_accumulators` | Back-to-back RUNs don't double |
| `test_negative_zero` | E2M1 -0 is exact zero |
| `test_random` | 12 randomized full-coverage trials, random mode |

Gate-level: `GATES=yes` runs the same suite (3 random trials) against the
synthesized netlist with `gl_preheat()` and VPWR/VGND wiring in `tb.v`.

## 6. File Structure

```
src/
  project.v     # TT top level, SPI pin wiring, constant-pin FFs
  tpu.v         # control + memories + array + result mux
  spi.v         # 16-bit instruction receiver (receive-only)
  control.v     # LOAD/RUN/STORE decode, skewed wavefront counter
  memory_a.v    # 4x4 ternary operand memory (2-bit)
  memory_b.v    # 4x4 value operand memory (4-bit)
  array.v       # 4x4 systolic array
  pe.v          # dual-decode mux-add PE
test/
  tb.v, test.py, Makefile
docs/
  info.md, Architecture.drawio, Dataflow.drawio, REPORT.md
```

## 7. Companion Design

[kashif-int7-sparse-tpu](https://github.com/kashif/kashif-int7-sparse-tpu):
same reference-derived skeleton, different numerics — Int7+1 weights with
1:2 structured sparsity along the contraction axis (select bit muxes an
INT4 activation pair; a mux replaces the second multiplier) plus a native
int8 dense mode, 3x3 with 13-bit exact accumulators on a 2x2 tile.
Together the two chips cover the sparse-integer and low-bit-float/ternary
corners of Roune's "numerics as competitive lever" argument.
