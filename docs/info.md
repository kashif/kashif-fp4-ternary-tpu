# NVFP4 Ternary TPU

## How it works

This project implements a 4x4 weight-stationary systolic array for matrix-vector multiplication using:

- **NVFP4/MXFP4 (E2M1) 4-bit weights** — the same 4-bit floating-point format used in NVIDIA Blackwell Tensor Cores
- **Ternary activations** constrained to {-1, 0, +1} — inspired by Ternary Weight Networks (Li et al. 2016)

The key innovation: with ternary activations, the multiply-accumulate operation reduces to **add / subtract / no-op** — no hardware multiplier is needed. Each processing element (PE) is just a 4-bit weight register, a magnitude LUT, and a 10-bit accumulator.

### Architecture

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

- 16 PEs, each storing one E2M1 weight
- Activations broadcast down columns each cycle
- After 16 cycles (one NVFP4 block): each PE holds acc = W[i][j] * sum(act[j])
- Host applies E4M3 block scale + FP32 tensor scale (NVFP4 dequantization) in software

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

1. **LOAD** (8 cycles): Load 16 weights, 2 per cycle via ui_in[7:4] and ui_in[3:0], with uio_in[5:3] selecting the pair index (0-7)
2. **COMPUTE** (16 cycles): Stream 4 ternary activations per cycle via ui_in (2 bits each), uio_in[0] = ReLU enable
3. **OUTPUT** (32 cycles): Read back 16 accumulators as 2 bytes each (high then low) via uo_out, uio_out[7] = done

## How to test

1. Reset the design (rst_n low for several cycles)
2. Set uio_in[7:6] = 01 (LOAD mode)
3. Load 8 weight pairs: ui_in = {weight_a, weight_b}, uio_in[5:3] = index
4. Set uio_in[7:6] = 10 (COMPUTE mode)
5. Stream 16 cycles of ternary activations: ui_in = {act3, act2, act1, act0} (2 bits each)
6. Set uio_in[7:6] = 11 (OUTPUT mode)
7. Read 32 bytes from uo_out (16 results × 2 bytes, high byte first)
8. Check uio_out[7] = 1 (done)

## External hardware

No external hardware required. All I/O via TT demo board pins.
