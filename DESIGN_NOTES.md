# Design Notes & Future Ideas

## Current Design: NVFP4 Ternary TPU (SKY130, 1x1 tile)

4x4 weight-stationary systolic array. E2M1 4-bit weights + ternary activations.
Multiply = MUX-add (no hardware multiplier). 16 PE accumulators output via streaming protocol.

## Future Idea: Int7+1 with 1:2 Structured Sparsity (Plan B)

From Roune's talk (NVIDIA/Google TPU architect):

> "A 7-bit integer multiplier with 1:2 structured sparsity baked in.
> The 8th bit that you'd use in a regular int8 is repurposed to encode
> which of the two adjacent entries is non-zero. You get sparsity for
> free in the data format."

### Concept
- Each "Int8" value is actually 7 bits of magnitude + 1 bit of "which neighbor is active"
- Pairs of values: one is always zero, the other carries 7-bit data
- 2:4 sparsity pattern (like NVIDIA Ampere 2:4, but with 1:2 = 50% by construction)
- No sparsity pattern detection needed — it's encoded in the data format itself

### Why it's elegant
- INT8 multiplier area = ~200 gates. INT7 multiplier = ~150 gates (25% savings)
- 50% sparsity means half the PEs skip computation → 2x throughput
- The "which is active" bit is free — no extra metadata
- INT8 is "always sufficient for inference if quantization is done properly" (Roune)

### TT implementation sketch
- 4x4 or 8x4 PE array, each with 7-bit signed multiplier
- Pairs of PEs share one activation slot: only one fires per cycle
- Comparator on weight[7] to select active PE in pair
- Could fit 8x4 = 32 PEs in 1x2 tile (each PE smaller due to 7-bit vs 8-bit)
- Effective throughput: 16 MACs/cycle (50% sparse over 32 PEs)

### References
- NVIDIA 2:4 structured sparsity: arXiv:2104.08378 (Mishra et al. 2021)
- Roune's talk: youtube.com/watch?v=GlAGtON6BIQ
- SemiAnalysis coverage of numerics arms race

## Other Future Ideas

### Ternary-weight TPU (simplified version of current design)
- Weights ∈ {-1, 0, +1} instead of E2M1 → no LUT needed, multiply = MUX
- Activations could be INT8 → more standard, but needs 8-bit multiplier
- Could do 8x8 array in 1x1 (each PE = 1 flip-flop + 1 full-adder)

### Bit-serial TPU
- Process 8-bit values one bit per cycle
- Each PE = 1 full-adder + 1 shift register
- Trade 8x latency for ~8x area savings → much larger array

### Stochastic TPU
- Bit-stream computing: multiply = AND gate, add = MUX
- Could fit 8x8 in 1x1 tile (each PE = 1 AND + 1 FF)
- Fundamentally different computing paradigm
