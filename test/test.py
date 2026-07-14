# SPDX-FileCopyrightText: (c) 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

GL_TEST = bool(os.environ.get("GATES") == "yes")

# E2M1 decode: 4-bit code -> integer value (scaled x2)
E2M1_MAG = {
    0b000: 0, 0b001: 1, 0b010: 2, 0b011: 3,
    0b100: 4, 0b101: 6, 0b110: 8, 0b111: 12,
}

def e2m1_decode(code):
    sign = (code >> 3) & 1
    mag = E2M1_MAG[code & 0x7]
    return -mag if sign else mag

TERNARY = {0: 0, 1: 1, 2: -1}

MODE_IDLE    = 0b00
MODE_LOAD    = 0b01
MODE_COMPUTE = 0b10
MODE_OUTPUT  = 0b11


def _safe_int(val, default=0):
    """Read a possibly-X LogicArray as int (for GL sim safety)."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


async def hw_reset(dut, n=10):
    """Reset with known-high start to guarantee 1->0 edge in GL."""
    dut.rst_n.value = 1
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 0
    for _ in range(n):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, weights):
    """Load 16 E2M1 weights (4x4 matrix, row-major) via 8 cycles."""
    for idx in range(8):
        row = idx >> 1
        col_pair = idx & 1
        col_a = col_pair * 2
        col_b = col_pair * 2 + 1
        wa = weights[row][col_a]
        wb = weights[row][col_b]
        dut.ui_in.value = (wa << 4) | wb
        dut.uio_in.value = (MODE_LOAD << 6) | (idx << 3)
        await ClockCycles(dut.clk, 1)


async def compute_block(dut, activations, relu_en=0):
    """Send compute mode for 22 cycles (1 clear + 17 compute + 1 drain + 1 snapshot + 2 spare)."""
    for cycle in range(22):
        if cycle < len(activations):
            a0, a1, a2, a3 = activations[cycle]
        else:
            a0, a1, a2, a3 = 0, 0, 0, 0
        dut.ui_in.value = (a0 << 6) | (a1 << 4) | (a2 << 2) | a3
        dut.uio_in.value = (MODE_COMPUTE << 6) | relu_en
        await ClockCycles(dut.clk, 1)


async def read_results(dut):
    """Read 32 bytes (16 results x 2 bytes each: high then low)."""
    results = []
    dut.uio_in.value = (MODE_OUTPUT << 6)
    await ClockCycles(dut.clk, 1)

    for i in range(16):
        await ClockCycles(dut.clk, 1)
        high = _safe_int(dut.uo_out.value)

        await ClockCycles(dut.clk, 1)
        low = _safe_int(dut.uo_out.value)

        result = ((high & 0x03) << 8) | low
        if result & 0x200:
            result -= 0x400
        results.append(result)

    return results


async def run_full(dut, weights, activations, relu_en=0):
    """Helper: reset, load, compute, read results."""
    await hw_reset(dut)
    await load_weights(dut, weights)
    await ClockCycles(dut.clk, 2)
    await compute_block(dut, activations, relu_en)
    await ClockCycles(dut.clk, 2)
    return await read_results(dut)


def expected_acc(weight_code, act_list):
    """Compute expected accumulator value in scaled-by-2 domain."""
    w = e2m1_decode(weight_code)
    acc = 0
    for a_code in act_list:
        a = TERNARY.get(a_code, 0)
        acc += a * w
    return acc


@cocotb.test()
async def test_basic_mac(dut):
    """Test basic multiply-accumulate with known weights and activations."""
    dut._log.info("Start NVFP4 Ternary TPU test")

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    # Weight = +2.0 (E2M1 code 0100), scaled mag = 4
    weights = [[0b0100] * 4 for _ in range(4)]

    # Col 0: all +1, Col 1: all -1, Col 2: all 0, Col 3: alternating
    act_list = [(0b01, 0b10, 0b00, 0b01 if i % 2 == 0 else 0b10) for i in range(16)]

    results = await run_full(dut, weights, act_list)
    dut._log.info(f"Results: {results}")

    for r in range(4):
        assert results[r*4+0] == 64, f"PE[{r}][0]: expected 64, got {results[r*4+0]}"
        assert results[r*4+1] == -64, f"PE[{r}][1]: expected -64, got {results[r*4+1]}"
        assert results[r*4+2] == 0, f"PE[{r}][2]: expected 0, got {results[r*4+2]}"
        assert results[r*4+3] == 0, f"PE[{r}][3]: expected 0, got {results[r*4+3]}"

    dut._log.info("Basic MAC test PASSED!")


@cocotb.test()
async def test_varied_weights(dut):
    """Test with different E2M1 weight values per PE."""
    dut._log.info("Start varied weights test")

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    weights = [
        [0b0001, 0b0010, 0b0011, 0b0100],
        [0b0101, 0b0110, 0b0111, 0b0000],
        [0b1010, 0b1100, 0b1101, 0b1110],
        [0b0000, 0b1001, 0b1010, 0b1011],
    ]

    # All activations = +1 for all 16 cycles
    act_list = [(0b01, 0b01, 0b01, 0b01) for _ in range(16)]

    results = await run_full(dut, weights, act_list)
    dut._log.info(f"Results: {results}")

    for r in range(4):
        for c in range(4):
            w_scaled = e2m1_decode(weights[r][c])
            expected = w_scaled * 16
            actual = results[r*4 + c]
            assert actual == expected, \
                f"PE[{r}][{c}]: weight_scaled={w_scaled}, expected {expected}, got {actual}"

    dut._log.info("Varied weights test PASSED!")


@cocotb.test()
async def test_relu(dut):
    """Test ReLU activation on output."""
    dut._log.info("Start ReLU test")

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    # Weight = +1.0 (code 0010), scaled mag = 2
    weights = [[0b0010] * 4 for _ in range(4)]

    # Col 0: all -1 -> acc = -32, Col 1: all +1 -> acc = +32
    act_list = [(0b10, 0b01, 0b00, 0b00) for _ in range(16)]

    results = await run_full(dut, weights, act_list, relu_en=1)
    dut._log.info(f"ReLU Results: {results}")

    for r in range(4):
        assert results[r*4+0] == 0, f"PE[{r}][0] with ReLU: expected 0, got {results[r*4+0]}"
        assert results[r*4+1] == 32, f"PE[{r}][1] with ReLU: expected 32, got {results[r*4+1]}"

    dut._log.info("ReLU test PASSED!")
