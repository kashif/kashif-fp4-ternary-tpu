# SPDX-FileCopyrightText: (c) 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

# E2M1 decode: 4-bit code -> integer value (scaled x2)
E2M1_TABLE = {
    0b000: 0, 0b001: 1, 0b010: 2, 0b011: 3,
    0b100: 4, 0b101: 6, 0b110: 8, 0b111: 12,
}

def e2m1_decode(code):
    sign = (code >> 3) & 1
    mag = E2M1_TABLE[code & 0x7]
    return -mag if sign else mag

# Ternary encoding: 00=0, 01=+1, 10=-1
TERNARY = {0: 0, 1: 1, 2: -1}

async def load_weights(dut, weights):
    """Load 16 E2M1 weights (4x4 matrix, row-major) via 8 cycles."""
    dut.uio_in.value = 0  # will set mode below
    for idx in range(8):
        row = idx >> 1      # idx[2:1]
        col_pair = idx & 1  # idx[0]
        col_a = col_pair * 2
        col_b = col_pair * 2 + 1
        wa = weights[row][col_a]
        wb = weights[row][col_b]
        dut.ui_in.value = (wa << 4) | wb
        dut.uio_in.value = (0b01 << 6) | (idx << 3)  # mode=LOAD, idx
        await ClockCycles(dut.clk, 1)

async def compute_block(dut, activations, relu_en=0):
    """Stream 16 ternary activations (one per column per cycle)."""
    for cycle in range(16):
        a0 = activations[cycle][0]
        a1 = activations[cycle][1]
        a2 = activations[cycle][2]
        a3 = activations[cycle][3]
        dut.ui_in.value = (a0 << 6) | (a1 << 4) | (a2 << 2) | a3
        dut.uio_in.value = (0b10 << 6) | relu_en
        await ClockCycles(dut.clk, 1)

    # Extra cycle for snapshot to latch
    dut.uio_in.value = (0b10 << 6) | relu_en
    await ClockCycles(dut.clk, 1)

async def read_results(dut):
    """Read 32 bytes (16 results x 2 bytes each)."""
    results = []
    dut.uio_in.value = (0b11 << 6)  # mode=OUTPUT
    await ClockCycles(dut.clk, 1)

    for i in range(16):
        # High byte
        while True:
            val = dut.uo_out.value
            await ClockCycles(dut.clk, 1)
            break
        high = int(dut.uo_out.value)

        # Low byte
        await ClockCycles(dut.clk, 1)
        low = int(dut.uo_out.value)

        result = ((high & 0x03) << 8) | low
        # Sign extend from 10 bits
        if result & 0x200:
            result -= 0x400
        results.append(result)

    return results

def expected_result(weight_code, activations):
    """Compute expected accumulator value in software."""
    w = e2m1_decode(weight_code)
    acc = 0
    for a_code in activations:
        a = TERNARY[a_code]
        acc += a * w
    return acc


@cocotb.test()
async def test_basic_mac(dut):
    """Test basic multiply-accumulate with known weights and activations."""
    dut._log.info("Start NVFP4 Ternary TPU test")

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # Load weights: all PEs get weight = +2.0 (E2M1 code = 0b0100)
    weights = [[0b0100] * 4 for _ in range(4)]
    await load_weights(dut, weights)
    await ClockCycles(dut.clk, 2)

    # Compute: 16 cycles of activations
    # Column 0: all +1 -> acc = 16 * 2 = 32
    # Column 1: all -1 -> acc = 16 * (-2) = -32
    # Column 2: all 0  -> acc = 0
    # Column 3: alternating +1/-1 -> acc = 0
    activations = []
    for i in range(16):
        a0 = 0b01  # +1
        a1 = 0b10  # -1
        a2 = 0b00  # 0
        a3 = 0b01 if (i % 2 == 0) else 0b10  # +1/-1 alternating
        activations.append((a0, a1, a2, a3))

    await compute_block(dut, activations)
    await ClockCycles(dut.clk, 2)

    # Read results
    results = await read_results(dut)

    dut._log.info(f"Results: {results}")

    # Check expected values for row 0
    # PE[0][0]: w=2, sum(act)=16 -> acc=32
    # PE[0][1]: w=2, sum(act)=-16 -> acc=-32
    # PE[0][2]: w=2, sum(act)=0 -> acc=0
    # PE[0][3]: w=2, sum(act)=0 (8*+1 + 8*-1) -> acc=0
    assert results[0] == 32, f"PE[0][0]: expected 32, got {results[0]}"
    assert results[1] == -32, f"PE[0][1]: expected -32, got {results[1]}"
    assert results[2] == 0, f"PE[0][2]: expected 0, got {results[2]}"
    assert results[3] == 0, f"PE[0][3]: expected 0, got {results[3]}"

    dut._log.info("Basic MAC test PASSED!")


@cocotb.test()
async def test_varied_weights(dut):
    """Test with different E2M1 weight values per PE."""
    dut._log.info("Start varied weights test")

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # Load varied weights
    # Row 0: [0.5, 1.0, 1.5, 2.0] = codes [0001, 0010, 0011, 0100]
    # Row 1: [3.0, 4.0, 6.0, 0.0] = codes [0101, 0110, 0111, 0000]
    # Row 2: [-1.0, -2.0, -3.0, -4.0] = codes [1010, 1100, 1101, 1110]
    # Row 3: [0.0, 0.5, 1.0, 1.5] = codes [0000, 1001, 1010, 1011]
    weights = [
        [0b0001, 0b0010, 0b0011, 0b0100],
        [0b0101, 0b0110, 0b0111, 0b0000],
        [0b1010, 0b1100, 0b1101, 0b1110],
        [0b0000, 0b1001, 0b1010, 0b1011],
    ]
    await load_weights(dut, weights)
    await ClockCycles(dut.clk, 2)

    # All activations = +1 for all columns, 16 cycles
    activations = [(0b01, 0b01, 0b01, 0b01) for _ in range(16)]
    await compute_block(dut, activations)
    await ClockCycles(dut.clk, 2)

    results = await read_results(dut)
    dut._log.info(f"Results: {results}")

    # Check: each acc = weight_val * 16
    for r in range(4):
        for c in range(4):
            w_val = e2m1_decode(weights[r][c])
            expected = w_val * 16
            actual = results[r * 4 + c]
            assert actual == expected, \
                f"PE[{r}][{c}]: weight={w_val}, expected {expected}, got {actual}"

    dut._log.info("Varied weights test PASSED!")


@cocotb.test()
async def test_relu(dut):
    """Test ReLU activation on output."""
    dut._log.info("Start ReLU test")

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    # Weight = +1.0 (code 0010) for all PEs
    weights = [[0b0010] * 4 for _ in range(4)]
    await load_weights(dut, weights)
    await ClockCycles(dut.clk, 2)

    # Column 0: all -1 -> acc = -16 (should be clamped to 0 with ReLU)
    # Column 1: all +1 -> acc = +16 (should stay 16)
    activations = [(0b10, 0b01, 0b00, 0b00) for _ in range(16)]
    await compute_block(dut, activations, relu_en=1)
    await ClockCycles(dut.clk, 2)

    results = await read_results(dut)
    dut._log.info(f"ReLU Results: {results}")

    # With ReLU: col 0 should be 0, col 1 should be 16
    assert results[0] == 0, f"PE[0][0] with ReLU: expected 0, got {results[0]}"
    assert results[1] == 16, f"PE[0][1] with ReLU: expected 16, got {results[1]}"

    dut._log.info("ReLU test PASSED!")
