# SPDX-FileCopyrightText: (c) 2026 Kashif
# SPDX-License-Identifier: Apache-2.0
#
# NVFP4 ternary mini-TPU tests.
#
# The golden model is an INDEPENDENT matrix multiply built from first
# principles (decode E2M1 nibbles and ternary codes, then plain
# C = A @ W in the x2 integer domain) — it shares no structure with the
# RTL, so it cannot "pass artificially" by mirroring implementation
# quirks.

import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

GL_TEST = bool(os.environ.get("GATES") == "yes")

N = 3      # array is N x N, contraction depth K = 3

OP_RUN   = 0b01
OP_LOAD  = 0b10
OP_STORE = 0b11

# SPI pin positions within ui_in
PIN_MOSI = 0
PIN_CS   = 1
PIN_SCLK = 2

# SCLK half-period in clk cycles (SCLK = clk/8, under the clk/6 limit)
SCLK_HALF = 4

# E2M1 magnitude LUT (x2 integer domain): index = {exp[1:0], mant}
E2M1_MAG = [0, 1, 2, 3, 4, 6, 8, 12]


# ----------------------------------------------------------------------
# Encodings
# ----------------------------------------------------------------------

def ternary_code(t):
    """0 -> 00, +1 -> 01, -1 -> 10."""
    return {0: 0b00, 1: 0b01, -1: 0b10}[t]


def e2m1_decode(nibble):
    mag = E2M1_MAG[nibble & 0x7]
    return -mag if nibble & 0x8 else mag


def instr_load_a(row, elem, t):
    return (OP_LOAD << 14) | (0 << 13) | (row << 11) | (elem << 8) | ternary_code(t)


def instr_load_b(col, k, nibble):
    return (OP_LOAD << 14) | (1 << 13) | (col << 11) | (k << 8) | (nibble & 0xF)


def instr_run(relu=0):
    return (OP_RUN << 14) | (relu << 13)


def instr_store(row, col):
    return (OP_STORE << 14) | (row << 11) | (col << 9)


# ----------------------------------------------------------------------
# Golden model
# ----------------------------------------------------------------------

def golden_matmul(A, wnibbles, relu=0):
    """C = A (3x3 ternary) x W (3x3 E2M1, x2 integer domain).
    wnibbles[col][k] mirrors the LOAD B addressing. Max |C| = 36."""
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for c in range(N):
            s = sum(A[i][k] * e2m1_decode(wnibbles[c][k]) for k in range(N))
            C[i][c] = max(s, 0) if relu else s
    return C


# ----------------------------------------------------------------------
# SPI driver (LSB-first, sampled on SCLK rising edge)
# ----------------------------------------------------------------------

async def spi_send(dut, instr):
    def drive(mosi, cs, sclk):
        dut.ui_in.value = (mosi << PIN_MOSI) | (cs << PIN_CS) | (sclk << PIN_SCLK)

    drive(0, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF)
    for i in range(16):
        bit = (instr >> i) & 1
        drive(bit, 0, 0)
        await ClockCycles(dut.clk, SCLK_HALF)
        drive(bit, 0, 1)
        await ClockCycles(dut.clk, SCLK_HALF)
    drive(0, 1, 0)
    # Gap covers the data_ready pulse and a full RUN (7 cycles)
    await ClockCycles(dut.clk, 12)


async def hw_reset(dut):
    dut.ena.value = 1
    dut.ui_in.value = 1 << PIN_CS   # CS idle high
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


# ----------------------------------------------------------------------
# High-level operations
# ----------------------------------------------------------------------

async def load_operands(dut, A, wnibbles):
    for i in range(N):
        for k in range(N):
            await spi_send(dut, instr_load_a(i, k, A[i][k]))
    for c in range(N):
        for k in range(N):
            await spi_send(dut, instr_load_b(c, k, wnibbles[c][k]))


async def read_result(dut, row, col):
    await spi_send(dut, instr_store(row, col))
    val = int(dut.uo_out.value)
    if val & 0x80:
        val -= 0x100
    return val


async def run_matmul(dut, A, wnibbles, relu=0):
    await load_operands(dut, A, wnibbles)
    await spi_send(dut, instr_run(relu))
    return [[await read_result(dut, i, c) for c in range(N)] for i in range(N)]


def check(dut, got, expected, label):
    for i in range(N):
        for c in range(N):
            assert got[i][c] == expected[i][c], (
                f"{label}: C[{i}][{c}] expected {expected[i][c]}, "
                f"got {got[i][c]} (full: got={got} expected={expected})")


def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, 100, unit="ns").start())


def rand_ternary(rng):
    return rng.choice([-1, 0, 1])


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

@cocotb.test()
async def test_known_matmul(dut):
    """Hand-checked matmul with mixed E2M1 values incl. 0.5 (=1) and 6 (=12)."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[1, -1, 0],
         [0, 1, 1],
         [-1, -1, 1]]
    # wnibbles[col][k]: 0x1=1, 0x9=-1, 0x7=12, 0xF=-12, 0x5=6, 0x0=0
    wn = [[0x1, 0x9, 0x7],
          [0xF, 0x5, 0x0],
          [0x3, 0xB, 0x6]]

    results = await run_matmul(dut, A, wn)
    dut._log.info(f"results: {results}")
    check(dut, results, golden_matmul(A, wn), "known")
    dut._log.info("known matmul PASSED")


@cocotb.test()
async def test_ternary_semantics(dut):
    """+1 adds the weight, -1 subtracts it, 0 skips it."""
    start_clock(dut)
    await hw_reset(dut)

    # Single weight 12 at W[k=1][col=0]; everything else zero
    wn = [[0x0, 0x7, 0x0],
          [0x0, 0x0, 0x0],
          [0x0, 0x0, 0x0]]

    for t, expected in ((1, 12), (-1, -12), (0, 0)):
        A = [[0, t, 0]] * 3
        results = await run_matmul(dut, A, wn)
        for i in range(N):
            assert results[i][0] == expected, (
                f"t={t}: C[{i}][0] expected {expected}, got {results[i][0]}")
    dut._log.info("ternary semantics PASSED")


@cocotb.test()
async def test_relu(dut):
    """RUN with relu=1 clamps negative results to zero; relu=0 does not."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[1, 1, 1]] * 3
    wn = [[0x9, 0x9, 0x9],    # col 0: sum = -3
          [0x1, 0x1, 0x1],    # col 1: sum = +3
          [0xF, 0x0, 0x0]]    # col 2: sum = -12

    plain = await run_matmul(dut, A, wn, relu=0)
    check(dut, plain, golden_matmul(A, wn, relu=0), "no-relu")

    clamped = await run_matmul(dut, A, wn, relu=1)
    check(dut, clamped, golden_matmul(A, wn, relu=1), "relu")
    dut._log.info("relu PASSED")


@cocotb.test()
async def test_not_degenerate(dut):
    """Activation matrices with equal row sums but different ordering
    must give different results — the design must do real dot products,
    not w * sum(acts)."""
    start_clock(dut)
    await hw_reset(dut)

    A1 = [[1, -1, 0]] * 3
    A2 = [[-1, 1, 0]] * 3           # same row sums (0), swapped order
    wn = [[0x1, 0x7, 0x0],
          [0x5, 0x2, 0x9],
          [0x3, 0x0, 0xF]]

    r1 = await run_matmul(dut, A1, wn)
    r2 = await run_matmul(dut, A2, wn)
    check(dut, r1, golden_matmul(A1, wn), "A1")
    check(dut, r2, golden_matmul(A2, wn), "A2")
    assert r1 != r2, ("equal-sum inputs gave identical outputs — "
                      "design collapsed to w*sum(acts)")
    dut._log.info("non-degeneracy PASSED")


@cocotb.test()
async def test_run_clears_accumulators(dut):
    """Each RUN starts from zero — results must not double on rerun."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[1, 1, -1], [0, 1, 0], [-1, -1, -1]]
    wn = [[0x7, 0x5, 0x3], [0xF, 0xD, 0xB], [0x1, 0x2, 0x4]]

    expected = golden_matmul(A, wn)
    await load_operands(dut, A, wn)
    await spi_send(dut, instr_run())
    await spi_send(dut, instr_run())
    results = [[await read_result(dut, i, c) for c in range(N)]
               for i in range(N)]
    check(dut, results, expected, "rerun")
    dut._log.info("accumulator clear PASSED")


@cocotb.test()
async def test_negative_zero(dut):
    """E2M1 -0 (sign=1, code 000) must behave as exact zero."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[1, -1, 1]] * 3
    wn = [[0x8, 0x8, 0x8], [0x8, 0x0, 0x8], [0x0, 0x8, 0x0]]

    results = await run_matmul(dut, A, wn)
    check(dut, results, [[0] * N for _ in range(N)], "neg-zero")
    dut._log.info("negative zero PASSED")


@cocotb.test()
async def test_random(dut):
    """Randomized full-coverage trials against the golden model."""
    start_clock(dut)
    await hw_reset(dut)

    rng = random.Random(0xF4)
    trials = 3 if GL_TEST else 12

    for t in range(trials):
        A = [[rand_ternary(rng) for _ in range(N)] for _ in range(N)]
        wn = [[rng.randint(0, 15) for _ in range(N)] for _ in range(N)]
        relu = rng.randint(0, 1)
        results = await run_matmul(dut, A, wn, relu=relu)
        check(dut, results, golden_matmul(A, wn, relu=relu), f"trial {t}")
        dut._log.info(f"trial {t} OK (relu={relu})")

    dut._log.info(f"random test PASSED ({trials} trials)")
