/*
 * 4x4 Weight-Stationary Systolic Array for NVFP4 Ternary TPU
 *
 * 16 PEs in 4x4 grid. Weights loaded once, activations broadcast per column.
 *
 * Computation:
 *   PE[i][j] stores W[i][j] (E2M1, 4-bit)
 *   Each cycle, column j receives ternary activation act_j
 *   PE[i][j]: acc[i][j] += act_j * W[i][j]  (MUX-add, no multiplier)
 *   After 16 cycles (one NVFP4 block): acc[i][j] = W[i][j] * sum_k act_k[j]
 *
 * References:
 *   - PFW TPU 2x2: github.com/wangantian/pfw_tpu/src/systolic_array_2x2.v
 *   - Mini-TPU 3x3: github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi/src/array.v
 */

`default_nettype none

module systolic_array_4x4 (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        weight_load,
    input  wire [3:0]  weight_a,
    input  wire [3:0]  weight_b,
    input  wire [2:0]  load_idx,

    input  wire        act_valid,
    input  wire [1:0]  act_col0,
    input  wire [1:0]  act_col1,
    input  wire [1:0]  act_col2,
    input  wire [1:0]  act_col3,

    input  wire        acc_clear,

    output wire [9:0]  acc00, acc01, acc02, acc03,
    output wire [9:0]  acc10, acc11, acc12, acc13,
    output wire [9:0]  acc20, acc21, acc22, acc23,
    output wire [9:0]  acc30, acc31, acc32, acc33
);

    wire [1:0] load_row = load_idx[2:1];
    wire       load_col_hi = load_idx[0];

    // Internal accumulator wires from each PE
    wire [9:0] pe_acc [0:3][0:3];

    // Activation bus
    wire [1:0] act_bus [0:3];
    assign act_bus[0] = act_col0;
    assign act_bus[1] = act_col1;
    assign act_bus[2] = act_col2;
    assign act_bus[3] = act_col3;

    genvar r, c;
    generate
        for (r = 0; r < 4; r = r + 1) begin : ROWS
            for (c = 0; c < 4; c = c + 1) begin : COLS
                wire pe_wload = weight_load
                    && (load_row == r[1:0])
                    && (load_col_hi == c[1]);

                wire [3:0] pe_weight = c[0] ? weight_b : weight_a;

                pe pe_inst (
                    .clk         (clk),
                    .rst_n       (rst_n),
                    .weight_load (pe_wload),
                    .weight_in   (pe_weight),
                    .act_valid   (act_valid),
                    .act_in      (act_bus[c]),
                    .acc_clear   (acc_clear),
                    .acc_out     (pe_acc[r][c])
                );
            end
        end
    endgenerate

    // Route internal wires to output ports
    assign acc00 = pe_acc[0][0];  assign acc01 = pe_acc[0][1];
    assign acc02 = pe_acc[0][2];  assign acc03 = pe_acc[0][3];
    assign acc10 = pe_acc[1][0];  assign acc11 = pe_acc[1][1];
    assign acc12 = pe_acc[1][2];  assign acc13 = pe_acc[1][3];
    assign acc20 = pe_acc[2][0];  assign acc21 = pe_acc[2][1];
    assign acc22 = pe_acc[2][2];  assign acc23 = pe_acc[2][3];
    assign acc30 = pe_acc[3][0];  assign acc31 = pe_acc[3][1];
    assign acc32 = pe_acc[3][2];  assign acc33 = pe_acc[3][3];

endmodule
