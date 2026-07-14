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
 * Host post-processing:
 *   Apply E4M3 block scale and FP32 tensor scale (NVFP4 dequantization)
 *   Sum across rows for matrix-vector product: y[i] = sum_j acc[i][j]
 *
 * References:
 *   - PFW TPU 2x2: github.com/wangantian/pfw_tpu/src/systolic_array_2x2.v
 *   - Mini-TPU 3x3: github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi/src/array.v
 */

`default_nettype none

module systolic_array_4x4 (
    input  wire        clk,
    input  wire        rst_n,

    // Weight loading: 2 weights per cycle, 8 cycles for 16 weights
    input  wire        weight_load,
    input  wire [3:0]  weight_a,     // first weight of pair
    input  wire [3:0]  weight_b,     // second weight of pair
    input  wire [2:0]  load_idx,     // 0-7: which weight pair

    // Activation broadcast: 4 ternary activations (one per column)
    input  wire        act_valid,
    input  wire [1:0]  act_col0,
    input  wire [1:0]  act_col1,
    input  wire [1:0]  act_col2,
    input  wire [1:0]  act_col3,

    // Accumulator clear
    input  wire        acc_clear,

    // 16 accumulator outputs (10-bit each)
    output wire [9:0]  acc00, acc01, acc02, acc03,
    output wire [9:0]  acc10, acc11, acc12, acc13,
    output wire [9:0]  acc20, acc21, acc22, acc23,
    output wire [9:0]  acc30, acc31, acc32, acc33
);

    // Per-PE weight load enables
    // load_idx maps to (row, col_pair):
    //   idx 0: row 0, cols 0,1
    //   idx 1: row 0, cols 2,3
    //   idx 2: row 1, cols 0,1
    //   ...
    //   idx 7: row 3, cols 2,3
    wire [1:0] load_row = load_idx[2:1];
    wire       load_col_hi = load_idx[0];  // 0=cols 0,1; 1=cols 2,3

    // Generate 4x4 PEs
    genvar r, c;
    generate
        for (r = 0; r < 4; r = r + 1) begin : ROWS
            for (c = 0; c < 4; c = c + 1) begin : COLS
                // Weight load for this PE
                wire pe_wload;
                assign pe_wload = weight_load
                    && (load_row == r[1:0])
                    && (load_col_hi == c[1])   // c=0,1 -> col_hi=0; c=2,3 -> col_hi=1
                    && (c[0] ? 1'b1 : 1'b1);   // placeholder, actual decode below

                // Actually simpler: decode directly
                wire pe_wload_decoded;
                assign pe_wload_decoded = weight_load
                    && (load_row == r[1:0])
                    && (load_col_hi == c[1]);

                // Weight value for this PE
                wire [3:0] pe_weight;
                assign pe_weight = c[0] ? weight_b : weight_a;

                // Activation for this PE (broadcast per column)
                wire [1:0] pe_act;
                assign pe_act = (c == 0) ? act_col0 :
                                (c == 1) ? act_col1 :
                                (c == 2) ? act_col2 :
                                           act_col3;

                // PE instance
                pe pe_inst (
                    .clk         (clk),
                    .rst_n       (rst_n),
                    .weight_load (pe_wload_decoded),
                    .weight_in   (pe_weight),
                    .act_valid   (act_valid),
                    .act_in      (pe_act),
                    .acc_clear   (acc_clear),
                    .acc_out     (
                        r == 0 && c == 0 ? acc00 :
                        r == 0 && c == 1 ? acc01 :
                        r == 0 && c == 2 ? acc02 :
                        r == 0 && c == 3 ? acc03 :
                        r == 1 && c == 0 ? acc10 :
                        r == 1 && c == 1 ? acc11 :
                        r == 1 && c == 2 ? acc12 :
                        r == 1 && c == 3 ? acc13 :
                        r == 2 && c == 0 ? acc20 :
                        r == 2 && c == 1 ? acc21 :
                        r == 2 && c == 2 ? acc22 :
                        r == 2 && c == 3 ? acc23 :
                        r == 3 && c == 0 ? acc30 :
                        r == 3 && c == 1 ? acc31 :
                        r == 3 && c == 2 ? acc32 :
                                           acc33
                    )
                );
            end
        end
    endgenerate

endmodule
