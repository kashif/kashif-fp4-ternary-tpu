/*
 * TT NVFP4 Ternary TPU — Top-Level Tiny Tapeout Module
 *
 * A 4x4 weight-stationary systolic array that performs matrix-vector
 * multiplication using E2M1 (NVFP4/MXFP4) 4-bit weights and ternary
 * {-1, 0, +1} activations. The multiply is reduced to add/subtract —
 * no hardware multiplier needed.
 *
 * Protocol:
 *   uio_in[7:6] = mode: 00=idle, 01=load, 10=compute, 11=output
 *
 *   LOAD (8 cycles for 16 weights):
 *     ui_in[7:4]  = weight_a (E2M1 4-bit)
 *     ui_in[3:0]  = weight_b (E2M1 4-bit)
 *     uio_in[5:3] = pair index (0-7)
 *
 *   COMPUTE (16 cycles = one NVFP4 block):
 *     ui_in[7:6]  = act_col0 (ternary: 00=0, 01=+1, 10=-1)
 *     ui_in[5:4]  = act_col1
 *     ui_in[3:2]  = act_col2
 *     ui_in[1:0]  = act_col3
 *     uio_in[0]   = relu_en
 *
 *   OUTPUT (32 cycles for 16 results × 2 bytes):
 *     uo_out[7:0] = result byte (high then low per 10-bit accumulator)
 *     uio_out[7]  = done flag
 *     uio_out[3:0]= status
 *
 * Follows TT HDL guide:
 *   - Exact module port definition (tinytapeout.com/hdl/important/)
 *   - No initial blocks, explicit rst_n reset (tinytapeout.com/hdl/fpga_vs_asic/)
 *   - All outputs assigned (uo_out, uio_out, uio_oe)
 *   - _unused wire for unused inputs
 *
 * References:
 *   - NVFP4: E2M1 + E4M3 scale per 16-element block (NVIDIA Blackwell)
 *   - PFW TPU: github.com/wangantian/pfw_tpu (INT8 systolic, 1x2 tile)
 *   - Mini-TPU: github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi (4-bit, 1x1 tile)
 *   - TT Verilog template: github.com/TinyTapeout/ttsky-verilog-template
 */

`default_nettype none

module tt_um_kashif_fp4_ternary_tpu (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // will go high when the design is enabled
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

    // ─── Mode decode from uio_in[7:6] ───
    wire [1:0] mode = uio_in[7:6];

    // ─── Load interface ───
    wire [3:0] load_weight_a = ui_in[7:4];
    wire [3:0] load_weight_b = ui_in[3:0];
    wire [2:0] load_idx      = uio_in[5:3];

    // ─── Compute interface ───
    wire [1:0] act_col0 = ui_in[7:6];
    wire [1:0] act_col1 = ui_in[5:4];
    wire [1:0] act_col2 = ui_in[3:2];
    wire [1:0] act_col3 = ui_in[1:0];
    wire       relu_en  = uio_in[0];

    // ─── Control ↔ Array wires ───
    wire        weight_load;
    wire [3:0]  weight_a_w;
    wire [3:0]  weight_b_w;
    wire [2:0]  load_idx_w;
    wire        act_valid;
    wire [1:0]  act0_w, act1_w, act2_w, act3_w;
    wire        acc_clear;

    // ─── Accumulator bus (16 × 10-bit) ───
    wire [9:0] acc [0:15];

    // ─── Output from control FSM ───
    wire [7:0] result_byte;
    wire       done;
    wire [3:0] status;

    // ─── Control FSM ───
    control_fsm u_ctrl (
        .clk            (clk),
        .rst_n          (rst_n),
        .mode           (mode),
        .load_weight_a  (load_weight_a),
        .load_weight_b  (load_weight_b),
        .load_idx       (load_idx),
        .act_col0       (act_col0),
        .act_col1       (act_col1),
        .act_col2       (act_col2),
        .act_col3       (act_col3),
        .relu_en        (relu_en),
        .result_byte    (result_byte),
        .done           (done),
        .weight_load    (weight_load),
        .weight_a       (weight_a_w),
        .weight_b       (weight_b_w),
        .load_idx_out   (load_idx_w),
        .act_valid      (act_valid),
        .act0           (act0_w),
        .act1           (act1_w),
        .act2           (act2_w),
        .act3           (act3_w),
        .acc_clear      (acc_clear),
        .acc            (acc),
        .status         (status)
    );

    // ─── 4x4 Systolic Array ───
    systolic_array_4x4 u_array (
        .clk         (clk),
        .rst_n       (rst_n),
        .weight_load (weight_load),
        .weight_a    (weight_a_w),
        .weight_b    (weight_b_w),
        .load_idx    (load_idx_w),
        .act_valid   (act_valid),
        .act_col0    (act0_w),
        .act_col1    (act1_w),
        .act_col2    (act2_w),
        .act_col3    (act3_w),
        .acc_clear   (acc_clear),
        .acc00       (acc[0]),  .acc01       (acc[1]),
        .acc02       (acc[2]),  .acc03       (acc[3]),
        .acc10       (acc[4]),  .acc11       (acc[5]),
        .acc12       (acc[6]),  .acc13       (acc[7]),
        .acc20       (acc[8]),  .acc21       (acc[9]),
        .acc22       (acc[10]), .acc23       (acc[11]),
        .acc30       (acc[12]), .acc31       (acc[13]),
        .acc32       (acc[14]), .acc33       (acc[15])
    );

    // ─── Output assignments ───
    assign uo_out = result_byte;

    // uio_out: [7]=done, [6:4]=unused (0), [3:0]=status
    assign uio_out = {done, 3'b0, status};

    // uio_oe: output mode drives uio_out; other modes listen
    assign uio_oe = (mode == 2'b11) ? 8'b1111_1111 : 8'b0000_0000;

    // ─── Unused inputs ───
    wire _unused = &{ena, 1'b0};

endmodule
