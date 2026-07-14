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
 *   OUTPUT (32 cycles for 16 results x 2 bytes):
 *     uo_out[7:0] = result byte (high then low per 10-bit accumulator)
 *     uio_out[7]  = done flag
 *     uio_out[3:0]= status
 *
 * HDL guide compliance (tinytapeout.com/hdl/important/):
 *   - Exact module port definition matching TT template
 *   - No initial blocks; explicit rst_n reset (tinytapeout.com/hdl/fpga_vs_asic/)
 *   - All outputs assigned: uo_out, uio_out, uio_oe
 *   - _unused wire suppresses warnings for unused inputs
 *   - default_nettype none
 *   - Top module named tt_um_<github_username>_<project>
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

    // Mode decode from uio_in[7:6]
    wire [1:0] mode = uio_in[7:6];

    // Load interface
    wire [3:0] load_weight_a = ui_in[7:4];
    wire [3:0] load_weight_b = ui_in[3:0];
    wire [2:0] load_idx      = uio_in[5:3];

    // Compute interface
    wire [1:0] act_col0 = ui_in[7:6];
    wire [1:0] act_col1 = ui_in[5:4];
    wire [1:0] act_col2 = ui_in[3:2];
    wire [1:0] act_col3 = ui_in[1:0];
    wire       relu_en  = uio_in[0];

    // Control <-> Array wires
    wire        weight_load;
    wire [3:0]  weight_a_w;
    wire [3:0]  weight_b_w;
    wire [2:0]  load_idx_w;
    wire        act_valid;
    wire [1:0]  act0_w, act1_w, act2_w, act3_w;
    wire        acc_clear;

    // 16 accumulator wires (10-bit each) from array to control FSM
    wire [9:0] acc00, acc01, acc02, acc03;
    wire [9:0] acc10, acc11, acc12, acc13;
    wire [9:0] acc20, acc21, acc22, acc23;
    wire [9:0] acc30, acc31, acc32, acc33;

    // Output from control FSM
    wire [7:0] result_byte;
    wire       done;
    wire [3:0] status;

    // Control FSM
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
        .acc00          (acc00),  .acc01          (acc01),
        .acc02          (acc02),  .acc03          (acc03),
        .acc10          (acc10),  .acc11          (acc11),
        .acc12          (acc12),  .acc13          (acc13),
        .acc20          (acc20),  .acc21          (acc21),
        .acc22          (acc22),  .acc23          (acc23),
        .acc30          (acc30),  .acc31          (acc31),
        .acc32          (acc32),  .acc33          (acc33),
        .status         (status)
    );

    // 4x4 Systolic Array
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
        .acc00       (acc00),  .acc01       (acc01),
        .acc02       (acc02),  .acc03       (acc03),
        .acc10       (acc10),  .acc11       (acc11),
        .acc12       (acc12),  .acc13       (acc13),
        .acc20       (acc20),  .acc21       (acc21),
        .acc22       (acc22),  .acc23       (acc23),
        .acc30       (acc30),  .acc31       (acc31),
        .acc32       (acc32),  .acc33       (acc33)
    );

    // Output assignments — all output pins must be driven.
    // Use (* keep *) FFs for uio_oe/uio_out to prevent Yosys from tying
    // them to conb cells whose internal pulldown causes magic LVS to
    // short the pins to VGND (learned from Mini-TPU tt_um_tpu.v).
    assign uo_out = result_byte;

    (* keep = "true" *) reg [7:0] uio_oe_q;
    (* keep = "true" *) reg [7:0] uio_out_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uio_oe_q  <= 8'b0;
            uio_out_q <= 8'b0;
        end else begin
            uio_oe_q  <= (mode == 2'b11) ? 8'b1111_1111 : 8'b0000_0000;
            uio_out_q <= {done, 3'b0, status};
        end
    end
    assign uio_oe  = uio_oe_q;
    assign uio_out = uio_out_q;

    // List all unused inputs to prevent synthesis warnings
    wire _unused = &{ena, 1'b0};

endmodule
