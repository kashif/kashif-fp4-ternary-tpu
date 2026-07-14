/*
 * TT NVFP4 Ternary TPU — Top-Level Tiny Tapeout Module
 *
 * 4x4 weight-stationary systolic array using E2M1 (NVFP4/MXFP4) 4-bit
 * weights with ternary {-1,0,+1} activations. Multiply = MUX-add.
 *
 * Protocol:
 *   uio_in[7:6] = mode: 00=idle, 01=load, 10=compute, 11=output
 *
 *   LOAD (8 cycles): ui_in[7:4]=weight_a, ui_in[3:0]=weight_b, uio_in[5:3]=idx
 *   COMPUTE (17 cycles): ui_in={act3,act2,act1,act0}, uio_in[0]=relu_en
 *   OUTPUT (32 cycles): uo_out=result bytes, uio_out[7]=done, uio_out[3:0]=status
 *
 * HDL guide compliance (tinytapeout.com/hdl/important/):
 *   - Exact module port definition matching TT template
 *   - No initial blocks; explicit rst_n reset (/hdl/fpga_vs_asic/)
 *   - All outputs assigned: uo_out, uio_out, uio_oe
 *   - (* keep *) FFs for uio_oe/uio_out prevent LVS conb shorts
 *     (pattern from Mini-TPU tt_um_tpu.v)
 *   - _unused wire suppresses warnings for unused inputs
 *   - default_nettype none
 *   - Top module: tt_um_<github_username>_<project>
 *
 * References:
 *   - NVFP4: E2M1 + E4M3 scale per 16-element block (NVIDIA Blackwell)
 *   - vLLM Qwix: fuses E4M3+FP32 scales into single FP32 blockwise scale
 *   - PFW TPU: github.com/wangantian/pfw_tpu
 *   - Mini-TPU: github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi
 *   - TT template: github.com/TinyTapeout/ttsky-verilog-template
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

    wire [1:0] mode = uio_in[7:6];

    wire [3:0] load_weight_a = ui_in[7:4];
    wire [3:0] load_weight_b = ui_in[3:0];
    wire [2:0] load_idx      = uio_in[5:3];

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

    wire [9:0] acc00, acc01, acc02, acc03;
    wire [9:0] acc10, acc11, acc12, acc13;
    wire [9:0] acc20, acc21, acc22, acc23;
    wire [9:0] acc30, acc31, acc32, acc33;

    wire [7:0] result_byte;
    wire       done;
    wire [3:0] status;

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

    // uo_out driven directly (result_byte from FSM is already registered)
    assign uo_out = result_byte;

    // (* keep *) FFs for uio_oe/uio_out prevent Yosys conb cells from
    // causing magic LVS shorts to VGND (Mini-TPU pattern)
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

    // Unused inputs
    wire _unused = &{ena, 1'b0};

endmodule
