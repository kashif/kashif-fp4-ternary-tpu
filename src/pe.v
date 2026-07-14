/*
 * Processing Element: NVFP4 Ternary-Activation MAC
 *
 * Weight-stationary: E2M1 4-bit weight loaded once, held for block.
 * Activation: ternary {-1, 0, +1} broadcast to all PEs in a column.
 *
 * The "multiply" is a MUX-add (no hardware multiplier):
 *   act = +1 => acc += weight_val
 *   act = -1 => acc -= weight_val
 *   act =  0 => acc unchanged
 *
 * E2M1 decode (scaled x2 to integer domain):
 *   {exp[1:0], mant} => |value|*2
 *   000 => 0, 001 => 1, 010 => 2, 011 => 3
 *   100 => 4, 101 => 6, 110 => 8, 111 => 12
 *
 * Max accumulation: 16 cycles * 12 = 192 => 9 bits signed, use 10 for margin.
 *
 * ASIC optimization (per TT HDL guide /hdl/fpga_vs_asic/):
 *   - Weight magnitude and sign decoded combinationally (saves 2 flops/PE)
 *   - No initial blocks; explicit rst_n reset on accumulator only
 *   - Weight register doesn't need reset (loaded before use)
 *
 * References:
 *   - NVFP4: E2M1 + E4M3 scale per 16-element block (NVIDIA Blackwell)
 *   - Ternary Weight Networks: Li et al. 2016 (arXiv:1605.04711)
 *   - PFW TPU PE: github.com/wangantian/pfw_tpu
 *   - Mini-TPU PE: github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi
 */

`default_nettype none

module pe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        weight_load,
    input  wire [3:0]  weight_in,    // E2M1: {sign, exp[1:0], mant}
    input  wire        act_valid,
    input  wire [1:0]  act_in,       // 00=0, 01=+1, 10=-1
    input  wire        acc_clear,
    output wire [9:0]  acc_out
);

    reg [3:0] weight_code;
    always @(posedge clk) begin
        if (weight_load)
            weight_code <= weight_in;
    end

    // Combinational E2M1 decode
    wire       w_sign = weight_code[3];
    wire [3:0] w_mag;
    assign w_mag = weight_code[2:0] == 3'b000 ? 4'd0  :
                   weight_code[2:0] == 3'b001 ? 4'd1  :
                   weight_code[2:0] == 3'b010 ? 4'd2  :
                   weight_code[2:0] == 3'b011 ? 4'd3  :
                   weight_code[2:0] == 3'b100 ? 4'd4  :
                   weight_code[2:0] == 3'b101 ? 4'd6  :
                   weight_code[2:0] == 3'b110 ? 4'd8  : 4'd12;

    wire signed [9:0] w_val = w_sign ? -$signed({6'b0, w_mag})
                                      : $signed({6'b0, w_mag});

    // Compute product combinationally: act * weight
    // act=+1 => +w_val, act=-1 => -w_val, act=0 => 0
    wire signed [9:0] product;
    assign product = (act_in == 2'b01) ?  w_val :
                     (act_in == 2'b10) ? -w_val : 10'sd0;

    // 10-bit signed accumulator (only register needing reset)
    reg signed [9:0] acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 10'sd0;
        else if (acc_clear)
            acc <= act_valid ? product : 10'sd0;
        else if (act_valid)
            acc <= acc + product;
    end

    assign acc_out = acc;

endmodule
