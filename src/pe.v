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
 *   code |exp mant| => |value|*2
 *   000 => 0, 001 => 1, 010 => 2, 011 => 3
 *   100 => 4, 101 => 6, 110 => 8, 111 => 12
 *
 * Max accumulation: 16 cycles * 12 = 192 => 9 bits signed, use 10 for margin.
 *
 * ASIC optimization (per TT HDL guide /hdl/fpga_vs_asic/):
 *   - Weight magnitude and sign decoded combinationally from weight_code
 *     (only 1 register for weight, not 3 — saves 32 flops across 16 PEs)
 *   - No initial blocks; explicit rst_n reset only on accumulator
 *   - Accumulator is the only register that truly needs reset (weight loaded
 *     before use, pipeline regs don't need reset per Mini-TPU pattern)
 *
 * References:
 *   - NVFP4: E2M1 + E4M3 scale per 16-element block (NVIDIA Blackwell)
 *   - Ternary Weight Networks: Li et al. 2016 (arXiv:1605.04711)
 *   - PFW TPU PE: github.com/wangantian/pfw_tpu (INT8 weight-stationary)
 *   - Mini-TPU PE: github.com/MILOUDIAS (4-bit MAC, no-reset pipeline regs)
 */

`default_nettype none

module pe (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        weight_load,  // load weight this cycle
    input  wire [3:0]  weight_in,    // E2M1 4-bit: {sign, exp[1:0], mant}

    input  wire        act_valid,    // compute this cycle
    input  wire [1:0]  act_in,       // 00=zero, 01=+1, 10=-1, 11=reserved

    input  wire        acc_clear,    // start new block (clear accumulator)

    output wire [9:0]  acc_out       // 10-bit signed accumulator
);

    // Only register: the 4-bit weight code (loaded once, held for block)
    reg [3:0] weight_code;

    always @(posedge clk) begin
        if (weight_load)
            weight_code <= weight_in;
    end

    // Combinational decode: E2M1 magnitude LUT (saves 2 flops per PE)
    // {exp[1:0], mant} -> magnitude*2
    reg [3:0] weight_mag;
    reg       weight_sign;

    always @(*) begin
        weight_sign = weight_code[3];
        case (weight_code[2:0])
            3'b000:  weight_mag = 4'd0;   // 0.0 * 2 = 0
            3'b001:  weight_mag = 4'd1;   // 0.5 * 2 = 1
            3'b010:  weight_mag = 4'd2;   // 1.0 * 2 = 2
            3'b011:  weight_mag = 4'd3;   // 1.5 * 2 = 3
            3'b100:  weight_mag = 4'd4;   // 2.0 * 2 = 4
            3'b101:  weight_mag = 4'd6;   // 3.0 * 2 = 6
            3'b110:  weight_mag = 4'd8;   // 4.0 * 2 = 8
            3'b111:  weight_mag = 4'd12;  // 6.0 * 2 = 12
            default: weight_mag = 4'd0;
        endcase
    end

    // Signed weight value for add/sub
    wire signed [9:0] weight_signed;
    assign weight_signed = weight_sign ? -$signed({6'b0, weight_mag})
                                       :  $signed({6'b0, weight_mag});

    // 10-bit signed accumulator (the only reset register)
    reg signed [9:0] acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 10'sd0;
        else if (acc_clear)
            // Clear and apply first activation in same cycle
            acc <= act_valid ? (act_in == 2'b01 ? weight_signed :
                                act_in == 2'b10 ? -weight_signed : 10'sd0)
                             : 10'sd0;
        else if (act_valid) begin
            case (act_in)
                2'b01:   acc <= acc + weight_signed;   // +weight
                2'b10:   acc <= acc - weight_signed;   // -weight
                default: acc <= acc;                    // 0: nop
            endcase
        end
    end

    assign acc_out = acc;

endmodule
