/*
 * Processing Element: E2M1 (NVFP4) weight x ternary activation MAC
 * (output-stationary, no hardware multiplier).
 *
 * Dataflow follows the reference mini-TPU PE (activations flow right,
 * weights flow down; both streams change every cycle — real dot
 * products, accumulated in place).
 *
 * Weight: E2M1 4-bit {sign, exp[1:0], mant}, decoded to the x2 integer
 * domain: 0,1,2,3,4,6,8,12 with sign. Activation: ternary 2-bit
 * (00=0, 01=+1, 10=-1, 11 reserved=0). The "multiply" is a mux-add:
 *   +1 -> acc += w,  -1 -> acc -= w,  0 -> acc unchanged.
 *
 * K = 3 contraction: max |acc| = 3 * 12 = 36, exact in 7-bit signed.
 *
 * Accumulators have async reset; pipeline regs are no-reset dfxtp
 * (area — reference pattern). Gate-level X from the no-reset regs is
 * flushed by a zero-operand preheat RUN in the GL testbench.
 */

`default_nettype none

module pe (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       we,      // shift pipes + accumulate
    input  wire       clr,     // clear accumulator (start of RUN)
    input  wire [1:0] a_in,    // ternary activation
    input  wire [3:0] b_in,    // E2M1 weight {sign, exp[1:0], mant}
    output wire [1:0] a_out,   // activation passed right
    output wire [3:0] b_out,   // weight passed down
    output wire [6:0] c_out    // accumulated result
);

    // Pipeline regs without reset (smaller dfxtp cells, reference
    // pattern). The sim-only initial keeps them 0 in RTL simulation;
    // gate-level tests flush power-up X with a zero-operand preheat
    // RUN (see test.py gl_preheat / reference REPORT.md).
    reg [1:0] a_reg;
    reg [3:0] b_reg;
    reg signed [6:0] c_reg;

`ifndef SYNTHESIS
    initial begin
        a_reg = 2'd0;
        b_reg = 4'd0;
    end
`endif

    // E2M1 magnitude decode (x2 integer domain)
    wire [3:0] w_mag = b_in[2:0] == 3'b000 ? 4'd0  :
                       b_in[2:0] == 3'b001 ? 4'd1  :
                       b_in[2:0] == 3'b010 ? 4'd2  :
                       b_in[2:0] == 3'b011 ? 4'd3  :
                       b_in[2:0] == 3'b100 ? 4'd4  :
                       b_in[2:0] == 3'b101 ? 4'd6  :
                       b_in[2:0] == 3'b110 ? 4'd8  : 4'd12;

    wire signed [6:0] w_val = b_in[3] ? -$signed({3'b0, w_mag})
                                      :  $signed({3'b0, w_mag});

    // Ternary mux-add: no multiplier anywhere
    wire signed [6:0] prod = (a_in == 2'b01) ?  w_val :
                             (a_in == 2'b10) ? -w_val : 7'sd0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            c_reg <= 7'sd0;
        else if (clr)
            c_reg <= 7'sd0;
        else if (we)
            c_reg <= c_reg + prod;
    end

    always @(posedge clk) begin
        if (we) begin
            a_reg <= a_in;
            b_reg <= b_in;
        end
    end

    assign a_out = a_reg;
    assign b_out = b_reg;
    assign c_out = c_reg;

endmodule
