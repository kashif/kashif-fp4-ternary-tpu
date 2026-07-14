/*
 * Activation memory: 3 rows x 3 ternary elements (2 bits each).
 * Read out per row with the skewed wavefront pattern; rows read as
 * 00 (= ternary zero) when not enabled.
 */

`default_nettype none

module memory_a (
    input  wire       clk,
    input  wire       write_enable,
    input  wire [1:0] write_line,      // row 0..2
    input  wire [1:0] write_elem,      // element 0..2
    input  wire [1:0] data_in,
    input  wire [2:0] read_enable,     // per-row
    input  wire [5:0] read_elem,       // 2-bit element index per row
    output wire [5:0] data_out         // 3 rows x 2-bit ternary
);

    reg [1:0] mem [0:2][0:2];

    always @(posedge clk) begin
        if (write_enable && write_line < 2'd3 && write_elem < 2'd3)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : read_row
            wire [1:0] elem = read_elem[2*i +: 2];
            assign data_out[2*i +: 2] = read_enable[i]
                ? mem[i][elem]
                : 2'd0;
        end
    endgenerate

endmodule
