/*
 * Activation memory: 4 rows x 4 ternary elements (2 bits each).
 * Read out per row with the skewed wavefront pattern; rows read as
 * 00 (= ternary zero) when not enabled.
 */

`default_nettype none

module memory_a (
    input  wire       clk,
    input  wire       write_enable,
    input  wire [1:0] write_line,      // row 0..3
    input  wire [1:0] write_elem,      // element 0..3
    input  wire [1:0] data_in,
    input  wire [3:0] read_enable,     // per-row
    input  wire [7:0] read_elem,       // 2-bit element index per row
    output wire [7:0] data_out         // 4 rows x 2-bit ternary
);

    reg [1:0] mem [0:3][0:3];

    always @(posedge clk) begin
        if (write_enable)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : read_row
            wire [1:0] elem = read_elem[2*i +: 2];
            assign data_out[2*i +: 2] = read_enable[i]
                ? mem[i][elem]
                : 2'd0;
        end
    endgenerate

endmodule
