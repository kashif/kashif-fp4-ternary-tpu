/*
 * Weight memory: 3 columns x 3 E2M1 (NVFP4) 4-bit weights.
 * Read out per column with the skewed wavefront pattern; columns read
 * as 0 (E2M1 zero) when not enabled.
 */

`default_nettype none

module memory_b (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // column 0..2
    input  wire [1:0]  write_elem,      // contraction step 0..2
    input  wire [3:0]  data_in,
    input  wire [2:0]  read_enable,     // per-column
    input  wire [5:0]  read_elem,       // 2-bit step index per column
    output wire [11:0] data_out         // 3 cols x E2M1 nibble
);

    reg [3:0] mem [0:2][0:2];

    always @(posedge clk) begin
        if (write_enable && write_line < 2'd3 && write_elem < 2'd3)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : read_col
            wire [1:0] elem = read_elem[2*i +: 2];
            assign data_out[4*i +: 4] = read_enable[i]
                ? mem[i][elem]
                : 4'd0;
        end
    endgenerate

endmodule
