/*
 * Weight memory: 4 columns x 4 E2M1 (NVFP4) 4-bit weights.
 * Read out per column with the skewed wavefront pattern; columns read
 * as 0 (E2M1 zero) when not enabled.
 */

`default_nettype none

module memory_b (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // column 0..3
    input  wire [1:0]  write_elem,      // contraction step 0..3
    input  wire [3:0]  data_in,
    input  wire [3:0]  read_enable,     // per-column
    input  wire [7:0]  read_elem,       // 2-bit step index per column
    output wire [15:0] data_out         // 4 cols x E2M1 nibble
);

    reg [3:0] mem [0:3][0:3];

    always @(posedge clk) begin
        if (write_enable)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : read_col
            wire [1:0] elem = read_elem[2*i +: 2];
            assign data_out[4*i +: 4] = read_enable[i]
                ? mem[i][elem]
                : 4'd0;
        end
    endgenerate

endmodule
