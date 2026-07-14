/*
 * 3x3 output-stationary systolic array, E2M1 x ternary edition.
 *
 * Structure follows the reference mini-TPU array.v: activations flow
 * right (2-bit ternary pipes), weights flow down (4-bit E2M1 pipes),
 * results accumulate in place in 7-bit exact accumulators.
 */

`default_nettype none

module array (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        we,
    input  wire        clr,

    input  wire [5:0]  a_in,     // 3 rows x ternary activation
    input  wire [11:0] b_in,     // 3 cols x E2M1 weight
    output wire [62:0] data_out  // 9 accumulators x 7 bits, row-major
);

    wire [1:0] a_pipe [0:2][0:3];
    wire [3:0] b_pipe [0:3][0:2];
    wire [6:0] c_bus  [0:2][0:2];

    genvar row, col;
    generate
        for (row = 0; row < 3; row = row + 1) begin : map_a_in
            assign a_pipe[row][0] = a_in[2*row +: 2];
        end
        for (col = 0; col < 3; col = col + 1) begin : map_b_in
            assign b_pipe[0][col] = b_in[4*col +: 4];
        end
    endgenerate

    generate
        for (row = 0; row < 3; row = row + 1) begin : ROWS
            for (col = 0; col < 3; col = col + 1) begin : COLS
                pe pe_inst (
                    .clk   (clk),
                    .rst_n (rst_n),
                    .we    (we),
                    .clr   (clr),
                    .a_in  (a_pipe[row][col]),
                    .b_in  (b_pipe[row][col]),
                    .a_out (a_pipe[row][col+1]),
                    .b_out (b_pipe[row+1][col]),
                    .c_out (c_bus [row][col])
                );
            end
        end
    endgenerate

    generate
        for (row = 0; row < 3; row = row + 1) begin : flat_row
            for (col = 0; col < 3; col = col + 1) begin : flat_col
                assign data_out[7*(row*3+col) +: 7] = c_bus[row][col];
            end
        end
    endgenerate

endmodule
