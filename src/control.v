/*
 * Control unit — ported from the reference mini-TPU control.v (via the
 * companion Int7+1 design), 16-bit instruction, E2M1/ternary operands.
 *
 * Instruction format (16 bits, sent LSB-first over SPI):
 *
 *   [15:14] opcode: 00=NOP, 01=RUN, 10=LOAD, 11=STORE
 *
 *   LOAD:  [13]    mem_select (0 = A activations, 1 = B weights)
 *          [12:11] line  (A: row 0-3, B: column 0-3)
 *          [9:8]   elem  (element / contraction step 0-3)
 *          [7:0]   imm   (A: ternary in imm[1:0], B: E2M1 in imm[3:0])
 *
 *   RUN:   [13] relu_en. Clears accumulators, streams the skewed
 *          wavefront for 10 cycles (N = K = 4 contraction).
 *
 *   STORE: [12:11] row, [10:9] col
 *          Latches the selection; result output holds until next STORE.
 *
 * Memories A and B share the same skewed read pattern (line i active
 * during counter in [i+1, i+3], element walking 0,1,2), identical to
 * the reference.
 */

`default_nettype none

module control (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] instruction,

    output wire        array_write_enable,
    output wire        array_clear,
    output reg         relu_en_latched,
    output reg  [1:0]  store_row,
    output reg  [1:0]  store_col,

    output wire [1:0]  mema_data_in,
    output wire        mema_write_enable,
    output wire [1:0]  mema_write_line,
    output wire [1:0]  mema_write_elem,
    output wire [3:0]  mema_read_enable,
    output wire [7:0]  mema_read_elem,

    output wire [3:0]  memb_data_in,
    output wire        memb_write_enable,
    output wire [1:0]  memb_write_line,
    output wire [1:0]  memb_write_elem,
    output wire [3:0]  memb_read_enable,
    output wire [7:0]  memb_read_elem,

    output reg         ready_to_send
);

    localparam [1:0] RUN   = 2'b01;
    localparam [1:0] LOAD  = 2'b10;
    localparam [1:0] STORE = 2'b11;

    // Wavefront counter: useful range 1..10. The last product lands at
    // t = (N-1) + 1 + (K-1) + (N-1) = 10 for N = K = 4.
    reg [3:0] counter;

    wire [1:0] opcode     = instruction[15:14];
    wire       mem_select = instruction[13];
    wire [1:0] line       = instruction[12:11];
    wire [1:0] elem       = instruction[9:8];
    wire [3:0] imm        = instruction[3:0];

    // Bits reserved by the shared ISA layout but unused in this design
    wire _unused_instr = &{instruction[10], instruction[7:4], 1'b0};

    wire is_load  = (opcode == LOAD);
    wire is_store = (opcode == STORE);
    wire is_run   = (opcode == RUN) || (counter > 0);

    // Latch the ReLU flag on each RUN; applied at readout.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            relu_en_latched <= 1'b0;
        else if (opcode == RUN && counter == 4'd0)
            relu_en_latched <= instruction[13];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter       <= 4'd0;
            ready_to_send <= 1'b0;
        end else if (counter == 4'd10) begin
            counter       <= 4'd0;
            ready_to_send <= 1'b1;
        end else if (is_run)
            counter <= counter + 4'd1;
        else
            ready_to_send <= 1'b0;
    end

    // ------------------------------------------------------------------
    // Shared skewed read pattern (identical for A rows and B columns):
    // line i streams its 4 elements during counter in [i+1, i+4].
    // ------------------------------------------------------------------
    wire [3:0] read_enable_shared;
    wire [7:0] read_elem_shared;

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : read_pattern_gen
            assign read_enable_shared[i] = (counter > i) && (counter < (i + 5));
            assign read_elem_shared[2*i +: 2] =
                (counter == (i + 1)) ? 2'd0 :
                (counter == (i + 2)) ? 2'd1 :
                (counter == (i + 3)) ? 2'd2 :
                (counter == (i + 4)) ? 2'd3 : 2'd0;
        end
    endgenerate

    assign mema_read_enable = read_enable_shared;
    assign memb_read_enable = read_enable_shared;
    assign mema_read_elem   = read_elem_shared;
    assign memb_read_elem   = read_elem_shared;

    // ------------------------------------------------------------------
    // Write path
    // ------------------------------------------------------------------
    wire load_a = is_load && !mem_select;
    wire load_b = is_load &&  mem_select;

    assign mema_data_in      = imm[1:0];
    assign mema_write_enable = load_a;
    assign mema_write_line   = line;
    assign mema_write_elem   = elem;

    assign memb_data_in      = imm;
    assign memb_write_enable = load_b;
    assign memb_write_line   = line;
    assign memb_write_elem   = elem;

    // ------------------------------------------------------------------
    // STORE: latch result selection so uo_out holds a stable value
    // between SPI transactions (instruction is a 1-cycle pulse).
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            store_row <= 2'd0;
            store_col <= 2'd0;
        end else if (is_store) begin
            store_row <= instruction[12:11];
            store_col <= instruction[10:9];
        end
    end

    assign array_write_enable = is_run;
    // Clear accumulators on the RUN-issue cycle; the wavefront reads are
    // still disabled then (counter == 0), so no products are lost.
    assign array_clear        = (opcode == RUN) && (counter == 4'd0);

endmodule
