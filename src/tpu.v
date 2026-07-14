/*
 * NVFP4 ternary mini-TPU core: control + operand memories + 3x3
 * systolic array + result readout mux.
 *
 * Computes C = A x W with A a 3x3 ternary activation matrix and W a
 * 3x3 E2M1 (NVFP4) weight matrix, in the x2 integer domain. Results
 * are exact 7-bit signed values (max |C| = 36), read out one byte at
 * a time via STORE, with optional ReLU latched at RUN.
 */

`default_nettype none

module tpu (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [15:0] instruction,
    output wire        ready_to_send,
    output wire [7:0]  result
);

    wire [62:0] array_data_out;

    wire        array_write_enable;
    wire        array_clear;
    wire        relu_en_latched;
    wire [1:0]  store_row, store_col;

    wire [1:0]  mema_data_in;
    wire        mema_write_enable;
    wire [1:0]  mema_write_line;
    wire [1:0]  mema_write_elem;
    wire [2:0]  mema_read_enable;
    wire [5:0]  mema_read_elem;

    wire [3:0]  memb_data_in;
    wire        memb_write_enable;
    wire [1:0]  memb_write_line;
    wire [1:0]  memb_write_elem;
    wire [2:0]  memb_read_enable;
    wire [5:0]  memb_read_elem;

    wire [5:0]  array_a_in;
    wire [11:0] array_b_in;

    control control_unit (
        .clk                (clk),
        .rst_n              (rst_n),
        .instruction        (instruction),
        .array_write_enable (array_write_enable),
        .array_clear        (array_clear),
        .relu_en_latched    (relu_en_latched),
        .store_row          (store_row),
        .store_col          (store_col),
        .mema_data_in       (mema_data_in),
        .mema_write_enable  (mema_write_enable),
        .mema_write_line    (mema_write_line),
        .mema_write_elem    (mema_write_elem),
        .mema_read_enable   (mema_read_enable),
        .mema_read_elem     (mema_read_elem),
        .memb_data_in       (memb_data_in),
        .memb_write_enable  (memb_write_enable),
        .memb_write_line    (memb_write_line),
        .memb_write_elem    (memb_write_elem),
        .memb_read_enable   (memb_read_enable),
        .memb_read_elem     (memb_read_elem),
        .ready_to_send      (ready_to_send)
    );

    memory_a memory_act (
        .clk          (clk),
        .write_enable (mema_write_enable),
        .write_line   (mema_write_line),
        .write_elem   (mema_write_elem),
        .data_in      (mema_data_in),
        .read_enable  (mema_read_enable),
        .read_elem    (mema_read_elem),
        .data_out     (array_a_in)
    );

    memory_b memory_wgt (
        .clk          (clk),
        .write_enable (memb_write_enable),
        .write_line   (memb_write_line),
        .write_elem   (memb_write_elem),
        .data_in      (memb_data_in),
        .read_enable  (memb_read_enable),
        .read_elem    (memb_read_elem),
        .data_out     (array_b_in)
    );

    array array_inst (
        .clk      (clk),
        .rst_n    (rst_n),
        .we       (array_write_enable),
        .clr      (array_clear),
        .a_in     (array_a_in),
        .b_in     (array_b_in),
        .data_out (array_data_out)
    );

    // ------------------------------------------------------------------
    // Result readout: STORE latches {row, col}; the selected accumulator
    // (optionally ReLU'd) drives `result` until the next STORE.
    // ------------------------------------------------------------------
    wire [6:0] acc [0:8];
    genvar i;
    generate
        for (i = 0; i < 9; i = i + 1) begin : extract_results
            assign acc[i] = array_data_out[7*i +: 7];
        end
    endgenerate

    // Index arithmetic in 4 bits — 2-bit operands would wrap modulo 4
    wire [3:0] sel_idx = {2'b0, store_row} * 4'd3 + {2'b0, store_col};
    wire [6:0] selected = (store_row < 2'd3 && store_col < 2'd3)
        ? acc[sel_idx]
        : 7'd0;

    wire [6:0] relued = (relu_en_latched && selected[6]) ? 7'd0 : selected;

    // Sign-extend the 7-bit accumulator to the 8-bit result port
    assign result = {relued[6], relued};

endmodule
