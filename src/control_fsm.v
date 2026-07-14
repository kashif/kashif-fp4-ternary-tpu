/*
 * Control FSM for NVFP4 Ternary TPU
 *
 * Protocol over TT pins (8-bit streaming):
 *
 * uio_in[7:6] selects mode:
 *   00 = IDLE
 *   01 = LOAD_WEIGHTS: load 16 E2M1 weights, 2 per cycle, 8 cycles
 *       ui_in[7:4]  = weight_a (E2M1, 4 bits)
 *       ui_in[3:0]  = weight_b (E2M1, 4 bits)
 *       uio_in[5:3] = load_idx (0-7, which pair: row=idx[2:1], col_pair=idx[0])
 *   10 = COMPUTE: stream 16 ternary activations, one per cycle
 *       ui_in[7:6] = act_col0 (2-bit ternary: 00=0, 01=+1, 10=-1)
 *       ui_in[5:4] = act_col1
 *       ui_in[3:2] = act_col2
 *       ui_in[1:0] = act_col3
 *       uio_in[0]  = relu_en (1 = clamp negative outputs to 0)
 *   11 = OUTPUT: read back 16 accumulators
 *       uo_out[7:0] = result bytes (2 per accumulator: high then low)
 *       uio_out[7]  = done (1 when all 32 bytes sent)
 *
 * Accumulator format: 10-bit signed, output as 2 bytes:
 *   byte 0 (even cycle): {6'b0, acc[9:8]}
 *   byte 1 (odd cycle):  acc[7:0]
 *
 * References:
 *   - PFW TPU control: github.com/wangantian/pfw_tpu/src/control_unit.v (IDLE/LOAD/COMPUTE FSM)
 *   - Mini-TPU control: github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi/src/control.v (LOAD/RUN/STORE)
 *   - NVFP4 block size = 16 (NVIDIA Blackwell spec)
 */

`default_nettype none

module control_fsm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,          // uio_in[7:6]

    // Load inputs
    input  wire [3:0]  load_weight_a, // ui_in[7:4]
    input  wire [3:0]  load_weight_b, // ui_in[3:0]
    input  wire [2:0]  load_idx,      // uio_in[5:3]

    // Compute inputs
    input  wire [1:0]  act_col0,      // ui_in[7:6]
    input  wire [1:0]  act_col1,      // ui_in[5:4]
    input  wire [1:0]  act_col2,      // ui_in[3:2]
    input  wire [1:0]  act_col3,      // ui_in[1:0]
    input  wire        relu_en,       // uio_in[0]

    // Output
    output reg  [7:0]  result_byte,   // uo_out
    output reg         done,          // uio_out[7]

    // Array control
    output reg         weight_load,
    output reg  [3:0]  weight_a,
    output reg  [3:0]  weight_b,
    output reg  [2:0]  load_idx_out,
    output reg         act_valid,
    output reg  [1:0]  act0, act1, act2, act3,
    output reg         acc_clear,

    // Accumulator readback from array
    input  wire [9:0]  acc [0:15],

    // Status
    output reg  [3:0]  status         // uio_out[3:0]: state + counters
);

    localparam [1:0] MODE_IDLE    = 2'b00;
    localparam [1:0] MODE_LOAD    = 2'b01;
    localparam [1:0] MODE_COMPUTE = 2'b10;
    localparam [1:0] MODE_OUTPUT  = 2'b11;

    // Compute counter (0-16, one NVFP4 block)
    reg [4:0] compute_cnt;

    // Output counter (0-31, 16 results × 2 bytes)
    reg [5:0] output_cnt;

    // Latched accumulators
    reg [9:0] acc_snap [0:15];

    // Apply ReLU to snapshot
    wire [9:0] acc_relu [0:15];
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : RELU
            assign acc_relu[i] = relu_en && acc_snap[i][9] ? 10'd0 : acc_snap[i];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_cnt  <= 5'd0;
            output_cnt   <= 6'd0;
            weight_load  <= 1'b0;
            weight_a     <= 4'd0;
            weight_b     <= 4'd0;
            load_idx_out <= 3'd0;
            act_valid    <= 1'b0;
            act0         <= 2'b00;
            act1         <= 2'b00;
            act2         <= 2'b00;
            act3         <= 2'b00;
            acc_clear    <= 1'b0;
            result_byte  <= 8'd0;
            done         <= 1'b0;
            status       <= 4'd0;
        end else begin
            // Defaults
            weight_load <= 1'b0;
            act_valid   <= 1'b0;
            acc_clear   <= 1'b0;
            done        <= 1'b0;

            case (mode)
                // ──────────────────────────────────────────────────
                MODE_LOAD: begin
                    weight_load  <= 1'b1;
                    weight_a     <= load_weight_a;
                    weight_b     <= load_weight_b;
                    load_idx_out <= load_idx;
                    status       <= {2'b01, load_idx[2:0]};
                end

                // ──────────────────────────────────────────────────
                MODE_COMPUTE: begin
                    if (compute_cnt == 5'd0) begin
                        // First cycle: clear accumulators, start computing
                        acc_clear   <= 1'b1;
                        act_valid   <= 1'b1;
                        act0        <= act_col0;
                        act1        <= act_col1;
                        act2        <= act_col2;
                        act3        <= act_col3;
                        compute_cnt <= 5'd1;
                        status      <= 4'b1000;
                    end else if (compute_cnt < 5'd16) begin
                        act_valid   <= 1'b1;
                        act0        <= act_col0;
                        act1        <= act_col1;
                        act2        <= act_col2;
                        act3        <= act_col3;
                        compute_cnt <= compute_cnt + 5'd1;
                        status      <= {2'b10, compute_cnt[3:0]};
                    end else begin
                        // Block complete: snapshot accumulators
                        compute_cnt <= 5'd0;
                        status      <= 4'b1010;
                        for (i = 0; i < 16; i = i + 1) begin
                            acc_snap[i] <= acc[i];
                        end
                    end
                end

                // ──────────────────────────────────────────────────
                MODE_OUTPUT: begin
                    if (output_cnt < 6'd32) begin
                        // 16 results × 2 bytes each = 32 cycles
                        // Even cycle (output_cnt[0]=0): high byte
                        // Odd cycle  (output_cnt[0]=1): low byte
                        if (output_cnt[0]) begin
                            result_byte <= acc_relu[output_cnt[5:1]][7:0];
                        end else begin
                            result_byte <= {6'b0, acc_relu[output_cnt[5:1]][9:8]};
                        end
                        output_cnt <= output_cnt + 6'd1;
                        status     <= {2'b11, output_cnt[3:0]};
                    end else begin
                        done       <= 1'b1;
                        output_cnt <= 6'd0;
                        status     <= 4'b1100;
                    end
                end

                // ──────────────────────────────────────────────────
                default: begin
                    // IDLE
                    status <= 4'b0000;
                end
            endcase
        end
    end

endmodule
