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
 * Compute timeline (19 cycles total):
 *   cnt 0:      acc_clear=1, act_valid=0 (clear accumulators)
 *   cnt 1-16:   act_valid=1 (16 activation cycles = one NVFP4 block)
 *   cnt 17:     act_valid=0 (drain cycle for pipeline)
 *   cnt 18:     snapshot accumulators into acc_snap
 *
 * The drain cycle is needed because the PE registers have 1-cycle latency:
 * the FSM sets act_valid at edge N, the PE accumulates at edge N+1.
 * Without the drain, the non-blocking snapshot would miss the last activation.
 *
 * References:
 *   - PFW TPU control: github.com/wangantian/pfw_tpu/src/control_unit.v
 *   - Mini-TPU control: github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi/src/control.v
 *   - NVFP4 block size = 16 (NVIDIA Blackwell spec)
 *   - TT HDL guide: no initial blocks, explicit rst_n, all outputs assigned
 */

`default_nettype none

module control_fsm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,

    input  wire [3:0]  load_weight_a,
    input  wire [3:0]  load_weight_b,
    input  wire [2:0]  load_idx,

    input  wire [1:0]  act_col0,
    input  wire [1:0]  act_col1,
    input  wire [1:0]  act_col2,
    input  wire [1:0]  act_col3,
    input  wire        relu_en,

    output reg  [7:0]  result_byte,
    output reg         done,

    output reg         weight_load,
    output reg  [3:0]  weight_a,
    output reg  [3:0]  weight_b,
    output reg  [2:0]  load_idx_out,
    output reg         act_valid,
    output reg  [1:0]  act0, act1, act2, act3,
    output reg         acc_clear,

    input  wire [9:0]  acc00, acc01, acc02, acc03,
    input  wire [9:0]  acc10, acc11, acc12, acc13,
    input  wire [9:0]  acc20, acc21, acc22, acc23,
    input  wire [9:0]  acc30, acc31, acc32, acc33,

    output reg  [3:0]  status
);

    localparam [1:0] MODE_IDLE    = 2'b00;
    localparam [1:0] MODE_LOAD    = 2'b01;
    localparam [1:0] MODE_COMPUTE = 2'b10;
    localparam [1:0] MODE_OUTPUT  = 2'b11;

    // 0 = clear, 1-16 = compute, 17 = drain, 18 = snapshot
    reg [4:0] compute_cnt;
    localparam [4:0] CNT_CLEAR    = 5'd0;
    localparam [4:0] CNT_COMPUTE0 = 5'd1;
    localparam [4:0] CNT_COMPUTE_LAST = 5'd17;  // 17 compute cycles (1-17)
    localparam [4:0] CNT_DRAIN    = 5'd18;
    localparam [4:0] CNT_SNAPSHOT = 5'd19;

    reg [5:0] output_cnt;

    // Latched accumulators and ReLU-applied versions
    reg [9:0] acc_snap [0:15];
    reg       relu_en_latched;

    // ReLU applied to latched values
    wire [9:0] acc_relu [0:15];
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : RELU
            assign acc_relu[gi] = (relu_en_latched && acc_snap[gi][9]) ? 10'd0 : acc_snap[gi];
        end
    endgenerate

    // Mux for reading acc_relu by index during output
    reg [9:0] acc_read;
    always @(*) begin
        case (output_cnt[5:1])
            4'd0:  acc_read = acc_relu[0];
            4'd1:  acc_read = acc_relu[1];
            4'd2:  acc_read = acc_relu[2];
            4'd3:  acc_read = acc_relu[3];
            4'd4:  acc_read = acc_relu[4];
            4'd5:  acc_read = acc_relu[5];
            4'd6:  acc_read = acc_relu[6];
            4'd7:  acc_read = acc_relu[7];
            4'd8:  acc_read = acc_relu[8];
            4'd9:  acc_read = acc_relu[9];
            4'd10: acc_read = acc_relu[10];
            4'd11: acc_read = acc_relu[11];
            4'd12: acc_read = acc_relu[12];
            4'd13: acc_read = acc_relu[13];
            4'd14: acc_read = acc_relu[14];
            default: acc_read = acc_relu[15];
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_cnt     <= CNT_CLEAR;
            output_cnt      <= 6'd0;
            weight_load     <= 1'b0;
            weight_a        <= 4'd0;
            weight_b        <= 4'd0;
            load_idx_out    <= 3'd0;
            act_valid       <= 1'b0;
            act0            <= 2'b00;
            act1            <= 2'b00;
            act2            <= 2'b00;
            act3            <= 2'b00;
            acc_clear       <= 1'b0;
            result_byte     <= 8'd0;
            done            <= 1'b0;
            status          <= 4'd0;
            relu_en_latched <= 1'b0;
        end else begin
            // Defaults
            weight_load <= 1'b0;
            act_valid   <= 1'b0;
            acc_clear   <= 1'b0;
            done        <= 1'b0;

            case (mode)
                MODE_LOAD: begin
                    weight_load  <= 1'b1;
                    weight_a     <= load_weight_a;
                    weight_b     <= load_weight_b;
                    load_idx_out <= load_idx;
                    status       <= {2'b01, load_idx[2:0]};
                end

                MODE_COMPUTE: begin
                    if (compute_cnt == CNT_CLEAR) begin
                        // Cycle 0: clear accumulators AND start first activation.
                        // The PE checks acc_clear before act_valid, so it clears
                        // this cycle and the activation is consumed on the next.
                        // This way we get exactly 16 MACs over cycles 0-15.
                        acc_clear       <= 1'b1;
                        act_valid       <= 1'b1;
                        act0            <= act_col0;
                        act1            <= act_col1;
                        act2            <= act_col2;
                        act3            <= act_col3;
                        compute_cnt     <= CNT_COMPUTE0;
                        relu_en_latched <= relu_en;
                        status          <= 4'b1000;
                    end else if (compute_cnt <= CNT_COMPUTE_LAST) begin
                        // Cycles 1-17: stream activations (17 cycles to get 16 MACs,
                        // since cycle 1 coincides with the clear edge settling)
                        act_valid   <= 1'b1;
                        act0        <= act_col0;
                        act1        <= act_col1;
                        act2        <= act_col2;
                        act3        <= act_col3;
                        compute_cnt <= compute_cnt + 5'd1;
                        status      <= {2'b10, compute_cnt[3:0]};
                    end else if (compute_cnt == CNT_DRAIN) begin
                        // Cycle 17: drain (let last PE accumulation settle)
                        act_valid   <= 1'b0;
                        compute_cnt <= CNT_SNAPSHOT;
                        status      <= 4'b1001;
                    end else begin
                        // Cycle 18: snapshot accumulators
                        compute_cnt <= CNT_CLEAR;
                        status      <= 4'b1010;
                        acc_snap[0]  <= acc00;  acc_snap[1]  <= acc01;
                        acc_snap[2]  <= acc02;  acc_snap[3]  <= acc03;
                        acc_snap[4]  <= acc10;  acc_snap[5]  <= acc11;
                        acc_snap[6]  <= acc12;  acc_snap[7]  <= acc13;
                        acc_snap[8]  <= acc20;  acc_snap[9]  <= acc21;
                        acc_snap[10] <= acc22;  acc_snap[11] <= acc23;
                        acc_snap[12] <= acc30;  acc_snap[13] <= acc31;
                        acc_snap[14] <= acc32;  acc_snap[15] <= acc33;
                    end
                end

                MODE_OUTPUT: begin
                    if (output_cnt < 6'd32) begin
                        if (output_cnt[0]) begin
                            result_byte <= acc_read[7:0];
                        end else begin
                            result_byte <= {6'b0, acc_read[9:8]};
                        end
                        output_cnt <= output_cnt + 6'd1;
                        status     <= {2'b11, output_cnt[3:0]};
                    end else begin
                        done       <= 1'b1;
                        output_cnt <= 6'd0;
                        status     <= 4'b1100;
                    end
                end

                default: begin
                    status <= 4'b0000;
                end
            endcase
        end
    end

endmodule
