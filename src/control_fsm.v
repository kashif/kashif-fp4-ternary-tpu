/*
 * Control FSM for NVFP4 Ternary TPU
 *
 * Protocol over TT pins (8-bit streaming):
 *   uio_in[7:6] = mode: 00=idle, 01=load, 10=compute, 11=output
 *
 *   LOAD (8 cycles): ui_in[7:4]=weight_a, ui_in[3:0]=weight_b, uio_in[5:3]=idx
 *   COMPUTE (17 cycles): ui_in={act3,act2,act1,act0} (2-bit ternary each), uio_in[0]=relu_en
 *   OUTPUT (32 cycles): uo_out=result bytes (2 per 10-bit acc), uio_out[7]=done
 *
 * Compute timeline:
 *   cnt 0: clear+first activation (PE clears then applies first MAC)
 *   cnt 1-16: remaining 16 activations (total 17 act_valid cycles = 16 MACs)
 *   cnt 17: drain (let last PE accumulation settle)
 *   cnt 18: snapshot accumulators
 *
 * ASIC optimization (per /hdl/fpga_vs_asic/):
 *   - weight_a/b/load_idx passed combinationally to array (saves 11 flops)
 *   - act0..3 passed combinationally (saves 8 flops)
 *   - Only compute_cnt, output_cnt, acc_snap[16], relu_en_latched need registers
 *
 * References:
 *   - PFW TPU control: github.com/wangantian/pfw_tpu/src/control_unit.v
 *   - Mini-TPU control: github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi/src/control.v
 *   - NVFP4 block size = 16 (NVIDIA Blackwell spec)
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

    // Array control (combinational pass-through, no registers)
    output wire        weight_load,
    output wire [3:0]  weight_a,
    output wire [3:0]  weight_b,
    output wire [2:0]  load_idx_out,
    output wire        act_valid,
    output wire [1:0]  act0, act1, act2, act3,
    output wire        acc_clear,

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

    localparam [4:0] CNT_CLEAR        = 5'd0;
    localparam [4:0] CNT_COMPUTE_LAST = 5'd18;  // 18 compute cycles (1-18) = 18 act_valid
    localparam [4:0] CNT_DRAIN        = 5'd19;
    localparam [4:0] CNT_SNAPSHOT     = 5'd20;

    reg [4:0] compute_cnt;
    reg [5:0] output_cnt;
    reg       relu_en_latched;
    reg       compute_active;

    // Latched accumulators
    reg [9:0] acc_snap [0:15];

    // ReLU applied to latched values
    wire [9:0] acc_relu [0:15];
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : RELU
            assign acc_relu[gi] = (relu_en_latched && acc_snap[gi][9]) ? 10'd0 : acc_snap[gi];
        end
    endgenerate

    // 16:1 mux for output readback
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

    // Registered pass-through to array (aligned with act_valid/acc_clear)
    reg [1:0] act0_r, act1_r, act2_r, act3_r;
    assign weight_load  = (mode == MODE_LOAD);
    assign weight_a     = load_weight_a;
    assign weight_b     = load_weight_b;
    assign load_idx_out = load_idx;
    assign act0         = act0_r;
    assign act1         = act1_r;
    assign act2         = act2_r;
    assign act3         = act3_r;

    // acc_clear and act_valid — fully registered for clean timing
    reg act_valid_r;
    reg acc_clear_r;
    assign act_valid = act_valid_r;
    assign acc_clear = acc_clear_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_cnt      <= CNT_CLEAR;
            output_cnt       <= 6'd0;
            act_valid_r      <= 1'b0;
            acc_clear_r      <= 1'b0;
            act0_r           <= 2'b00;
            act1_r           <= 2'b00;
            act2_r           <= 2'b00;
            act3_r           <= 2'b00;
            result_byte      <= 8'd0;
            done             <= 1'b0;
            status           <= 4'd0;
            relu_en_latched  <= 1'b0;
            compute_active   <= 1'b0;
        end else begin
            // Defaults
            act_valid_r <= 1'b0;
            acc_clear_r <= 1'b0;
            done        <= 1'b0;

            case (mode)
                MODE_LOAD: begin
                    status <= {2'b01, load_idx[2:0]};
                end

                MODE_COMPUTE: begin
                    if (compute_cnt == CNT_CLEAR) begin
                        acc_clear_r     <= 1'b1;
                        act_valid_r     <= 1'b1;
                        act0_r          <= act_col0;
                        act1_r          <= act_col1;
                        act2_r          <= act_col2;
                        act3_r          <= act_col3;
                        compute_cnt     <= 5'd1;
                        relu_en_latched <= relu_en;
                        status          <= 4'b1000;
                    end else if (compute_cnt <= CNT_COMPUTE_LAST) begin
                        act_valid_r <= 1'b1;
                        act0_r      <= act_col0;
                        act1_r      <= act_col1;
                        act2_r      <= act_col2;
                        act3_r      <= act_col3;
                        compute_cnt <= compute_cnt + 5'd1;
                        status      <= {2'b10, compute_cnt[3:0]};
                    end else if (compute_cnt == CNT_DRAIN) begin
                        compute_cnt <= CNT_SNAPSHOT;
                        status      <= 4'b1001;
                    end else begin
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
                        result_byte <= output_cnt[0] ? acc_read[7:0]
                                                     : {6'b0, acc_read[9:8]};
                        output_cnt  <= output_cnt + 6'd1;
                        status      <= {2'b11, output_cnt[3:0]};
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
