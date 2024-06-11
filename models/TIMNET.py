import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        input = input.permute(0, 2, 1)
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            result = result[:, :, : -self.__padding]
        result = result.permute(0, 2, 1)
        return result


class SpatialDropout1d(nn.Dropout):
    def __init__(self, p=0.1):
        super(SpatialDropout1d, self).__init__()
        self.p = p

    def forward(self, x):
        x = x.permute(0, 2, 1)  # convert to [batch, feature, timestep]
        x = F.dropout1d(x, self.p, training=self.training)
        x = x.permute(0, 2, 1)
        return x


# class SpatialDropout1d(nn.Module):
#     def __init__(self, p=0.1):
#         super(SpatialDropout1d, self).__init__()
#         self.p = p
#     def forward(self, x):
#         x = F.dropout1d(x, self.p, training=self.training)
#         return x


class TemporalAwareBlock(nn.Module):
    def __init__(
        self, nb_filters, kernel_size, dilation_rate, dropout_rate, activation
    ):
        super(TemporalAwareBlock, self).__init__()
        self.conv1 = CausalConv1d(
            nb_filters, nb_filters, kernel_size, dilation=dilation_rate
        )
        self.bn1 = nn.LazyBatchNorm1d()
        self.activation = getattr(F, activation)
        self.dropout1 = SpatialDropout1d(dropout_rate)

        self.conv2 = CausalConv1d(
            nb_filters, nb_filters, kernel_size, dilation=dilation_rate
        )
        self.bn2 = nn.LazyBatchNorm1d()
        self.dropout2 = SpatialDropout1d(dropout_rate)
        # self.attention = nn.MultiheadAttention(nb_filters, 8, batch_first=True)

        self.sigmoid = nn.Sigmoid()
        self.project = nn.Conv1d(nb_filters, nb_filters, kernel_size=1)

    def forward(self, x):
        original_x = x

        # 1.1
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        # x = self.dropout1(x)

        # 2.1
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        # x = self.dropout2(x)

        if original_x.shape[1] != x.shape[1]:
            original_x = self.project(original_x)

        x = self.sigmoid(x)
        F_x = original_x * x
        return F_x


class TIMNET(nn.Module):
    def __init__(
        self,
        nb_filters=64,
        kernel_size=2,
        nb_stacks=1,
        dilations=None,
        activation="relu",
        dropout_rate=0.1,
        return_sequences=True,
    ):
        super(TIMNET, self).__init__()
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = (
            dilations if dilations is not None else [2**i for i in range(8)]
        )
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.conv_forward = CausalConv1d(
            self.nb_filters, self.nb_filters, kernel_size=1, dilation=1
        )
        self.conv_backward = CausalConv1d(
            self.nb_filters, self.nb_filters, kernel_size=1, dilation=1
        )

        self.temporal_blocks_forward = nn.ModuleList()
        self.temporal_blocks_backward = nn.ModuleList()

        for s in range(self.nb_stacks):
            for dilation in self.dilations:
                self.temporal_blocks_forward.append(
                    TemporalAwareBlock(
                        nb_filters, kernel_size, dilation, dropout_rate, activation
                    )
                )
                self.temporal_blocks_backward.append(
                    TemporalAwareBlock(
                        nb_filters, kernel_size, dilation, dropout_rate, activation
                    )
                )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def build(self, input):
        print("aaaa", input.shape)
        self.conv_forward = CausalConv1d(
            input.shape[1], self.nb_filters, kernel_size=1, dilation=1
        ).to(input.device)
        self.conv_backward = CausalConv1d(
            input.shape[1], self.nb_filters, kernel_size=1, dilation=1
        ).to(input.device)

    def forward(self, inputs):
        if not hasattr(self, "conv_forward"):
            self.build(inputs)
        forward = inputs
        backward = torch.flip(inputs, dims=[1])

        forward_convd = self.conv_forward(forward)
        # print(forward.shape, forward_convd.shape)
        backward_convd = self.conv_backward(backward)

        final_skip_connection = []

        skip_out_forward = forward_convd
        skip_out_backward = backward_convd

        for i, (block_forward, block_backward) in enumerate(
            zip(self.temporal_blocks_forward, self.temporal_blocks_backward)
        ):
            skip_out_forward = block_forward(skip_out_forward)
            skip_out_backward = block_backward(skip_out_backward)

            temp_skip = skip_out_forward + skip_out_backward  # b, nb_filters, c
            temp_skip = self.global_avg_pool(temp_skip.permute(0, 2, 1)).view(
                inputs.size(0), 1, -1
            )  # b, 1, nb_filters
            final_skip_connection.append(temp_skip)

        output_2 = final_skip_connection[0]
        for i, item in enumerate(final_skip_connection):
            if i == 0:
                continue
            output_2 = torch.cat((output_2, item), dim=-2)

        x = output_2  # b, nb_step, nb_filters
        return x


class WeightLayer(nn.Module):
    def __init__(self):
        super(WeightLayer, self).__init__()
        self.kernel = nn.Parameter(torch.Tensor(8, 1))
        nn.init.uniform_(self.kernel)  # Use uniform initialization

    def build(self, x):
        input_shape = x.shape
        self.kernel = nn.Parameter(torch.Tensor(input_shape[1], 1)).to(x.device)
        nn.init.uniform_(self.kernel)  # Use uniform initialization

    def forward(self, x):
        if not hasattr(self, "kernel"):
            self.build(x)
        tempx = x.transpose(1, 2)
        x = torch.matmul(tempx, self.kernel)
        x = torch.squeeze(x, dim=-1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


# Example usage:
# inputs = torch.randn(8, 39, 128)  # Batch size 8, 1 channel, 128 time steps
# model = TIMNET()
# weight_layer = WeightLayer()
# output = model(inputs)
# output = weight_layer(output)
# print(output.shape)
