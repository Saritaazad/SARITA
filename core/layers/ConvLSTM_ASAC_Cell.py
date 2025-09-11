import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=8, spatial_dim=86):
        super(SpatialAttention, self).__init__()

        self.input_channels = input_channels  # Single channel for Moran's matrix
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.spatial_dim = spatial_dim  # For reshaping after attention calculations

        # Query, Key, and Value layers
        self.query_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)  # 1 -> 8
        self.key_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)    # 1 -> 8
        self.value_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)  # 1 -> 8

        # Output projection to match input channels
        self.output_conv = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)  # 8 -> 1

    def forward(self, x):
        """
        Forward pass for spatial attention.

        Args:
            x: Input tensor of shape [B, C, H, W], where:
                B: Batch size
                C: Channels (should be 1 for Moran's matrix)
                H, W: Spatial dimensions (86 x 86 for Moran's matrix)

        Returns:
            output: Tensor of shape [B, C, H, W] with spatial attention applied
        """
        batch_size, channels, height, width = x.size()

        # Ensure input has the expected single channel
        assert channels == self.input_channels, f"Expected {self.input_channels} channels, got {channels}"

        # Compute Query, Key, and Value
        Q = self.query_conv(x)  # [B, hidden_channels, H, W]
        K = self.key_conv(x)    # [B, hidden_channels, H, W]
        V = self.value_conv(x)  # [B, hidden_channels, H, W]

        # Reshape for matrix multiplication: [B, hidden_channels, H*W]
        Q = Q.view(batch_size, self.hidden_channels, -1)
        K = K.view(batch_size, self.hidden_channels, -1).transpose(-2, -1)  # Transpose for compatibility
        V = V.view(batch_size, self.hidden_channels, -1)

        # Attention score: [B, hidden_channels, H*W] x [B, hidden_channels, H*W]^T -> [B, H*W, H*W]
        attention_scores = torch.matmul(Q, K) / (self.hidden_channels ** 0.5)  # Scale by sqrt(d_k)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # Softmax across spatial dims

        # Weighted value: [B, H*W, H*W] x [B, hidden_channels, H*W] -> [B, hidden_channels, H*W]
        weighted_values = torch.matmul(attention_weights, V)

        # Reshape back to [B, hidden_channels, H, W]
        weighted_values = weighted_values.view(batch_size, self.hidden_channels, height, width)

        # Project back to the original number of channels (e.g., 1 for Moran's matrix)
        output = self.output_conv(weighted_values)  # [B, output_channels, H, W]

        return output

class ConvLSTM_SAC_Cell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTM_SAC_Cell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([3, width, width])
        )
        self.attention_conv = nn.Conv2d(
            #num_hidden * 4, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding
            1, 1, kernel_size=1, padding=0
        )
        self.spatial_attention = SpatialAttention(input_channels=1, output_channels=1, hidden_channels=8, spatial_dim=width)

        self.Wci = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcf = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcg = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wco = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()

    def forward(self, x_t, h_t, c_t, m, m_t):
        # attention_weights = self.attention_conv(m)
        # attention_weights = nn.functional.softmax(attention_weights, dim=1)
        m = self.spatial_attention(m)
        #m = attention_weights*m

        x_concat = self.conv_x(x_t).cuda()
        h_concat = self.conv_h(h_t).cuda()

        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + self.Wci * c_t)

        f_t = torch.sigmoid(f_x + f_h + self.Wcf * c_t + self._forget_bias)

        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h + self.Wco * c_new)

        h_new = o_t * torch.tanh(c_new)

        combined = self.conv(torch.cat([m, m_t], dim=1))
        mo, mg, mi = torch.split(combined, 1, dim=1)

        mi = torch.sigmoid(mi)
        m_new = (1 - mi) * m + mi * torch.tanh(mg)

        h_new = (1 - torch.sigmoid(m_new)) * x_t + torch.sigmoid(m_new) * h_new

        return h_new, c_new, m_new
