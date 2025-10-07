import torch
import torch.nn as nn
from layers.FastKAN import FastKAN

import matplotlib.pyplot as plt
class Model(nn.Module):
    def __init__(self, configs, offset=3):
        super(Model, self).__init__()  
        self.seq_len = configs.seq_len
        self.offset = offset
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.use_tq = True  # ablation parameter, default: True
        self.channel_aggre = True   # ablation parameter, default: True
        if self.use_tq:
            self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5)

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        
        
        self.ffc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

        #原
        self.FastKAN = FastKAN(configs, layers_hidden=[
            int(self.seq_len / self.offset), 
            self.d_model, 
            int(self.seq_len / self.offset)
        ])

        # Create multiple attention layers dynamically based on offset
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=int(self.seq_len / self.offset), 
                num_heads=4, 
                batch_first=True, 
                dropout=0.5
            ) for _ in range(self.offset)
        ])

    def forward(self, x, cycle_index):

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # b,s,c -> b,c,s
        x_input = x.permute(0, 2, 1)#torch.Size([128, 7, 96])


        subsequences = [
            x_input[..., i::self.offset] for i in range(self.offset)
        ]
        # Apply shared FastKAN encoder
        kan_encoded = [self.FastKAN(seq) for seq in subsequences]

        # Apply independent attention layers
        attended = [
            self.attentions[i](seq, seq, seq)[0] for i, seq in enumerate(kan_encoded)
        ]

        # Concatenate attended results back to original order
        B, C, L = attended[0].shape
        x_cat = torch.zeros(B, C, L * self.offset, device=x.device, dtype=x.dtype)
        for i in range(self.offset):
            x_cat[..., i::self.offset] = attended[i]


        if self.use_tq:
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=x_cat, key=x_input, value=x_input)[0]
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=x_input, key=x_input, value=x_input)[0]
            else:
                channel_information = 0

        input = self.input_proj(x_input+channel_information)

        hidden = self.ffc(input)

        output = self.output_proj(hidden+input).permute(0, 2, 1)

        # instance denorm
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output


