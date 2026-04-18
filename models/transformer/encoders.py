from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention
from models.transformer.attention import ScaledDotProductAttention


from models.transformer.utils import *



class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None, dilation=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs, dilation=dilation)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, group_mask=None,
                input_gl=None, memory=None, isencoder=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights, group_mask=group_mask,
                         input_gl=input_gl, memory=memory, isencoder=isencoder)
        ff = self.pwff(att)
        return ff

class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout

        self.attention_gate_fc = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model)
        )

        self.film_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 2 * self.d_model)
        )

        self.fusion_dropout = nn.Dropout(p=self.dropout)

        self.layers = nn.Sequential(EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                 identity_map_reordering=identity_map_reordering,
                                                 attention_module=attention_module,
                                                 attention_module_kwargs=attention_module_kwargs, dilation=2),
                                    EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                 identity_map_reordering=identity_map_reordering,
                                                 attention_module=attention_module,
                                                 attention_module_kwargs=attention_module_kwargs, dilation=4),
                                    EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                 identity_map_reordering=identity_map_reordering,
                                                 attention_module=attention_module,
                                                 attention_module_kwargs=attention_module_kwargs, dilation=8)
                                    )

        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=ScaledDotProductAttention,
                                        attention_module_kwargs=attention_module_kwargs)
        self.padding_idx = padding_idx

    def forward(self, input, input_gl=None, isencoder=None, detections_mask=None, memory=None, attention_weights=None):
        # fine module
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
        input = self.mhatt(input, input, input, attention_mask)
        
        B, num_boxes, d_model = input.shape
        
        if input_gl is not None:
            if len(input_gl.shape) == 4:  # [B, 1, 49, 512]
                input_gl = input_gl.squeeze(1).mean(dim=1)  # [B, 512]
            elif len(input_gl.shape) == 3:  # [B, 49, 512]

                input_gl = input_gl.mean(dim=1)  # [B, 512]
            elif len(input_gl.shape) == 2 and input_gl.shape[1] == d_model:
                pass
            else:
                print(f"Warning: Unexpected input_gl shape: {input_gl.shape}")
                if len(input_gl.shape) >= 2:
                    input_gl = input_gl.reshape(B, -1)
                    if input_gl.shape[1] != d_model:

                        input_gl = self.film_mlp[0](input_gl) 

        global_feat_expanded = input_gl.unsqueeze(1).expand(-1, num_boxes, -1)  # [B, num_boxes, d_model]
        
        combined_for_gate = torch.cat([input, global_feat_expanded], dim=-1)  # [B, num_boxes, 2*d_model]
        combined_for_gate = self.fusion_dropout(combined_for_gate)
        attention_weights_gate = torch.sigmoid(self.attention_gate_fc(combined_for_gate))  # [B, num_boxes, d_model]
        gated_input = input * attention_weights_gate  

        film_params = self.film_mlp(input_gl)  # [B, 2*d_model]
        gamma = film_params[:, :self.d_model].unsqueeze(1)  # [B, 1, d_model] 
        beta = film_params[:, self.d_model:].unsqueeze(1)  # [B, 1, d_model] 

        out = gamma * gated_input + beta  # [B, num_boxes, d_model]
        
        group_mask = detections_mask 
        outs = []
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights, group_mask=group_mask, input_gl=input_gl,
                    memory=memory, isencoder=isencoder)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask




class MemoryAugmentedEncoder(MultiLevelEncoder):
    
    def __init__(self, N, padding_idx, d_in=2048, d_clip=512, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model) 
        self.fc_clip = nn.Linear(d_clip, self.d_model) 

        self.dropout_1 = nn.Dropout(p=self.dropout) 
        self.layer_norm_1 = nn.LayerNorm(self.d_model) 

        self.dropout_2 = nn.Dropout(p=self.dropout) 
        self.layer_norm_2 = nn.LayerNorm(self.d_model) 

    def forward(self, input, input_gl=None, detections_mask=None, isencoder=None, attention_weights=None):
        input_gl = F.relu(self.fc_clip(input_gl))
        input_gl = self.dropout_1(input_gl)
        input_gl = self.layer_norm_1(input_gl)

        out = F.relu(self.fc(input))
        out = self.dropout_2(out)
        out = self.layer_norm_2(out)
        return super(MemoryAugmentedEncoder, self).forward(out, input_gl=input_gl, detections_mask=detections_mask,
                                                           isencoder=isencoder, attention_weights=attention_weights)


