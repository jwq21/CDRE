import numpy as np
import torch
from torch import nn
from models.containers import Module
from torch.nn import functional as F
from models.transformer.utils import *


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h, dilation=None):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, group_mask=None,
                input_gl=None, memory=None,
                isencoder=None, dilation=None):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        # att[0][0].argmax()
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k) 10 8 50 64
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) 10 8 64 50
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v) 10 8 50 64

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk) 10 8 50 50

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.bool(), -np.inf)

        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq,self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model) 10 50 512
        return out





class MultiHeadAttention(Module):
    """
    Multi-head attention layer with Dropout and Layer Normalization.
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, isenc=None, dilation=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs,
                                                  dilation=dilation)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, dilation=dilation)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)


        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, group_mask=None,
                input_gl=None, memory=None,isencoder=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights, input_gl=input_gl)
            out = queries + self.dropout(torch.relu(out))
        else:
            if isencoder == True:
                out = self.attention(queries, keys, values, attention_mask, attention_weights, group_mask=group_mask,
                                     input_gl=input_gl, memory=memory,isencoder=isencoder)
            else:
                out = self.attention(queries, keys, values, attention_mask, attention_weights,
                                     input_gl=None, memory=memory, isencoder=isencoder)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


class RelationEnhanceAttention(nn.Module):


    def __init__(self, d_model, d_k, d_v, h, dilation=1):

        super(RelationEnhanceAttention, self).__init__()

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        
        self.relation_embed = nn.Embedding(14, d_k)  
        self.relation_proj = nn.Linear(d_k, d_model)
        self.relation_gate = nn.Linear(d_model, 1)
        
        self.relation_classifier = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 14)  
        )
        self.hard_relation = False  
        self.top_k = 14  
        
        self.spatial_fc = nn.Linear(49, 1)
        
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        
        self.spatial_net = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(3, 3), 
                      stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(d_model, affine=False), 
            nn.ReLU(inplace=True)
        )
        
        self.relation_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        
        self.RELATION_NEAR               = 0
        self.RELATION_SURROUNDED         = 1
        self.RELATION_BESIDE             = 2
        self.RELATION_AROUND             = 3
        self.RELATION_BETWEEN            = 4
        self.RELATION_THROUGH            = 5
        self.RELATION_NEXT_TO            = 6
        self.RELATION_ACROSS             = 7
        self.RELATION_CORNER             = 8
        self.RELATION_ALONG              = 9
        self.RELATION_OVER               = 10
        self.RELATION_IN_FRONT_OF        = 11
        self.RELATION_ALONGSIDE          = 12
        self.RELATION_INSIDE             = 13
        # self.RELATION_ABOVE              = 14
        # self.RELATION_RIGHT              = 15
        # self.RELATION_BEHIND             = 16
        # self.RELATION_BELOW              = 17
        # self.RELATION_LEFT               = 18
        # self.RELATION_ADJACENT           = 19
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.xavier_uniform_(self.spatial_fc.weight)
        nn.init.xavier_uniform_(self.relation_proj.weight)
        nn.init.xavier_uniform_(self.relation_gate.weight)
        
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        nn.init.constant_(self.spatial_fc.bias, 0)
        nn.init.constant_(self.relation_proj.bias, 0)
        nn.init.constant_(self.relation_gate.bias, 0)

    def generate_relation_features(self, q_feats, k_feats):

        batch_size, n_q, _ = q_feats.shape
        batch_size, n_k, _ = k_feats.shape
        
        q_expand = q_feats.unsqueeze(2).expand(-1, -1, n_k, -1)  # [B, nq, nk, d]
        k_expand = k_feats.unsqueeze(1).expand(-1, n_q, -1, -1)  # [B, nq, nk, d]
        
        pair_feats = torch.cat([q_expand, k_expand], dim=-1)  # [B, nq, nk, 2d]
        
        relation_logits = self.relation_classifier(pair_feats)  # [B, nq, nk, 14]
        relation_probs = F.softmax(relation_logits, dim=-1)

        if self.hard_relation:
            relation_indices = torch.argmax(relation_probs, dim=-1)  # [B, nq, nk]
            relation_embeddings = self.relation_embed(relation_indices)  # [B, nq, nk, d_k]

        elif self.top_k > 1:
            topk_probs, topk_indices = torch.topk(relation_probs, k=self.top_k, dim=-1)

            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

            topk_embeddings = self.relation_embed(topk_indices)  # [B, nq, nk, k, d_k]

            relation_embeddings = torch.sum(topk_probs.unsqueeze(-1) * topk_embeddings, dim=-2)  
        else:

            all_relation_embeds = self.relation_embed(torch.arange(14, device=q_feats.device))
            relation_embeddings = torch.matmul(relation_probs, all_relation_embeds)
        
        projected_relations = self.relation_proj(relation_embeddings)  # [B, nq, nk, d_k]
        
        relation_feats = self.relation_net(pair_feats)  # [B, nq, nk, d]
        
        combined_relation_feats = relation_feats + projected_relations.to(relation_feats.dtype)
        relation_gates = torch.sigmoid(self.relation_gate(combined_relation_feats))  # [B, nq, nk, 1]
        
        return relation_feats, relation_gates, relation_probs

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, group_mask=None,
                input_gl=None, memory=None, isencoder=None, dilation=None):
       
        batch_size, n_queries = queries.shape[:2]
        x = queries.permute(0, 2, 1)  # [B, d, nq]
        x = x.reshape(x.shape[0], x.shape[1], 7, 7) 
        x = self.spatial_net(x)
        x_spatial = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(0, 2, 1)  # [B, nq, d]
        
        enhanced_queries = x_spatial + queries
        enhanced_keys = x_spatial + keys if keys.shape[1] == queries.shape[1] else keys
        enhanced_values = x_spatial + values if values.shape[1] == queries.shape[1] else values
        
        n_keys = enhanced_keys.shape[1]
        q = self.fc_q(enhanced_queries).view(batch_size, n_queries, self.h, self.d_k).permute(0, 2, 1, 3)  # [B, h, nq, dk]
        k = self.fc_k(enhanced_keys).view(batch_size, n_keys, self.h, self.d_k).permute(0, 2, 3, 1)  # [B, h, dk, nk]
        v = self.fc_v(enhanced_values).view(batch_size, n_keys, self.h, self.d_v).permute(0, 2, 1, 3)  # [B, h, nk, dv]
        
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # [B, h, nq, nk]
        
        relation_feats, relation_gates, relation_probs = self.generate_relation_features(
            enhanced_queries, enhanced_keys
        )
        
        relation_att = relation_gates.squeeze(-1).unsqueeze(1).expand(-1, self.h, -1, -1)  # [B, h, nq, nk]
        
        enhanced_att = att * (1.0 + relation_att)  
        
        if attention_weights is not None:
            enhanced_att = enhanced_att * attention_weights
            
        if attention_mask is not None:
            enhanced_att = enhanced_att.masked_fill(attention_mask.bool(), -np.inf)
            
        if group_mask is not None:
            group_mask_mat = group_mask.masked_fill(group_mask.bool(), torch.tensor(-1e9))
            enhanced_att = enhanced_att + group_mask_mat
        
        att_weights = torch.softmax(enhanced_att, -1)

        if hasattr(self, 'attention_maps'):
            self.attention_maps.append(att_weights)
            
        if hasattr(self, 'relation_distributions'):
            self.relation_distributions.append(relation_probs)
        
        out = torch.matmul(att_weights, v).permute(0, 2, 1, 3).contiguous().view(batch_size, n_queries, self.h * self.d_v)
        out = self.fc_o(out)
        
        return out


