import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

# ==============================================================================
# Helper: Numerically stable Scatter Softmax (Core for preventing NaN)
# ==============================================================================
def stable_scatter_softmax(src, index, num_segments):
    src = torch.clamp(src, min=-10.0, max=10.0)
    src_exp = torch.exp(src)
    out_sum = torch.zeros(num_segments, src.shape[1], src.shape[2], device=src.device)
    out_sum.index_add_(0, index, src_exp)
    output = src_exp / (out_sum[index] + 1e-6)
    return output


# ==============================================================================
# 1. Physics-Informed Gated Message Passing Layer (GateMPL)
#
# Methodology Reference:
# The gated message passing mechanism in this layer is partially inspired by
# and adapted from the Crystal Graph Convolutional Neural Networks (CGCNN).
# Reference: https://github.com/txie-93/cgcnn/blob/master/cgcnn/model.py
# ==============================================================================
class PhysicsGatedLayer(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_node_dim, use_edge_update=False):
        super().__init__()
        self.use_edge_update = use_edge_update
        concat_dim = 2 * in_node_dim + in_edge_dim

        self.W_f = nn.Linear(concat_dim, out_node_dim)
        self.W_s = nn.Linear(concat_dim, out_node_dim)
        self.layer_norm = nn.LayerNorm(out_node_dim)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if use_edge_update:
            self.edge_update = nn.Linear(out_node_dim, in_edge_dim)

    def edge_udf(self, edges):
        z = torch.cat([edges.src['h'], edges.dst['h'], edges.data['h']], dim=1)
        gate = self.sigmoid(self.W_f(z))
        signal = self.activation(self.W_s(z))
        msg = gate * signal
        ret = {'msg': msg}
        if self.use_edge_update:
            delta_e = self.edge_update(signal)
            ret['e_new'] = edges.data['h'] + delta_e
        return ret

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['h'] = edge_feats
            g.apply_edges(self.edge_udf)
            g.update_all(fn.copy_e('msg', 'm'), fn.sum('m', 'neigh_agg'))
            v_new = node_feats + g.ndata['neigh_agg']
            v_new = self.layer_norm(v_new)
            e_new = g.edata['e_new'] if self.use_edge_update else edge_feats
            return v_new, e_new


# ==============================================================================
# 2. Cross-Scale Interaction Module: D2U & U2D Lightweight Nonlinear Residual Mapping
# ==============================================================================
class DualInteractionBlock(nn.Module):
    def __init__(self, micro_dim, meso_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = meso_dim // n_heads
        assert self.head_dim * n_heads == meso_dim, "Hidden dim must be divisible by n_heads"

        self.d2u_q = nn.Linear(meso_dim, meso_dim)
        self.d2u_k = nn.Linear(micro_dim, meso_dim)
        self.d2u_v = nn.Linear(micro_dim, meso_dim)

        self.d2u_update = nn.Sequential(
            nn.Linear(meso_dim * 2, meso_dim),
            nn.Tanh(),
            nn.Linear(meso_dim, meso_dim)
        )

        self.u2d_update = nn.Sequential(
            nn.Linear(micro_dim + meso_dim, micro_dim),
            nn.Tanh(),
            nn.Linear(micro_dim, micro_dim)
        )

        self.reset_parameters()
        self.last_attn_weights = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.d2u_q.weight)
        nn.init.xavier_uniform_(self.d2u_k.weight)
        nn.init.xavier_uniform_(self.d2u_v.weight)

    def down2up(self, g_micro, h_micro, h_meso):
        meso_ids = g_micro.ndata['meso_id']
        valid_mask = (meso_ids >= 0)
        valid_micro_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        target_meso_indices = meso_ids[valid_mask]

        if len(target_meso_indices) == 0:
            return h_meso

        q_full = self.d2u_q(h_meso)
        k_valid = self.d2u_k(h_micro[valid_micro_indices])
        v_valid = self.d2u_v(h_micro[valid_micro_indices])

        q_expanded = q_full[target_meso_indices]

        q_h = q_expanded.view(-1, self.n_heads, self.head_dim)
        k_h = k_valid.view(-1, self.n_heads, self.head_dim)
        v_h = v_valid.view(-1, self.n_heads, self.head_dim)

        attn_score = (q_h * k_h).sum(dim=-1, keepdim=True) / (self.head_dim ** 0.5)
        attn_weights = stable_scatter_softmax(attn_score, target_meso_indices, h_meso.shape[0])

        self.last_attn_weights = attn_weights.detach().cpu()

        msg_h = v_h * attn_weights
        msg = msg_h.view(-1, self.n_heads * self.head_dim)

        agg_meso = torch.zeros_like(h_meso)
        agg_meso.index_add_(0, target_meso_indices, msg)

        z_meso = torch.cat([h_meso, agg_meso], dim=-1)
        h_meso_new = h_meso + self.d2u_update(z_meso)

        return h_meso_new

    def up2down(self, g_micro, h_micro, h_meso):
        meso_ids = g_micro.ndata['meso_id']
        valid_mask = (meso_ids >= 0)
        target_meso_indices = meso_ids[valid_mask]

        if len(target_meso_indices) == 0:
            return h_micro

        curr_micro = h_micro[valid_mask]
        curr_meso_context = h_meso[target_meso_indices]

        z_mic = torch.cat([curr_micro, curr_meso_context], dim=-1)
        residual_update = self.u2d_update(z_mic)

        h_micro_new = h_micro.clone()
        h_micro_new[valid_mask] = curr_micro + residual_update

        return h_micro_new


# ==============================================================================
# 3. Overall Model Architecture
# ==============================================================================
class DualScaleGNN(nn.Module):
    def __init__(self, micro_in_dim, micro_edge_dim, meso_in_dim, meso_edge_dim, hidden_dim, out_dim, layers=3):
        super().__init__()

        # --- 0. Deep Sets Geometric Feature Aggregator ---
        self.geo_embed_dim = 16
        self.vertex_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, self.geo_embed_dim)
        )

        assert micro_in_dim == 20, "Expected micro_in_dim from dataset to be 20"
        new_micro_in_dim = 3 + self.geo_embed_dim + 1 + 4

        # --- 1. Input Normalization ---
        self.micro_node_norm = nn.LayerNorm(new_micro_in_dim)
        self.micro_edge_norm = nn.LayerNorm(micro_edge_dim)
        self.meso_node_norm  = nn.LayerNorm(meso_in_dim)
        self.meso_edge_norm  = nn.LayerNorm(meso_edge_dim)

        # --- 2. Input Embedding ---
        self.micro_node_enc = nn.Linear(new_micro_in_dim, hidden_dim)
        self.micro_edge_enc = nn.Linear(micro_edge_dim, hidden_dim)
        self.meso_node_enc  = nn.Linear(meso_in_dim, hidden_dim)
        self.meso_edge_enc  = nn.Linear(meso_edge_dim, hidden_dim)

        # --- 3. Multi-Layer Interaction Network ---
        # Four sequential steps per layer (corresponding to the paper):
        #   Step 1 — D2U:      Micro -> Meso Aggregation            (interaction.down2up)
        #   Step 2 — meso_mp:  Meso-scale Gated Graph Update
        #   Step 3 — U2D:      Meso -> Micro Feature Broadcast      (interaction.up2down)
        #   Step 4 — micro_mp: Micro-scale Gated Update, assimilating hierarchical context
        # Natural transition between layers, without the issue of continuous repetitive micro_mp.
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                'interaction': DualInteractionBlock(hidden_dim, hidden_dim),
                'meso_mp':     PhysicsGatedLayer(hidden_dim, hidden_dim, hidden_dim, use_edge_update=True),
                'micro_mp':    PhysicsGatedLayer(hidden_dim, hidden_dim, hidden_dim, use_edge_update=True),
            }))

        # --- 4. Readout Layer ---
        self.micro_readout_norm = nn.LayerNorm(hidden_dim)
        self.meso_readout_norm  = nn.LayerNorm(hidden_dim)

        # --- 5. Prediction Head ---
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, g_micro, g_meso):
        # --- Intra-node geometric canonicalization via Deep Sets ---
        raw_feat  = g_micro.ndata['feat']
        cent      = raw_feat[:, 0:3]
        v_coords  = raw_feat[:, 3:15]
        vol_props = raw_feat[:, 15:20]

        v_sets        = v_coords.view(-1, 4, 3)
        v_rel         = v_sets - cent.unsqueeze(1)
        geo_canonical = torch.sum(self.vertex_mlp(v_rel), dim=1)

        micro_n_feat_canonical = torch.cat([cent, geo_canonical, vol_props], dim=1)

        # --- Input Normalization ---
        micro_n_feat = self.micro_node_norm(micro_n_feat_canonical)
        micro_e_feat = self.micro_edge_norm(g_micro.edata['feat'])
        meso_n_feat  = self.meso_node_norm(g_meso.ndata['feat'])

        if g_meso.num_edges() > 0:
            meso_e_feat = self.meso_edge_norm(g_meso.edata['feat'])
        else:
            meso_e_feat = torch.zeros((0, self.meso_edge_enc.in_features), device=meso_n_feat.device)

        # --- Input Embedding ---
        h_micro = self.micro_node_enc(micro_n_feat)
        e_micro = self.micro_edge_enc(micro_e_feat)
        h_meso  = self.meso_node_enc(meso_n_feat)
        e_meso  = self.meso_edge_enc(meso_e_feat) if g_meso.num_edges() > 0 else \
                  torch.zeros((0, h_meso.shape[1]), device=h_meso.device)

        # --- Multi-Layer Interaction ---
        for layer in self.layers:
            # Step 1: D2U — Micro -> Meso Aggregation
            h_meso = layer['interaction'].down2up(g_micro, h_micro, h_meso)

            # Step 2: Meso-scale Gated Graph Update
            if g_meso.num_edges() > 0:
                h_meso, e_meso = layer['meso_mp'](g_meso, h_meso, e_meso)

            # Step 3: U2D — Meso -> Micro Feature Broadcast
            h_micro = layer['interaction'].up2down(g_micro, h_micro, h_meso)

            # Step 4: Micro-scale Gated Update, assimilating hierarchical context
            h_micro, e_micro = layer['micro_mp'](g_micro, h_micro, e_micro)

        # --- Readout ---
        g_micro.ndata['h_out'] = h_micro
        hg_micro = dgl.readout_nodes(g_micro, 'h_out', op='sum')
        hg_micro = self.micro_readout_norm(hg_micro)

        g_meso.ndata['h_out'] = h_meso
        hg_meso = dgl.readout_nodes(g_meso, 'h_out', op='sum')
        hg_meso = self.meso_readout_norm(hg_meso)

        # --- Prediction ---
        return self.predictor(torch.cat([hg_micro, hg_meso], dim=1))