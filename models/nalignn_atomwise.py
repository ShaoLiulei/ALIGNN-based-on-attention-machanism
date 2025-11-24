"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

from typing import Tuple, Union
from torch.autograd import grad
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn import AvgPooling
import torch
from config import ALIGNNAtomWiseConfig
# from dgl.nn.functional import edge_softmax
from typing import Literal
from torch import nn
from torch.nn import functional as F
from models.utils import (
    RBFExpansion,
    compute_cartesian_coordinates,
    compute_pair_vector_and_distance,
    MLPLayer,
)
from graphs import compute_bond_cosines
from utils import BaseSettings


# class ALIGNNAtomWiseConfig(BaseSettings):
#     """Hyperparameter schema for jarvisdgl.models.alignn."""
#
#     name: Literal["alignn_atomwise_gat"]
#     alignn_layers: int = 4
#     gcn_layers: int = 4
#     atom_input_features: int = 1
#     # atom_input_features: int = 92
#     edge_input_features: int = 80
#     triplet_input_features: int = 40
#     embedding_features: int = 64
#     hidden_features: int = 64
#     # hidden_features: int = 256
#     # fc_layers: int = 1
#     # fc_features: int = 64
#     output_features: int = 1
#     graphwise_weight: float = 1.0  # 误差放大因子
#     # if link == log, apply `exp` to final outputs
#     # to constrain predictions to be positive
#     link: Literal["identity", "log", "logit"] = "identity"
#     zero_inflated: bool = False
#     classification: bool = False
#     force_mult_natoms: bool = False
#     energy_mult_natoms: bool = True  # 使用energy_mult_natoms时对总能量修正，未用到
#     # include_pos_deriv: bool = False  # 是否计算位置梯度
#     use_cutoff_function: bool = False
#     inner_cutoff: float = 3  # Ansgtrom
#     stress_multiplier: float = 1
#     # add_reverse_forces: bool = True  # will make True as default soon
#     lg_on_fly: bool = True  # will make True as default soon
#     batch_stress: bool = True
#     multiply_cutoff: bool = False
#     exponent: int = 5  # 5
#     use_penalty: bool = True  # 使用energy_mult_natoms时对总能量修正
#     penalty_factor: float = 0.1
#     penalty_threshold: float = 1
#
#     class Config:
#         """Configure model settings behavior."""
#
#         env_prefix = "jv_model"


def cutoff_function_based_edges_old(r, inner_cutoff=4):
    """Apply smooth cutoff to pairwise interactions

    r: bond lengths
    inner_cutoff: cutoff radius

    inside cutoff radius, apply smooth cutoff envelope
    outside cutoff radius: hard zeros
    """
    ratio = r / inner_cutoff
    return torch.where(
        ratio <= 1,
        1 - 6 * ratio**5 + 15 * ratio**4 - 10 * ratio**3,
        torch.zeros_like(r),
    )


def cutoff_function_based_edges(r, inner_cutoff=4, exponent=3):
    """Apply smooth cutoff to pairwise interactions

    r: bond lengths
    inner_cutoff: cutoff radius

    inside cutoff radius, apply smooth cutoff envelope
    outside cutoff radius: hard zeros
    """
    ratio = r / inner_cutoff
    c1 = -(exponent + 1) * (exponent + 2) / 2
    c2 = exponent * (exponent + 2)
    c3 = -exponent * (exponent + 1) / 2
    envelope = (
        1
        + c1 * ratio**exponent
        + c2 * ratio ** (exponent + 1)
        + c3 * ratio ** (exponent + 2)
    )
    # r_cut = inner_cutoff
    # r_on = inner_cutoff+1

    # r_sq = r * r
    # r_on_sq = r_on * r_on
    # r_cut_sq = r_cut * r_cut
    # envelope = (r_cut_sq - r_sq)
    # ** 2 * (r_cut_sq + 2 * r_sq - 3 * r_on_sq)/ (r_cut_sq - r_on_sq) ** 3
    return torch.where(r <= inner_cutoff, envelope, torch.zeros_like(r))


# class EdgeGatedGraphConv(nn.Module):
#     """Edge gated graph convolution from arxiv:1711.07553.
#
#     see also arxiv:2003.0098.
#
#     This is similar to CGCNN, but edge features only go into
#     the soft attention / edge gating function, and the primary
#     node update function is W cat(u, v) + b
#     """
#
#     def __init__(
#         self, input_features: int, output_features: int, heads: int = 1, residual: bool = True,
#     ):
#         """Initialize parameters for ALIGNN update."""
#         super().__init__()
#         # CGCNN-Conv operates on augmented edge features
#         # z_ij = cat(v_i, v_j, u_ij)
#         # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
#         # coalesce parameters for W_f and W_s
#         # but -- split them up along feature dimension
#         super().__init__()
#         self.heads = heads
#         self.output_features = output_features
#         self.residual = residual
#         # 门控单元算子
#         self.src_gate = nn.Linear(input_features, heads * output_features)
#         self.dst_gate = nn.Linear(input_features, heads * output_features)
#         self.edge_gate = nn.Linear(input_features, heads * output_features)
#         self.x_gate = nn.Linear(heads * output_features, output_features)
#         self.m_gate = nn.Linear(heads * output_features, output_features)
#         # 节点注意力机制
#         self.att_src = torch.nn.Parameter(torch.Tensor(1, heads, output_features))
#         self.att_dst = torch.nn.Parameter(torch.Tensor(1, heads, output_features))
#         # 节点更新算子
#         self.src_update = nn.Linear(input_features, heads * output_features)
#         self.dst_update = nn.Linear(input_features, heads * output_features)
#         self.bn_nodes = nn.LayerNorm(output_features)
#         self.bn_edges = nn.LayerNorm(output_features)
#
#         self.reset_parameters()  # 初始化参数
#     # 重置参数
#     def reset_parameters(self):
#         """
#         可修改初始化参数
#         """
#         self.src_gate.reset_parameters()
#         self.dst_gate.reset_parameters()
#         self.edge_gate.reset_parameters()
#         self.bn_edges.reset_parameters()
#
#         nn.init.xavier_uniform_(self.att_src)
#         nn.init.xavier_uniform_(self.att_dst)
#         # if self.bias is not None:
#         #     nn.init.xavier_uniform_(self.bias)
#
#         self.src_update.reset_parameters()
#         self.dst_update.reset_parameters()
#         self.bn_nodes.reset_parameters()
#
#     def forward(
#         self,
#         g: dgl.DGLGraph,
#         node_feats: torch.Tensor,
#         edge_feats: torch.Tensor,
#     ) -> torch.Tensor:
#         """Edge-gated graph convolution.
#
#         h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
#         """
#         # instead of concatenating (u || v || e) and applying one weight matrix
#         # split the weight matrix into three, apply, then sum
#         # see https://docs.dgl.ai/guide/message-efficient.html
#         # but split them on feature dimensions to update u, v, e separately
#         # m = BatchNorm(Linear(cat(u, v, e)))
#
#         # compute edge updates, equivalent to:
#         # Softplus(Linear(u || v || e))
#
#         assert node_feats.dim() == 2, "Static graphs not supported"
#         assert edge_feats.dim() == 2, "Static graphs not supported"
#         # print("node_feats", node_feats.shape)
#         # print("edge_feats", edge_feats.shape)
#         H, C = self.heads, self.output_features
#         g = g.local_var()
#
#         # 节点级注意力机制
#         g.ndata["e_src"] = self.src_gate(node_feats).view(-1, H, C)
#         g.ndata["att_src"] = (g.ndata["e_src"] * self.att_src).sum(-1).sum(-1)
#         g.ndata["e_dst"] = self.dst_gate(node_feats).view(-1, H, C)
#         g.ndata["att_dst"] = (g.ndata["e_dst"] * self.att_dst).sum(-1).sum(-1)
#         # 节点注意力移到边上并归一化
#         g.apply_edges(fn.copy_u("att_src", "att_src"))
#         g.edata["att_src"] = dgl.nn.functional.edge_softmax(g, g.edata.pop("att_src"))
#         g.apply_edges(fn.copy_u("att_dst", "att_dst"))
#         g.edata["att_dst"] = dgl.nn.functional.edge_softmax(g, g.edata.pop("att_dst"))
#         # 消息传递
#         g.apply_edges(fn.u_mul_e("e_src", "att_src", "e_nodes_src"))
#         g.apply_edges(fn.u_mul_e("e_dst", "att_dst", "e_nodes_dst"))
#         # print(g.edata["e_nodes_src"].shape)
#         # raise "error"
#         m = g.edata.pop("e_nodes_src") + g.edata.pop("e_nodes_dst") + self.edge_gate(edge_feats).view(-1, H, C)
#         m = m.view(-1, H * C)
#         # F.dropout(m, dropout)
#         # 节点更新
#         g.edata["sigma"] = torch.sigmoid(m)  # 1/(1+e^(-x_i))
#         g.ndata["Bh"] = self.src_update(node_feats)
#         g.update_all(
#             fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
#         )
#         g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
#         g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
#
#         x = self.dst_update(node_feats) + g.ndata.pop("h")
#
#         # softmax version seems to perform slightly worse
#         # that the sigmoid-gated version
#         # compute node updates
#         # Linear(u) + edge_gates ⊙ Linear(v)
#         # g.edata["gate"] = edge_softmax(g, y)
#         # g.ndata["h_dst"] = self.dst_update(node_feats)
#         # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
#         # x = self.src_update(node_feats) + g.ndata.pop("h")
#
#         # node and edge updates
#         x = F.silu(self.bn_nodes(self.x_gate(x)))
#         y = F.silu(self.bn_edges(self.m_gate(m)))
#
#         if self.residual:
#             x = node_feats + x
#             y = edge_feats + y
#         return x, y

class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, heads: int = 1, residual: bool = True,
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        super().__init__()
        self.heads = heads
        self.head_dim = output_features // heads
        self.residual = residual
        # 门控单元算子
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.x_gate = nn.Linear(output_features, output_features)
        self.m_gate = nn.Linear(output_features, output_features)
        # 节点注意力机制
        self.att_src = torch.nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.att_dst = torch.nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        # 节点更新算子
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        self.reset_parameters()  # 初始化参数
    # 重置参数
    def reset_parameters(self):
        """
        可修改初始化参数
        """
        self.src_gate.reset_parameters()
        self.dst_gate.reset_parameters()
        self.edge_gate.reset_parameters()
        self.bn_edges.reset_parameters()

        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        # if self.bias is not None:
        #     nn.init.xavier_uniform_(self.bias)

        self.src_update.reset_parameters()
        self.dst_update.reset_parameters()
        self.bn_nodes.reset_parameters()

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))

        assert node_feats.dim() == 2, "Static graphs not supported"
        assert edge_feats.dim() == 2, "Static graphs not supported"
        # print("node_feats", node_feats.shape)
        # print("edge_feats", edge_feats.shape)
        H, C = self.heads, self.head_dim
        g = g.local_var()

        # 节点级注意力机制
        g.ndata["e_src"] = self.src_gate(node_feats).view(-1, H, C)
        g.ndata["att_src"] = (g.ndata["e_src"] * self.att_src).sum(-1).sum(-1)
        g.ndata["e_dst"] = self.dst_gate(node_feats).view(-1, H, C)
        g.ndata["att_dst"] = (g.ndata["e_dst"] * self.att_dst).sum(-1).sum(-1)
        # 节点注意力移到边上并归一化
        g.apply_edges(fn.copy_u("att_src", "att_src"))
        g.edata["att_src"] = dgl.nn.functional.edge_softmax(g, g.edata.pop("att_src"))
        g.apply_edges(fn.copy_u("att_dst", "att_dst"))
        g.edata["att_dst"] = dgl.nn.functional.edge_softmax(g, g.edata.pop("att_dst"))
        # 消息传递
        g.apply_edges(fn.u_mul_e("e_src", "att_src", "e_nodes_src"))
        g.apply_edges(fn.u_mul_e("e_dst", "att_dst", "e_nodes_dst"))
        # print(g.edata["e_nodes_src"].shape)
        # raise "error"
        m = g.edata.pop("e_nodes_src") + g.edata.pop("e_nodes_dst") + self.edge_gate(edge_feats).view(-1, H, C)
        m = m.view(-1, H * C)
        # F.dropout(m, dropout)
        # 节点更新
        g.edata["sigma"] = torch.sigmoid(m)  # 1/(1+e^(-x_i))
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)

        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(self.x_gate(x)))
        y = F.silu(self.bn_edges(self.m_gate(m)))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y
        # 输出x、y的shape并抛出错误
        # print(x.shape)
        # print(y.shape)
        # raise "error"
        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features, heads)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features, heads)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class N_ALIGNNAtomWise(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        config: ALIGNNAtomWiseConfig = ALIGNNAtomWiseConfig(
            name="alignn_atomwise"
        ),
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification
        self.config = config
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                    config.heads,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features, config.heads,
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 1)
            self.softmax = nn.Sigmoid()
            # self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            if len(g) == 3:
                g, lg, lat = g
                lg = lg.local_var()
                # z = self.angle_embedding(lg.edata.pop("h"))
                # z = self.angle_embedding(lg.edata["h"])
            else:
                g, lat = g
                g.ndata["cart_coords"] = compute_cartesian_coordinates(g, lat)
                g.ndata["cart_coords"].requires_grad_(True)
                r, bondlength = compute_pair_vector_and_distance(g)
                lg = g.line_graph(shared=True)
                lg.ndata["r"] = r
                lg.apply_edges(compute_bond_cosines)
                # print('lg',lg)
                # angle features (fixed)
        else:
            g, lat = g
        # g = g.local_var()
        result = {}
        # initial node features: atom feature network...
        x = g.ndata["atom_features"]
        # x = g.ndata.pop("atom_features")
        # print('x1',x,x.shape)

        x = self.atom_embedding(x)
        # print('x2',x,x.shape)
        r = g.edata["r"]
        # if self.config.include_pos_deriv:
        #     # Not tested yet
        #     g.ndata["cart_coords"] = compute_cartesian_coordinates(g, lat)
        #     g.ndata["cart_coords"].requires_grad_(True)
        #     r, bondlength = compute_pair_vector_and_distance(g)
        #     lg = g.line_graph(shared=True)
        #     lg.ndata["r"] = r
        #     lg.apply_edges(compute_bond_cosines)

            # bondlength = torch.norm(r, dim=1)
            # y = self.edge_embedding(bondlength)
        bondlength = torch.norm(r, dim=1)
        # mask = bondlength >= self.config.inner_cutoff
        # bondlength[mask]=float(1.1)
        if self.config.lg_on_fly and len(self.alignn_layers) > 0:
            # re-compute bond angle cosines here to ensure
            # the three-body interactions are fully included
            # in the autograd graph. don't rely on dataloader/caching.

            lg.ndata["r"] = r  # overwrites precomputed r values
            lg.apply_edges(compute_bond_cosines)  # overwrites precomputed h
            z = self.angle_embedding(lg.edata["h"])
            # z = self.angle_embedding(lg.edata.pop("h"))

        # r = g.edata["r"].clone().detach().requires_grad_(True)
        if self.config.use_cutoff_function:
            # bondlength = cutoff_function_based_edges(
            if self.config.multiply_cutoff:
                c_off = cutoff_function_based_edges(
                    bondlength,
                    inner_cutoff=self.config.inner_cutoff,
                    exponent=self.config.exponent,
                ).unsqueeze(dim=1)

                y = self.edge_embedding(bondlength) * c_off
            else:
                bondlength = cutoff_function_based_edges(
                    bondlength,
                    inner_cutoff=self.config.inner_cutoff,
                    exponent=self.config.exponent,
                )
                y = self.edge_embedding(bondlength)
        else:
            y = self.edge_embedding(bondlength)
        # y = self.edge_embedding(bondlength)
        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)
        # norm-activation-pool-classify
        out = torch.empty(1)
        if self.config.output_features is not None:
            h = self.readout(g, x)
            out = self.fc(h)
            out = torch.squeeze(out)

        natoms = torch.tensor([gg.num_nodes() for gg in dgl.unbatch(g)]).to(
            g.device
        )
        en_out = out
        if self.config.energy_mult_natoms:
            en_out = out * natoms  # g.num_nodes()
        # 对总能量修正
        if self.config.use_penalty:
            penalty_factor = (
                self.config.penalty_factor
            )  # Penalty weight, tune as needed
            penalty_threshold = self.config.penalty_threshold  # 1 angstrom

            penalties = torch.where(
                bondlength < penalty_threshold,
                penalty_factor * (penalty_threshold - bondlength),
                torch.zeros_like(bondlength),
            )
            total_penalty = torch.sum(penalties)
            en_out += total_penalty

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.max(out,dim=1)
            out = self.softmax(out)
        result["out"] = out
        # print(result)
        return result


"""
if __name__ == "__main__":
    from jarvis.core.atoms import Atoms
    from alignn.graphs import Graph

    FIXTURES = {
        "lattice_mat": [
            [2.715, 2.715, 0],
            [0, 2.715, 2.715],
            [2.715, 0, 2.715],
        ],
        "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
    }
    Si = Atoms(
        lattice_mat=FIXTURES["lattice_mat"],
        coords=FIXTURES["coords"],
        elements=FIXTURES["elements"],
    )
    g, lg = Graph.atom_dgl_multi_graph(
        atoms=Si, neighbor_strategy="radius_graph", cutoff=5
    )
    lat = torch.tensor(atoms.lattice_mat)
    model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(name="alignn_atomwise"))
    out = model([g, lg, lat])
    print(out)
"""
