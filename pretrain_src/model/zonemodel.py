import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_scatter import scatter
from torch_sparse import SparseTensor, fill_diag, index_select, matmul, remove_diag

from torch_sparse import t as transpose

from torch_geometric.nn import LEConv, GraphConv, GCNConv
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import add_remaining_self_loops, softmax


class AdaptiveZonePartition(torch.nn.Module):
    """
    
    """
    def __init__(self, in_channels, GNN = None, dropout = 0.0, ratio = 0.2,
                 negative_slope = 0.2, add_self_loops = False, opts = None, 
                 **kwargs):
        super().__init__()

        self.opts = opts
        self.in_channels = in_channels
        self.ratio = opts.zoner_ratio
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels, **kwargs)
        else:
            self.gnn_intra_cluster = None
        self.reset_parameters()

    def reset_parameters(self):
            self.lin.reset_parameters()
            self.att.reset_parameters()
            self.gnn_score.reset_parameters()
            if self.gnn_intra_cluster is not None:
                self.gnn_intra_cluster.reset_parameters()


    def forward(self, x, edge_index, edge_weight, k = None, point_size = None, batch = None):
        N = x.size(0)
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_pool = x
        if self.gnn_intra_cluster is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)
        
        x_pool_j = x_pool[edge_index[0]]
        #x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce='max')     
        #x_q = self.lin(x_q)[edge_index[1]]
        x_q = self.lin(x)[edge_index[1]]

        score = self.att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N)

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout, training=self.training) 

        v_j = x[edge_index[0]] * score.view(-1, 1)
        x = scatter(v_j, edge_index[1], dim=0, reduce='add')    

        # Cluster selection.
        fitness = self.gnn_score(x, edge_index).sigmoid().view(-1)  
        if self.opts.zoner_part_loss:
            perm = topk(fitness, (k-0.5) / N, batch)                   
        else:
            perm = topk(fitness, self.ratio, batch)
        x = x[perm] * fitness[perm].view(-1, 1)                    
        #batch = batch[perm]
        
        # Graph coarsening.
        row, col = edge_index[0], edge_index[1]

        S = SparseTensor(row=row, col=col, value=score, sparse_sizes=(N, N))

        S = index_select(S, 1, perm)
        S = S.to_dense()
        
        if self.opts.zoner_part_loss:
            if self.opts.zoner == "hard_zone":
                zone_partition_logit = S.clone()
            elif self.opts.zoner == "soft_zone":
                zone_partition_logit = S[1:,:].clone()
        else:
            zone_partition_logit = None

        
        gmap_vpzone = None
        if self.opts.zoner == "hard_zone":
            gmap_vpzone = S.argmax(dim=1)
            for i in range(len(perm)):
                gmap_vpzone[perm[i]] = i
            gmap_vpzone = torch.cat([gmap_vpzone.new_zeros(1), gmap_vpzone], dim=0) 
            gmap_vpzone = gmap_vpzone.tolist()

        elif self.opts.zoner == "soft_zone":
            S = torch.cat([S, torch.ones((point_size - S.size(0), S.size(1))).cuda() * -float('inf')], dim=0)
   
        zone_embed = x

        return gmap_vpzone, S, zone_embed, zone_partition_logit

def gmap_embeds_attention_calculate(gmap_ss, zone_weights, gmap_masks):
    '''
    gmap_ss: batch_size x point_size x zone_size
    zone_weights: batch_size x zone_size
    '''
    zone_weights = F.softmax(zone_weights, dim=1)
    zone_weights = zone_weights.unsqueeze(-1)
    gmap_ss = gmap_ss.masked_fill(gmap_ss==-float("inf"), 0)
    gmap_embeds_attention = torch.bmm(gmap_ss, zone_weights).squeeze(-1) 
    gmap_embeds_attention = gmap_embeds_attention.masked_fill(gmap_masks, -float("inf"))
    gmap_embeds_attention = F.softmax(gmap_embeds_attention, dim=1)
    return gmap_embeds_attention


