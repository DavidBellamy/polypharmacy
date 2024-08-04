import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from polypharmacy.models.decoder_module import BilinearDecoder, DEDICOM
from polypharmacy.models.conv_with_basis import GeneralConvWithBasis
from polypharmacy.utils import set_seed


set_seed(42)
torch_geometric.seed_everything(42)

class HeteroVGAE(nn.Module):
    def __init__(
        self,
        hidden_dims,
        out_dim,
        node_types,
        edge_types,
        decoder_2_relation,
        relation_2_decoder,
        num_bases=None,
        input_dim=None,
        dropout=0.5,
        latent_encoder_type="gconv",
        device="cpu",
    ):
        super().__init__()

        self.__mu__ = {}
        self.__logstd__ = {}

        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.edge_types = edge_types
        self.node_types = node_types
        self.decoder_2_relation = decoder_2_relation
        self.relation_2_decoder = relation_2_decoder
        self.num_bases = num_bases
        self.input_dim = input_dim
        self.latent_encoder_type = latent_encoder_type
        self.device = device

        if self.num_bases is not None:
            self.setup_basis_matrices()

        self.encoder = nn.ModuleList()
        conv_dicts = self.generate_hetero_conv_dict(self.hidden_dims, len(self.hidden_dims))
        for i in range(len(conv_dicts)):
            conv = pyg_nn.HeteroConv(conv_dicts[i], aggr="sum")
            self.encoder.append(conv)

        if latent_encoder_type == "gconv":
            latent_encoder_dict = self.generate_hetero_conv_dict([self.out_dim * 2], 1)
            self.latent_encoder = pyg_nn.HeteroConv(latent_encoder_dict[0], aggr="sum")
        elif latent_encoder_type == "linear":
            self.mu_latent_encoder = nn.ModuleDict()
            self.logstd_latent_encoder = nn.ModuleDict()
            for node_type in self.node_types:
                self.mu_latent_encoder[node_type] = nn.Sequential(
                    nn.Linear(self.hidden_dims[-1], self.out_dim // 2),
                    nn.Tanh(),
                    nn.Linear(self.out_dim // 2, self.out_dim)
                ).to(device)
                self.logstd_latent_encoder[node_type] = nn.Sequential(
                    nn.Linear(self.hidden_dims[-1], self.out_dim // 2),
                    nn.Tanh(),
                    nn.Linear(self.out_dim // 2, self.out_dim)
                ).to(device)

        self.decoder = nn.ModuleDict()
        for decoder_type in self.decoder_2_relation.keys():
            if decoder_type == "bilinear":
                self.decoder[decoder_type] = BilinearDecoder(
                    out_dim, self.decoder_2_relation["bilinear"], device
                )
            elif decoder_type == "dedicom":
                self.decoder[decoder_type] = DEDICOM(
                    out_dim, self.decoder_2_relation["dedicom"], device
                )
            else:
                raise NotImplementedError

        self.dropout = dropout

    def setup_basis_matrices(self):
        self.basis_lin_msg_wt = nn.ParameterList()
        self.basis_lin_msg_biases = nn.ParameterList()
        self.basis_lin_self_wt = nn.ParameterList()
        self.basis_lin_self_biases = nn.ParameterList()
        self.linear_combinations = nn.ModuleList()

        # First layer
        self.basis_lin_msg_wt.append(nn.Parameter(torch.FloatTensor(self.input_dim['drug'], self.hidden_dims[0], self.num_bases)))
        self.basis_lin_msg_biases.append(nn.Parameter(torch.FloatTensor(self.hidden_dims[0], self.num_bases)))
        self.basis_lin_self_wt.append(nn.Parameter(torch.FloatTensor(self.input_dim['drug'], self.hidden_dims[0], self.num_bases)))
        self.basis_lin_self_biases.append(nn.Parameter(torch.FloatTensor(self.hidden_dims[0], self.num_bases)))
        
        self.linear_combinations.append(nn.ParameterDict({
            str(edge_type): nn.Parameter(torch.FloatTensor(1, self.num_bases))
            for edge_type in self.edge_types if edge_type[0] == 'drug' and edge_type[2] == 'drug'
        }))

        nn.init.xavier_uniform_(self.basis_lin_msg_wt[0])
        nn.init.xavier_uniform_(self.basis_lin_msg_biases[0])
        nn.init.xavier_uniform_(self.basis_lin_self_wt[0])
        nn.init.xavier_uniform_(self.basis_lin_self_biases[0])
        for params in self.linear_combinations[0].values():
            nn.init.xavier_uniform_(params)

        # Hidden layers
        for i in range(1, len(self.hidden_dims)):
            self.basis_lin_msg_wt.append(nn.Parameter(torch.FloatTensor(self.hidden_dims[i-1], self.hidden_dims[i], self.num_bases)))
            self.basis_lin_msg_biases.append(nn.Parameter(torch.FloatTensor(self.hidden_dims[i], self.num_bases)))
            self.basis_lin_self_wt.append(nn.Parameter(torch.FloatTensor(self.hidden_dims[i-1], self.hidden_dims[i], self.num_bases)))
            self.basis_lin_self_biases.append(nn.Parameter(torch.FloatTensor(self.hidden_dims[i], self.num_bases)))
            
            self.linear_combinations.append(nn.ParameterDict({
                str(edge_type): nn.Parameter(torch.FloatTensor(1, self.num_bases))
                for edge_type in self.edge_types if edge_type[0] == 'drug' and edge_type[2] == 'drug'
            }))

            nn.init.xavier_uniform_(self.basis_lin_msg_wt[-1])
            nn.init.xavier_uniform_(self.basis_lin_msg_biases[-1])
            nn.init.xavier_uniform_(self.basis_lin_self_wt[-1])
            nn.init.xavier_uniform_(self.basis_lin_self_biases[-1])
            for params in self.linear_combinations[-1].values():
                nn.init.xavier_uniform_(params)

    def generate_hetero_conv_dict(self, dims, num_layers):
        conv_dicts = []

        for i in range(num_layers):
            D = {}
            in_channels = self.input_dim if i == 0 else {node_type: dims[i-1] for node_type in self.node_types}
            for edge_type in self.edge_types:
                src, relation, dst = edge_type
                
                if self.num_bases is not None and src == "drug" and dst == "drug":
                    D[edge_type] = GeneralConvWithBasis(
                        in_channels,
                        dims[i],
                        self.basis_lin_msg_wt[i],
                        self.basis_lin_msg_biases[i],
                        self.linear_combinations[i][str(edge_type)],
                        self.basis_lin_self_wt[i],
                        self.basis_lin_self_biases[i],
                        aggr="sum",
                        skip_linear=True,
                        l2_normalize=True,
                    )
                else:
                    D[edge_type] = pyg_nn.GeneralConv(
                        (in_channels[src], in_channels[dst]),
                        dims[i],
                        aggr="sum",
                        skip_linear=True,
                        l2_normalize=True,
                    )
            conv_dicts.append(D)
        return conv_dicts

    def encode(self, x_dict, edge_index_dict):
        z_dict = x_dict
        for idx, conv in enumerate(self.encoder):
            z_dict = conv(z_dict, edge_index_dict)
            if idx < len(self.encoder) - 1:
                z_dict = {key: F.relu(z) for key, z in z_dict.items()}
            z_dict = {
                key: F.dropout(z, p=self.dropout, training=self.training)
                for key, z in z_dict.items()
            }

        if self.latent_encoder_type == "gconv":
                latent = self.latent_encoder(z_dict, edge_index_dict)
                self.__mu__ = {node_type: latent[node_type][..., :self.out_dim] for node_type in self.node_types}
                self.__logstd__ = {node_type: latent[node_type][..., self.out_dim:] for node_type in self.node_types}
        elif self.latent_encoder_type == "linear":
                self.__mu__ = {node_type: self.mu_latent_encoder[node_type](z_dict[node_type]) for node_type in self.node_types}
                self.__logstd__ = {node_type: self.logstd_latent_encoder[node_type](z_dict[node_type]) for node_type in self.node_types}

        for node_type in self.node_types:
            self.__logstd__[node_type] = self.__logstd__[node_type].clamp(max=10)

        return self.reparameterize(self.__mu__, self.__logstd__), self.__mu__, self.__logstd__
    
    def reparameterize(self, mu, logstd):
        if self.training:
            return {
                node_type: mu[node_type] + torch.randn_like(logstd[node_type]) * torch.exp(logstd[node_type])
                for node_type in self.node_types
            }
        else:
            return mu

    def decode(self, z_dict, edge_index, edge_type, sigmoid=False):
        src, relation, dst = edge_type
        decoder_type = self.relation_2_decoder[relation]
        z = (z_dict[src], z_dict[dst])
        output = self.decoder[decoder_type](z, edge_index, relation)
        if sigmoid:
            output = torch.sigmoid(output)
        return output

    def decode_all_relation(self, z_dict, edge_index_dict, sigmoid=False):
        output = {}
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict.keys():
                continue
            src, relation, dst = edge_type
            decoder_type = self.relation_2_decoder[relation]
            z = (z_dict[src], z_dict[dst])
            output[relation] = self.decoder[decoder_type](z, edge_index_dict[edge_type], relation)
            if sigmoid:
                output[relation] = torch.sigmoid(output[relation])
        return output

    def forward(self, x_dict, edge_index_dict):
        z, mu, logstd = self.encode(x_dict, edge_index_dict)
        return self.decode_all_relation(z, edge_index_dict), mu, logstd

    def kl_loss(self, node_type, mu=None, logstd=None):
        mu = self.__mu__[node_type] if mu is None else mu
        logstd = self.__logstd__[node_type] if logstd is None else logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def kl_loss_all(self, reduce="sum", lambda_=None):
        assert len(self.__mu__) > 0 and len(self.__logstd__) > 0
        loss = {}
        for node_type in self.node_types:
            loss[node_type] = self.kl_loss(node_type, self.__mu__[node_type], self.__logstd__[node_type])
        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "ratio":
            assert lambda_ is not None
            loss = sum([loss[node_type] * lambda_[node_type] for node_type in self.node_types])
        return loss
    
    def recon_loss(self, z_dict, edge_type, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decode(z_dict, pos_edge_index, edge_type, sigmoid=True) + 1e-15
        ).mean()

        if neg_edge_index is None:
            src, relation, dst = edge_type
            num_src_node, num_dst_node = z_dict[src].shape[0], z_dict[dst].shape[0]
            neg_edge_index = pyg_utils.negative_sampling(pos_edge_index, num_nodes=(num_src_node, num_dst_node))

        neg_loss = -torch.log(
            1 - self.decode(z_dict, neg_edge_index, edge_type, sigmoid=True) + 1e-15
        ).mean()

        return pos_loss + neg_loss

    def recon_loss_all_relation(self, z_dict, edge_index_dict, reduce="sum"):
        loss = {}
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict.keys():
                continue
            loss[edge_type] = self.recon_loss(z_dict, edge_type, edge_index_dict[edge_type])

        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "mean":
            loss = sum(loss.values()) / len(self.edge_types)
        return loss

    def loss(self, x_dict, edge_index_dict):
        z, _, _ = self.encode(x_dict, edge_index_dict)
        recon_loss = self.recon_loss_all_relation(z, edge_index_dict)
        kl_loss = self.kl_loss_all(reduce="sum")  # or "ratio" if you want to use lambda_
        return recon_loss + kl_loss
    
    def test(self, z_dict, pos_edge_index, neg_edge_index, edge_type):
        pos_pred = self.decode(z_dict, pos_edge_index, edge_type, sigmoid=True)
        neg_pred = self.decode(z_dict, neg_edge_index, edge_type, sigmoid=True)

        pos_y = torch.ones(pos_pred.size(0), device=self.device)
        neg_y = torch.zeros(neg_pred.size(0), device=self.device)

        y = torch.cat([pos_y, neg_y])
        pred = torch.cat([pos_pred, neg_pred])

        return y, pred