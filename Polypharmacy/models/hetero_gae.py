import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GeneralConv 

from polypharmacy.models.decoder_module import BilinearDecoder, DEDICOM
from polypharmacy.models.conv_with_basis import GeneralConvWithBasis


def reset_params(self):
    self.lin_msg.weight_initializer = "glorot"
    self.lin_msg.reset_parameters()

    if hasattr(self.lin_self, "reset_parameters"):
        self.lin_self.weight_initializer = "glorot"
        self.lin_self.reset_parameters()

    if not self.directed_msg:
        self.lin_msg_i.weight_initializer = "glorot"
        self.lin_msg_i.reset_parameters()

    if self.in_edge_channels is not None:
        self.lin_edge.weight_initializer = "glorot"
        self.lin_edge_weight.reset_parameters()


GeneralConv.reset_parameters = reset_params


class HeteroGAE(nn.Module):
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
        device="cpu",
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.edge_types = edge_types
        self.node_types = node_types
        self.decoder_2_relation = decoder_2_relation
        self.relation_2_decoder = relation_2_decoder
        self.num_bases = (
            num_bases  # number of basis functions for the basis decomposition
        )
        # (less than number of interaction types)
        if self.num_bases:
            self.input_dim = input_dim  # input dimension for each node type: a dictionary of node_type: input_dim
            self.basis_lin_msg_wt = [
                nn.Parameter(
                    torch.FloatTensor(input_dim["drug"], hidden_dims[0], self.num_bases)
                )
            ]
            # basis_lin_msg_wt is a basis of the linear transformation for
            # the message passing from drug node to drug node
            self.basis_lin_msg_biases = [
                nn.Parameter(torch.FloatTensor(hidden_dims[0], self.num_bases))
            ]
            # basis_lin_msg_biases is a basis of the linear transformation for the bias
            self.basis_lin_self_wt = [
                nn.Parameter(
                    torch.FloatTensor(input_dim["drug"], hidden_dims[0], self.num_bases)
                )
            ]
            # basis_lin_self_wt is a basis of the linear transformation for the self-connection
            self.basis_lin_self_biases = [
                nn.Parameter(torch.FloatTensor(hidden_dims[0], self.num_bases))
            ]
            # basis_lin_self_biases is a basis of the linear transformation for the bias for self-connection
            self.linear_combinations = [
                nn.ParameterDict(
                    {
                        str(edge_type): nn.Parameter(
                            torch.FloatTensor(1, self.num_bases)
                        )
                        for edge_type in self.edge_types
                        if edge_type[0] == "drug" and edge_type[2] == "drug"
                    }
                )
            ]
            # linear_combinations is a dictionary of edge_type: linear combination of basis functions
            nn.init.xavier_uniform_(self.basis_lin_msg_wt[0])
            nn.init.xavier_uniform_(self.basis_lin_msg_biases[0])
            nn.init.xavier_uniform_(self.basis_lin_self_wt[0])
            nn.init.xavier_uniform_(self.basis_lin_self_biases[0])
            for params in self.linear_combinations[0].values():
                nn.init.xavier_uniform_(params)
            for i, hidden_dim in enumerate(
                hidden_dims[1:]
            ):  # instantiate the basis_lin_msg_wt, basis_lin_msg_biases,
                # basis_lin_self_wt, basis_lin_self_biases for the hidden layers
                self.basis_lin_msg_wt.append(
                    nn.Parameter(
                        torch.FloatTensor(hidden_dims[i], hidden_dim, self.num_bases)
                    )
                )
                self.basis_lin_msg_biases.append(
                    nn.Parameter(torch.FloatTensor(hidden_dim, self.num_bases))
                )
                self.basis_lin_self_wt.append(
                    nn.Parameter(
                        torch.FloatTensor(hidden_dims[i], hidden_dim, self.num_bases)
                    )
                )
                self.basis_lin_self_biases.append(
                    nn.Parameter(torch.FloatTensor(hidden_dim, self.num_bases))
                )
                # initialize the basis_lin_msg_wt, basis_lin_msg_biases, basis_lin_self_wt,
                # basis_lin_self_biases for the hidden layers
                nn.init.xavier_uniform_(self.basis_lin_msg_wt[i + 1])
                nn.init.xavier_uniform_(self.basis_lin_msg_biases[i + 1])
                nn.init.xavier_uniform_(self.basis_lin_self_wt[i + 1])
                nn.init.xavier_uniform_(self.basis_lin_self_biases[i + 1])
                self.linear_combinations.append(
                    nn.ParameterDict(
                        {
                            str(edge_type): nn.Parameter(
                                torch.FloatTensor(1, self.num_bases)
                            )
                            for edge_type in self.edge_types
                            if edge_type[0] == "drug" and edge_type[2] == "drug"
                        }
                    )
                )
                for params in self.linear_combinations[i + 1].values():
                    nn.init.xavier_uniform_(params)

        self.encoder = nn.ModuleList()
        conv_dicts = self.generate_hetero_conv_dict()

        for i in range(len(conv_dicts)):
            conv = pyg_nn.HeteroConv(conv_dicts[i], aggr="sum")
            self.encoder.append(conv)

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
                raise NotImplemented
        self.dropout = dropout

    def forward(self):
        pass

    def generate_hetero_conv_dict(self):
        conv_dicts = []

        for i in range(len(self.hidden_dims)):
            D = {}

            for edge_type in self.edge_types:
                src, relation, dst = edge_type
                if relation not in ["interact", "has_target", "get_target"]:
                    if self.num_bases is not None and src == "drug" and dst == "drug":
                        in_channels = (
                            self.hidden_dims[i - 1] if i > 0 else self.input_dim["drug"]
                        )

                        D[edge_type] = GeneralConvWithBasis(
                            in_channels,
                            self.hidden_dims[i],
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
                        D[edge_type] = GeneralConv(
                            (-1, -1),
                            self.hidden_dims[i],
                            aggr="sum",
                            skip_linear=True,
                            l2_normalize=True,
                        )
                else:
                    D[edge_type] = GeneralConv(
                        (-1, -1),
                        self.hidden_dims[i],
                        aggr="sum",
                        skip_linear=True,
                        l2_normalize=True,
                    )
            conv_dicts.append(D)
        return conv_dicts

    def encode(self, x_dict, edge_index_dict):
        """
        The encode method takes an input dictionary x_dict (node features)
        and an edge_index_dict (edge indices) and applies the encoder layers
        to generate the latent node representations (z_dict).
        Dropout and ReLU activation are applied after each layer except the last one
        where only dropout is applied.
        """

        z_dict = x_dict
        for idx, conv in enumerate(self.encoder):
            z_dict = conv(z_dict, edge_index_dict)

            if idx < len(self.encoder) - 1:
                z_dict = {key: F.relu(z) for key, z in z_dict.items()}

            z_dict = {
                key: F.dropout(z, p=self.dropout, training=self.training)
                for key, z in z_dict.items()
            }
        return z_dict

    def decode_all_relation(self, z_dict, edge_index_dict, sigmoid=False):
        output = {}  # stores the edge predictions for each relation
        for edge_type in self.edge_types:  # iterate over all edge types
            if (
                edge_type not in edge_index_dict.keys()
            ):  # skip if edge type is not present in the graph
                continue
            src, relation, dst = edge_type  # unpack the edge type
            decoder_type = self.relation_2_decoder[
                relation
            ]  # get the decoder type for the relation
            z = (
                z_dict[src],
                z_dict[dst],
            )  # get the latent representations for the source and destination nodes
            output[relation] = self.decoder[decoder_type](
                z, edge_index_dict[edge_type], relation
            )  # decode
            # Decode the edge prediction using the decoder for the relation
            if sigmoid:
                output[relation] = F.sigmoid(output[relation])
        return output
