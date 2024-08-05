from operator import itemgetter

import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from polypharmacy.models.decoder_module_pl import BilinearDecoder, DEDICOM
from polypharmacy.models.conv_with_basis import GeneralConvWithBasis
from polypharmacy.utils import set_seed


set_seed(42)
torch_geometric.seed_everything(42)


class HeteroVGAE_PL(pl.LightningModule):
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
        kl_lambda=None,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

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

        if self.num_bases is not None:
            self.setup_basis_matrices()

        self.encoder = nn.ModuleList()
        conv_dicts = self.generate_hetero_conv_dict(
            self.hidden_dims, len(self.hidden_dims)
        )
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
                    nn.Linear(self.out_dim // 2, self.out_dim),
                )
                self.logstd_latent_encoder[node_type] = nn.Sequential(
                    nn.Linear(self.hidden_dims[-1], self.out_dim // 2),
                    nn.Tanh(),
                    nn.Linear(self.out_dim // 2, self.out_dim),
                )

        self.decoder = nn.ModuleDict()
        for decoder_type in self.decoder_2_relation.keys():
            if decoder_type == "bilinear":
                self.decoder[decoder_type] = BilinearDecoder(
                    out_dim, self.decoder_2_relation["bilinear"]
                )
            elif decoder_type == "dedicom":
                self.decoder[decoder_type] = DEDICOM(
                    out_dim, self.decoder_2_relation["dedicom"]
                )
            else:
                raise NotImplementedError

        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        z, mu, logstd = self.encode(x_dict, edge_index_dict)
        return self.decode_all_relation(z, edge_index_dict), mu, logstd

    def training_step(self, batch, batch_idx):
        z, _, _ = self.encode(batch.x_dict, batch.edge_index_dict)
        kl_loss = self.kl_loss_all(reduce="ratio", lambda_=self.hparams.kl_lambda)
        recon_loss = self.recon_loss_all_relation(z, batch.edge_label_index_dict)
        loss = recon_loss + kl_loss
        self.log("train_loss", loss, batch_size=1, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z, _, _ = self.encode(batch.x_dict, batch.edge_index_dict)
        pos_edge_label_index_dict = batch.edge_label_index_dict
        edge_label_index_dict = {}
        edge_label_dict = {}

        for edge_type in self.edge_types:
            src, relation, dst = edge_type
            if relation == "get_target" or edge_type not in pos_edge_label_index_dict:
                continue

            num_nodes = (batch.x_dict[src].shape[0], batch.x_dict[dst].shape[0])
            neg_edge_label_index = pyg_utils.negative_sampling(
                pos_edge_label_index_dict[edge_type], num_nodes=num_nodes
            )
            edge_label_index_dict[edge_type] = torch.cat(
                [pos_edge_label_index_dict[edge_type], neg_edge_label_index], dim=-1
            )

            pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
            neg_label = torch.zeros(neg_edge_label_index.shape[1])
            edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim=0)

        edge_pred = self.decode_all_relation(z, edge_label_index_dict, sigmoid=True)

        # Move predictions to CPU for metric calculation
        edge_pred = {k: v.cpu() for k, v in edge_pred.items()}
        edge_label_dict = {k: v.cpu() for k, v in edge_label_dict.items()}

        roc_auc, _, _ = self.cal_roc_auc_score_per_side_effect(
            edge_pred, edge_label_dict, self.edge_types
        )
        self.log('val_roc_auc', roc_auc, batch_size=1, on_step=False, on_epoch=True)

        recon_loss = self.recon_loss_all_relation(z, batch.edge_label_index_dict)
        kl_loss = self.kl_loss_all(reduce="ratio", lambda_=self.hparams.kl_lambda)
        loss = recon_loss + kl_loss
        self.log('val_loss', loss, batch_size=1, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        z, _, _ = self.encode(batch.x_dict, batch.edge_index_dict)
        pos_edge_label_index_dict = batch.edge_label_index_dict
        edge_label_index_dict = {}
        edge_label_dict = {}
        for edge_type in batch.edge_label_index_dict.keys():
            src, relation, dst = edge_type
            if relation == "get_target":
                continue
            num_nodes = (batch.x_dict[src].shape[0], batch.x_dict[dst].shape[0])
            neg_edge_label_index = pyg_utils.negative_sampling(
                pos_edge_label_index_dict[edge_type], num_nodes=num_nodes
            )
            edge_label_index_dict[edge_type] = torch.cat(
                [pos_edge_label_index_dict[edge_type], neg_edge_label_index], dim=-1
            )

            pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
            neg_label = torch.zeros(neg_edge_label_index.shape[1])
            edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim=0)

        edge_pred = self.decode_all_relation(z, edge_label_index_dict, sigmoid=True)
        for relation in edge_pred.keys():
            edge_pred[relation] = edge_pred[relation].cpu()
        roc_auc, _, _ = self.cal_roc_auc_score_per_side_effect(
            edge_pred, edge_label_dict, self.hparams.edge_types
        )
        prec, _, _ = self.cal_average_precision_score_per_side_effect(
            edge_pred, edge_label_dict, self.hparams.edge_types
        )
        apk, _ = self.cal_apk(edge_pred, edge_label_dict, self.hparams.edge_types, k=50)
        self.log_dict(
            {"test_roc_auc": roc_auc, "test_auprc": prec, "test_ap@50": apk},
            batch_size=1,
            on_step=False,
            on_epoch=True,
        )

    def cal_roc_auc_score_per_side_effect(self, preds, labels, edge_types):
        total_roc_auc = {}
        counts_dict = {}
        for _, relation, _ in edge_types:
            if (
                relation not in ["has_target", "get_target", "interact"]
                and relation in preds
                and relation in labels
            ):
                score = roc_auc_score(labels[relation], preds[relation])
                total_roc_auc[relation] = score
                counts_dict[relation] = len(labels[relation])

        if not total_roc_auc:
            return 0.0, {}, {}  # Return default values if no valid relations found

        return (
            sum(total_roc_auc.values()) / len(total_roc_auc),
            total_roc_auc,
            counts_dict,
        )

    def cal_roc_auc_score(self, pred, label, edge_types):
        all_label = self.concat_all(label)
        all_pred = self.concat_all(pred)
        return roc_auc_score(all_label, all_pred)

    def concat_all(self, item_dict):
        return torch.cat([item_dict[relation] for relation in item_dict.keys()], dim=-1)

    def cal_average_precision_score_per_side_effect(self, preds, labels, edge_types):
        total_prec = {}
        counts_dict = {}
        for src, relation, dst in edge_types:
            if relation not in ["has_target", "get_target", "interact"]:
                score = average_precision_score(labels[relation], preds[relation])
                total_prec[relation] = score
                counts_dict[relation] = len(labels[relation])
        return sum(total_prec.values()) / len(total_prec), total_prec, counts_dict

    def cal_apk(self, preds, labels, edge_types, k=50):
        total_apk = {}

        for src, relation, dst in edge_types:
            if relation in ["has_target", "get_target", "interact"]:
                continue

            actual = []
            predicted = []
            edge_ind = 0

            for idx, score in enumerate(preds[relation]):
                if labels[relation][idx] == 1:
                    actual.append(idx)
                predicted.append((score, idx))
            predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[
                1
            ]
            total_apk[relation] = self.apk(actual, predicted, k=k)

        return sum(total_apk.values()) / len(total_apk), total_apk

    def apk(self, actual, predicted, k=10):
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def setup_basis_matrices(self):
        self.basis_lin_msg_wt = nn.ParameterList()
        self.basis_lin_msg_biases = nn.ParameterList()
        self.basis_lin_self_wt = nn.ParameterList()
        self.basis_lin_self_biases = nn.ParameterList()
        self.linear_combinations = nn.ModuleList()

        # First layer
        self.basis_lin_msg_wt.append(
            nn.Parameter(
                torch.FloatTensor(
                    self.input_dim["drug"], self.hidden_dims[0], self.num_bases
                )
            )
        )
        self.basis_lin_msg_biases.append(
            nn.Parameter(torch.FloatTensor(self.hidden_dims[0], self.num_bases))
        )
        self.basis_lin_self_wt.append(
            nn.Parameter(
                torch.FloatTensor(
                    self.input_dim["drug"], self.hidden_dims[0], self.num_bases
                )
            )
        )
        self.basis_lin_self_biases.append(
            nn.Parameter(torch.FloatTensor(self.hidden_dims[0], self.num_bases))
        )

        self.linear_combinations.append(
            nn.ParameterDict(
                {
                    str(edge_type): nn.Parameter(torch.FloatTensor(1, self.num_bases))
                    for edge_type in self.edge_types
                    if edge_type[0] == "drug" and edge_type[2] == "drug"
                }
            )
        )

        nn.init.xavier_uniform_(self.basis_lin_msg_wt[0])
        nn.init.xavier_uniform_(self.basis_lin_msg_biases[0])
        nn.init.xavier_uniform_(self.basis_lin_self_wt[0])
        nn.init.xavier_uniform_(self.basis_lin_self_biases[0])
        for params in self.linear_combinations[0].values():
            nn.init.xavier_uniform_(params)

        # Hidden layers
        for i in range(1, len(self.hidden_dims)):
            self.basis_lin_msg_wt.append(
                nn.Parameter(
                    torch.FloatTensor(
                        self.hidden_dims[i - 1], self.hidden_dims[i], self.num_bases
                    )
                )
            )
            self.basis_lin_msg_biases.append(
                nn.Parameter(torch.FloatTensor(self.hidden_dims[i], self.num_bases))
            )
            self.basis_lin_self_wt.append(
                nn.Parameter(
                    torch.FloatTensor(
                        self.hidden_dims[i - 1], self.hidden_dims[i], self.num_bases
                    )
                )
            )
            self.basis_lin_self_biases.append(
                nn.Parameter(torch.FloatTensor(self.hidden_dims[i], self.num_bases))
            )

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
            in_channels = (
                self.input_dim
                if i == 0
                else {node_type: dims[i - 1] for node_type in self.node_types}
            )
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

    def decode_all_relation(self, z_dict, edge_index_dict, sigmoid=False):
        output = {}
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict.keys():
                continue
            src, relation, dst = edge_type
            decoder_type = self.relation_2_decoder[relation]
            z = (z_dict[src], z_dict[dst])
            output[relation] = self.decoder[decoder_type](
                z, edge_index_dict[edge_type], relation
            )
            if sigmoid:
                output[relation] = torch.sigmoid(output[relation])
        return output

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
            self.__mu__ = {
                node_type: latent[node_type][..., : self.out_dim]
                for node_type in self.node_types
            }
            self.__logstd__ = {
                node_type: latent[node_type][..., self.out_dim :]
                for node_type in self.node_types
            }
        elif self.latent_encoder_type == "linear":
            self.__mu__ = {
                node_type: self.mu_latent_encoder[node_type](z_dict[node_type])
                for node_type in self.node_types
            }
            self.__logstd__ = {
                node_type: self.logstd_latent_encoder[node_type](z_dict[node_type])
                for node_type in self.node_types
            }

        for node_type in self.node_types:
            self.__logstd__[node_type] = self.__logstd__[node_type].clamp(max=10)

        return (
            self.reparameterize(self.__mu__, self.__logstd__),
            self.__mu__,
            self.__logstd__,
        )

    def reparameterize(self, mu, logstd):
        if self.training:
            return {
                node_type: mu[node_type]
                + torch.randn_like(logstd[node_type]) * torch.exp(logstd[node_type])
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

    def kl_loss(self, node_type, mu=None, logstd=None):
        mu = self.__mu__[node_type] if mu is None else mu
        logstd = self.__logstd__[node_type] if logstd is None else logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp() ** 2, dim=1)
        )

    def kl_loss_all(self, reduce="sum", lambda_=None):
        assert len(self.__mu__) > 0 and len(self.__logstd__) > 0
        loss = {}
        for node_type in self.node_types:
            loss[node_type] = self.kl_loss(
                node_type, self.__mu__[node_type], self.__logstd__[node_type]
            )
        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "ratio":
            assert lambda_ is not None
            loss = sum(
                [loss[node_type] * lambda_[node_type] for node_type in self.node_types]
            )
        return loss

    def recon_loss(self, z_dict, edge_type, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decode(z_dict, pos_edge_index, edge_type, sigmoid=True) + 1e-15
        ).mean()

        if neg_edge_index is None:
            src, relation, dst = edge_type
            num_src_node, num_dst_node = z_dict[src].shape[0], z_dict[dst].shape[0]
            neg_edge_index = pyg_utils.negative_sampling(
                pos_edge_index, num_nodes=(num_src_node, num_dst_node)
            )

        neg_loss = -torch.log(
            1 - self.decode(z_dict, neg_edge_index, edge_type, sigmoid=True) + 1e-15
        ).mean()

        return pos_loss + neg_loss

    def recon_loss_all_relation(self, z_dict, edge_index_dict, reduce="sum"):
        loss = {}
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict.keys():
                continue
            loss[edge_type] = self.recon_loss(
                z_dict, edge_type, edge_index_dict[edge_type]
            )

        if reduce == "sum":
            loss = sum(loss.values())
        elif reduce == "mean":
            loss = sum(loss.values()) / len(self.edge_types)
        return loss
