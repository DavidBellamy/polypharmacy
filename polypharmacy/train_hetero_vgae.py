import argparse
import time
import warnings

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.profiler
import torch_geometric
import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils

from polypharmacy.data import load_data
from polypharmacy.metrics import cal_roc_auc_score_per_side_effect, cal_average_precision_score_per_side_effect, cal_apk
from polypharmacy.models.hetero_vgae import HeteroVGAE
from polypharmacy.utils import set_seed, setup_logger


logger = setup_logger('./logs')

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Polypharmacy Side Effect Prediction")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--num_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--chkpt_dir", type=str, default="./checkpoint/", help="checkpoint directory")
parser.add_argument("--latent_encoder_type", type=str, default="linear", help="latent encoder type")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--device", type=str, default="cuda:0", help="training device")
parser.add_argument("--num_bases", type=int, default=None, help="number of basis matrices for weight sharing (0 for no weight sharing)")

args = parser.parse_args()
set_seed(args.seed)
torch_geometric.seed_everything(args.seed)

logger.info("Load data")
data = load_data()
edge_types = data.edge_types
rev_edge_types = []

for (src, relation, dst) in edge_types:
    rev_relation = f"rev_{relation}"
    rev_edge_types.append((dst, rev_relation, src))

transform = pyg_T.Compose([
    pyg_T.AddSelfLoops(),
    pyg_T.RandomLinkSplit(num_val = 0.1, num_test = 0.1, is_undirected = True,
        edge_types = edge_types, rev_edge_types = rev_edge_types, neg_sampling_ratio = 0.0
        , disjoint_train_ratio = 0.2)])

train_data, valid_data, test_data = transform(data)

for node in data.node_types:
    train_data[node].x = train_data[node].x.to_sparse().float()
    valid_data[node].x = valid_data[node].x.to_sparse().float()
    test_data[node].x = test_data[node].x.to_sparse().float()

for edge_type in rev_edge_types:
    del train_data[edge_type]
    del valid_data[edge_type]
    del test_data[edge_type]

for edge_type in edge_types:
    if edge_type[0] == edge_type[2]:
        train_data[edge_type].edge_index = pyg_utils.to_undirected(train_data[edge_type].edge_index)
        valid_data[edge_type].edge_index = pyg_utils.to_undirected(valid_data[edge_type].edge_index)
        test_data[edge_type].edge_index = pyg_utils.to_undirected(test_data[edge_type].edge_index)

logger.info("Initialize model...")
hidden_dim = [64, 32]
num_layer = 2
decoder_2_relation = {
    "bilinear": ["interact", "has_target", "get_target"],
    "dedicom":  [relation for (_, relation, _) in edge_types
                 if relation not in ["interact", "has_target", "get_target"]]
}

relation_2_decoder = {
    "interact": "bilinear",
    "has_target": "bilinear",
    "get_target": "bilinear",
}

for (_, relation, _) in edge_types:
    if relation not in ["interact", "has_target", "get_target"]:
        relation_2_decoder[relation] = "dedicom"

latent_dim = 16

input_dim = {node_type: data[node_type].num_features for node_type in data.node_types}
net = HeteroVGAE(
    hidden_dim, latent_dim, data.node_types, data.edge_types, decoder_2_relation,
    relation_2_decoder, num_bases=args.num_bases, input_dim=input_dim, latent_encoder_type=args.latent_encoder_type,
    dropout=args.dropout, device=args.device
).to(args.device)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
num_epoch = args.num_epoch
logger.info(f"Training device: {args.device}")
logger.info(f"Number of basis matrices: {args.num_bases}")

train_data = train_data.to(args.device)
valid_data = valid_data.to(args.device)
best_val_roc = 0
kl_lambda = {
    "drug": 0.9,
    "gene": 0.9,   
}

logger.info("Training...")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True, record_shapes=True) as prof:

    for epoch in range(num_epoch):
        with record_function("train_epoch"):
            start = time.time()
            net.train()
            optimizer.zero_grad()
            z_dict, mu, logstd = net.encode(train_data.x_dict, train_data.edge_index_dict)
            kl_loss = net.kl_loss_all(reduce="ratio", lambda_=kl_lambda)
            recon_loss = net.recon_loss_all_relation(z_dict, train_data.edge_label_index_dict)
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()

        with record_function("eval_epoch"):
            net.eval()
            with torch.no_grad():
                z_dict, mu, logstd = net.encode(valid_data.x_dict, valid_data.edge_index_dict)
                pos_edge_label_index_dict = valid_data.edge_label_index_dict
                edge_label_index_dict = {}
                edge_label_dict = {}
                for edge_type in valid_data.edge_label_index_dict.keys():
                    src, relation, dst = edge_type
                    if relation == "get_target":
                        continue
                    num_nodes = (valid_data.x_dict[src].shape[0], valid_data.x_dict[dst].shape[0])
                    neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type], num_nodes = num_nodes)
                    edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type], 
                                                                    neg_edge_label_index], dim = -1)

                    pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
                    neg_label = torch.zeros(neg_edge_label_index.shape[1])
                    edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim = 0)
                
                edge_pred = net.decode_all_relation(z_dict, edge_label_index_dict, sigmoid = True)
                for relation in edge_pred.keys():
                    edge_pred[relation] = edge_pred[relation].cpu()
                roc_auc, _, _ = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict, edge_types)

            end = time.time()   
            logger.info(f"Epoch: {epoch} | Loss: {loss:.4f} | Val ROC: {roc_auc:.4f} | Best ROC: {best_val_roc:.4f} | Time: {end - start:.2f}s")

            if best_val_roc < roc_auc:
                best_val_roc = roc_auc
                model_name = f"hetero_vgae_bases_{args.num_bases}"
                torch.save(net.state_dict(), args.chkpt_dir + f"{model_name}_{args.seed}.pt")
                logger.info("Save Model")

        if epoch == 5:  # Profile for a few epochs
            break
    
    # Print the profiler results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace("trace.json")

    # You can also save the profiler results to a file
    prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")


# Test phase
test_data = test_data.to(args.device)
net.load_state_dict(torch.load(args.chkpt_dir + f"{model_name}_{args.seed}.pt"))
net.eval()
with torch.no_grad():
    z_dict, mu, logstd = net.encode(test_data.x_dict, test_data.edge_index_dict)
    pos_edge_label_index_dict = test_data.edge_label_index_dict
    edge_label_index_dict = {}
    edge_label_dict = {}
    for edge_type in test_data.edge_label_index_dict.keys():
        src, relation, dst = edge_type
        if relation == "get_target":
            continue
        num_nodes = (test_data.x_dict[src].shape[0], test_data.x_dict[dst].shape[0])
        neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type], num_nodes = num_nodes)
        edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type], 
                                                        neg_edge_label_index], dim = -1)

        pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
        neg_label = torch.zeros(neg_edge_label_index.shape[1])
        edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim = 0)
    
    edge_pred = net.decode_all_relation(z_dict, edge_label_index_dict, sigmoid = True)
    for relation in edge_pred.keys():
        edge_pred[relation] = edge_pred[relation].cpu()
    roc_auc, _, _ = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
    prec, _, _ = cal_average_precision_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
    apk, apk_dict = cal_apk(edge_pred, edge_label_dict, edge_types, k = 50)
    logger.info("-" * 100)
    logger.info(f'Test AUROC: {roc_auc:.4f} | Test AUPRC: {prec:.4f} | Test AP@50: {apk:.4f}')