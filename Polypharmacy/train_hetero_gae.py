import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils

from models.hetero_gae import HeteroGAE
from metrics import *
from metrics import *
import time
import numpy as np
from data import *
import os
import warnings

warnings.filterwarnings("ignore")

def run_experiment(seed, args):

    # parser = argparse.ArgumentParser(description="Polypharmacy Side Effect Prediction")
    # parser.add_argument("--seed", type=int, default=1, help="random seed")
    # parser.add_argument("--num_epoch", type=int, default=300, help="number of epochs")
    # parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    # parser.add_argument("--chkpt_dir", type=str, default="./checkpoint/", help="checkpoint directory")
    # parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    # parser.add_argument("--device", type=str, default="cuda:0", help="training device")
    # parser.add_argument("--pretrained", type=str, default=None, help="pretrained model checkpoint path")
    # parser.add_argument("--num_bases", type=int, default=None, help="number of basis functions")
    # parser.add_argument("--save_model", action="store_true", help="save model")
    # parser.add_argument("--randomize_ppi", action="store_true", help="randomize protein interactions")
    # parser.add_argument("--randomize_dpi", action="store_true", help="randomize drug protein interactions")
    # args = parser.parse_args()
    torch_geometric.seed_everything(seed)
    print("Running experiment with seed: ", seed)
    print("Load data")
    if args.randomize_ppi:
        print("Not Using Protein-Protein Interactions")
    if args.randomize_dpi:
        print("Not Using Drug-Protein Interactions")
    data = load_data(args.randomize_ppi, args.randomize_dpi)
    edge_types = data.edge_types
    rev_edge_types = []

    # Create reverse edge types based on the original edge types.
    # This is used later in the RandomLinkSplit transformation

    for (src, relation, dst) in edge_types:
        rev_relation = f"rev_{relation}"
        rev_edge_types.append((dst, rev_relation, src))

    transform = pyg_T.Compose([
        pyg_T.AddSelfLoops(),
        pyg_T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                              edge_types=edge_types, rev_edge_types=rev_edge_types,
                              neg_sampling_ratio=0.0, disjoint_train_ratio=0.2)])

    train_data, valid_data, test_data = transform(data)
    # data split into train, valid, test (each one is a object describing a heterogeneous graph)

    for node in data.node_types:
        train_data[node].x = train_data[node].x.to_sparse().double()
        valid_data[node].x = valid_data[node].x.to_sparse().double()
        test_data[node].x = test_data[node].x.to_sparse().double()

    # Remove the reverse edge types from the train, validation, and test datasets
    # as they were only needed for the RandomLinkSplit transformation:
    for edge_type in rev_edge_types:
        del train_data[edge_type]
        del valid_data[edge_type]
        del test_data[edge_type]

    # For edge types where the source and destination node types are the same,
    # make the edge indices undirected in the train, validation, and test datasets
    for edge_type in edge_types:
        if edge_type[0] == edge_type[2]:
            train_data[edge_type].edge_index = pyg_utils.to_undirected(train_data[edge_type].edge_index)
            valid_data[edge_type].edge_index = pyg_utils.to_undirected(valid_data[edge_type].edge_index)
            test_data[edge_type].edge_index = pyg_utils.to_undirected(test_data[edge_type].edge_index)

    print("Initialize model...")
    hidden_dim = [64, 32]  # hidden dimensions of the encoder
    num_layer = 2  # number of layers in the encoder

    # The decoder is a bilinear decoder for the "interact", "has_target", and "get_target" relations
    # and a dedicom decoder for all other relations (the drug-drug interaction relations)

    decoder_2_relation = {
        "bilinear": ["interact", "has_target", "get_target"],
        "dedicom": [relation for (_, relation, _) in edge_types
                    if relation not in ["interact", "has_target", "get_target"]]}

    relation_2_decoder = {
        "interact": "bilinear",
        "has_target": "bilinear",
        "get_target": "bilinear",
    }

    # Add the dedicom decoder for all other relations (the drug-drug interaction relations)
    for (_, relation, _) in edge_types:
        if relation not in ["interact", "has_target", "get_target"]:
            relation_2_decoder[relation] = "dedicom"

    # Set the output dimension, which is the same as the last hidden layer's dimension
    out_dim = hidden_dim[-1]
    input_dim = {"drug": train_data.x_dict["drug"].shape[1], "gene": train_data.x_dict["gene"].shape[1]}
    # Initialize the model
    net = HeteroGAE(hidden_dim, out_dim, data.node_types, data.edge_types, decoder_2_relation,
                    relation_2_decoder, num_bases=args.num_bases, input_dim=input_dim, dropout=args.dropout,
                    device=args.device).to(args.device)
    net = net.to(torch.double)

    # Load the pre-trained model checkpoint if provided as an argument
    if args.pretrained:
        print(f"Loading pre-trained model from {args.pretrained}")
        net.load_state_dict(torch.load(args.pretrained))

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    num_epoch = args.num_epoch

    print("Training device: ", args.device)
    train_edge_label_index_dict = train_data.edge_label_index_dict
    train_data = train_data.to(args.device)
    valid_data = valid_data.to(args.device)
    best_val_roc = 0  # best validation ROC-AUC score intialized to 0
    print("Training...")  # Training loop
    patience_counter = 0 # initialize the patience counter
    for epoch in range(num_epoch):
        start = time.time()
        net.train()  # set the model to training mode
        optimizer.zero_grad()  # clear the gradients
        z_dict = net.encode(train_data.x_dict, train_data.edge_index_dict)  # encode the graph
        pos_edge_label_index_dict = train_data.edge_label_index_dict  # get the positive edge indices
        edge_label_index_dict = {}  # initialize the dictionary for the edge indices
        edge_label_dict = {}  # initialize the dictionary for the edge labels
        for edge_type in edge_types:  # for each edge type
            src, relation, dst = edge_type  # get the source, relation, and destination node types
            if relation == "get_target":  # skip the "get_target" relation
                continue
            #
            num_nodes = (train_data.x_dict[src].shape[0], train_data.x_dict[dst].shape[0])
            neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type],
                                                               num_nodes=num_nodes)
            # negative sampling for the edge indices
            edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type],
                                                          neg_edge_label_index], dim=-1)

            pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])  # positive edge labels
            neg_label = torch.zeros(neg_edge_label_index.shape[1])  # negative edge labels
            edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim=0)  # concatenate the edge labels

        edge_pred = net.decode_all_relation(z_dict, edge_label_index_dict)  # decode the edge labels
        edge_pred = torch.cat([edge_pred[relation] for relation in edge_pred.keys()], dim=-1)
        edge_label = torch.cat([edge_label_dict[relation] for relation in edge_label_dict.keys()], dim=-1).to(
            args.device)
        loss = loss_fn(edge_pred, edge_label)
        loss.backward()
        optimizer.step()
        loss = loss.detach().item()

        net.eval()
        with torch.no_grad():
            z_dict = net.encode(valid_data.x_dict, valid_data.edge_index_dict)
            pos_edge_label_index_dict = valid_data.edge_label_index_dict
            edge_label_index_dict = {}
            edge_label_dict = {}
            for edge_type in edge_types:
                src, relation, dst = edge_type
                if relation == "get_target":
                    continue
                num_nodes = (train_data.x_dict[src].shape[0], train_data.x_dict[dst].shape[0])
                neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type],
                                                                   num_nodes=num_nodes)
                edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type],
                                                              neg_edge_label_index], dim=-1)

                pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
                neg_label = torch.zeros(neg_edge_label_index.shape[1])
                edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim=0)

            edge_pred = net.decode_all_relation(z_dict, edge_label_index_dict)
            for relation in edge_pred.keys():
                edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
            roc_auc, roc_auc_dict_, counts_dict_ = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict,
                                                                                   edge_types)

        end = time.time()
        print(f"| Epoch: {epoch} | Loss: {loss} | Val ROC: {roc_auc} | Best ROC: {best_val_roc} | Time: {end - start}")

        if best_val_roc < roc_auc:
            best_val_roc = roc_auc
            patience_counter = 0
            torch.save(net.state_dict(), args.chkpt_dir + f"/gae_{seed}.pt")
            print("---- Save Model ----")
        else:
            patience_counter += 1
            print("patience counter: {}".format(patience_counter))

        if patience_counter >= args.patience and epoch > 50:
            print("Early stopping due to no improvement in validation ROC-AUC score for {} epochs".format(args.patience))
            break

    test_data = test_data.to(args.device)
    net.load_state_dict(torch.load(args.chkpt_dir + f"/gae_{seed}.pt"))
    net.eval()
    with torch.no_grad():
        z_dict = net.encode(test_data.x_dict, test_data.edge_index_dict)
        pos_edge_label_index_dict = test_data.edge_label_index_dict
        edge_label_index_dict = {}
        edge_label_dict = {}
        for edge_type in test_data.edge_label_index_dict.keys():
            src, relation, dst = edge_type
            if relation == "get_target":
                continue
            num_nodes = (test_data.x_dict[src].shape[0], test_data.x_dict[dst].shape[0])
            neg_edge_label_index = pyg_utils.negative_sampling(pos_edge_label_index_dict[edge_type],
                                                               num_nodes=num_nodes)
            edge_label_index_dict[edge_type] = torch.cat([pos_edge_label_index_dict[edge_type],
                                                          neg_edge_label_index], dim=-1)

            pos_label = torch.ones(pos_edge_label_index_dict[edge_type].shape[1])
            neg_label = torch.zeros(neg_edge_label_index.shape[1])
            edge_label_dict[relation] = torch.cat([pos_label, neg_label], dim=0)

        edge_pred = net.decode_all_relation(z_dict, edge_label_index_dict)
        for relation in edge_pred.keys():
            edge_pred[relation] = F.sigmoid(edge_pred[relation]).cpu()
        roc_auc, roc_auc_dict, counts_dict = cal_roc_auc_score_per_side_effect(edge_pred, edge_label_dict, edge_types)
        prec, prec_dict, counts_dict_2 = cal_average_precision_score_per_side_effect(edge_pred, edge_label_dict,
                                                                                     edge_types)
        apk, apk_dict = cal_apk(edge_pred, edge_label_dict, edge_types, k=50)
        print("-" * 100)
        print()
        print(f'| Test AUROC: {roc_auc} | Test AUPRC: {prec} | Test AP@50: {apk}')

    # Comment out the following line if you want to keep the best model
    # model_path = args.chkpt_dir + f"/gae_{seed}.pt"
    # if os.path.exists(model_path) :
    #     os.remove(model_path)
    #     print("---- Deleted Best Model ----")
    return {
        "seed": seed,
        "auroc": roc_auc,
        "auprc": prec,
        "ap50": apk,
        "counts_dict_1": counts_dict,
        "prec_dict": prec_dict,
        "apk_dict": apk_dict,
        "roc_auc_dict": roc_auc_dict
    }


