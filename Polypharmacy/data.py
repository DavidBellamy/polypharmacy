from collections import defaultdict
import json

import torch
import torch_geometric.utils as pyg_utils
import torch_geometric.data as pyg_data

from polypharmacy import COMBO_SIDE_EFFECT_PATH, DRUG_GENE_PATH, PPI_PATH
from polypharmacy.utils import (
    convert_combo_side_effect_to_edge_index_list,
    generate_morgan_fingerprint,
    load_combo_side_effect,
    load_ppi,
    load_targets,
)


def load_data(randomize_ppi=False, randomize_dpi=False, return_augment=False):
    """
    Loads and processes different types of biological data to create a PyTorch Geometric Heterogeneous Graph.

    Returns a heterogeneous graph with different types of nodes (genes and drugs)
    and relationships (protein-protein, drug-drug, and drug-protein)

    """
    """ protein - protein """
    randomize_ppi = randomize_ppi
    randomize_dpi = randomize_dpi
    net, gene_2_idx = load_ppi(
        PPI_PATH, randomize_ppi=randomize_ppi
    )  # net: networkx graph from protein-protein interaction

    gene_edge_list = list(net.edges.data())  # list of edges
    gene_edge_index = [
        [gene_2_idx[gene_edge_list[i][0]], gene_2_idx[gene_edge_list[i][1]]]
        for i in range(len(gene_edge_list))
    ]
    # list of edges as identified by their indices in the gene_2_idx dictionary
    gene_edge_index = torch.tensor(gene_edge_index).long().T

    """ drug - drug """
    (
        combo_2_se,
        se_2_combo,
        se_2_name,
        combo_2_stitch,
        stitch_2_idx,
    ) = load_combo_side_effect(COMBO_SIDE_EFFECT_PATH)
    # combo_2_se: dictionary of side effects for each drug combination
    # se_2_combo: dictionary of drug combinations for each side effect
    # se_2_name: dictionary of side effect names for each side effect
    # combo_2_stitch: dictionary of stitch ids for each drug combination
    # stitch_2_idx: dictionary of indices for each stitch id
    edge_index_dict = defaultdict(list)

    drug_2_idx, edge_index_dict = convert_combo_side_effect_to_edge_index_list(
        se_2_combo, combo_2_stitch, stitch_2_idx
    )

    print("Number of side effects in consideration: ", len(edge_index_dict))

    """ drug - protein """
    drug_2_gene = load_targets(DRUG_GENE_PATH, randomize_dpi=randomize_dpi)
    drug_gene_edge_index = []
    for stitch, genes in drug_2_gene.items():
        for gene in genes:
            try:
                drug_gene_edge_index.append([stitch_2_idx[stitch], gene_2_idx[gene]])
            except:
                pass

    drug_gene_edge_index = torch.tensor(drug_gene_edge_index).long().T

    index = torch.LongTensor([1, 0])
    gene_drug_edge_index = torch.zeros_like(drug_gene_edge_index)
    gene_drug_edge_index[index] = drug_gene_edge_index

    edge_index_dict[("drug", "has_target", "gene")] = drug_gene_edge_index
    edge_index_dict[("gene", "get_target", "drug")] = gene_drug_edge_index
    edge_index_dict[("gene", "interact", "gene")] = gene_edge_index

    data = pyg_data.HeteroData()

    data["gene"].x = torch.eye(len(gene_2_idx))  # one-hot encoding of proteins
    data["drug"].x = torch.eye(len(stitch_2_idx))  # one-hot encoding of drugs

    if return_augment:
        with open(DRUG_SMILE_PATH, "r") as f:
            stitch_2_smile = json.load(f)
        data["drug"].augment = torch.tensor(
            generate_morgan_fingerprint(stitch_2_smile, stitch_2_idx)
        )

    for src, relation, dst in edge_index_dict.keys():
        data[src, relation, dst].edge_index = edge_index_dict[(src, relation, dst)]

    for edge_type in data.edge_types:
        data[edge_type].edge_index = pyg_utils.sort_edge_index(
            data[edge_type].edge_index
        )

    return data
