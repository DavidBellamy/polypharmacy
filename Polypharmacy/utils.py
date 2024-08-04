from collections import defaultdict
import logging
from logging.handlers import RotatingFileHandler
import os
from typing import DefaultDict

import networkx as nx
import numpy as np
import pandas as pd
import torch

from polypharmacy import MONO_SIDE_EFFECT_PATH


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "polypharmacy.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler() 
    f_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )  

    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        "%(asctime)s - %(message)s"
    )
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def randomize_dataframe_col2_values(df, col_randomize):
    """
    Randomizes the values in col2 of a dataframe while keeping the values in col1 the same.
    Used for randomizing the values in the drug-drug interaction dataset and drug-protein interaction dataset.
    """
    # Create a new column containing a random order of row indices
    print(df.head(3))
    df["random_order"] = np.random.permutation(len(df))
    print("Randomizing column: ", col_randomize, "...")
    # Sort the dataframe by the new column
    df_randomized = df.sort_values("random_order")

    # Reset the index of the randomized dataframe
    df_randomized = df_randomized.reset_index(drop=True)

    # Replace the values in col2 of the randomized dataframe with the corresponding values in col2 of the original dataframe
    df_randomized[col_randomize] = df[col_randomize]
    # Delete the random_order column
    df_randomized = df_randomized.drop(columns=["random_order"])
    print(df_randomized.head(3))
    print("Done randomizing column: ", col_randomize, "!")
    return df_randomized


def load_ppi(filepath="bio-decagon-ppi/bio-decagon-ppi.csv", randomize_ppi=False):
    """
    Loads the protein-protein interaction graph from the Bio-decagon dataset.
    """
    df = pd.read_csv(filepath)
    if randomize_ppi:
        df = randomize_dataframe_col2_values(df, "Gene 2")
    print("Load Protein-Protein Interaction Graph")
    src, dst = df["Gene 1"].tolist(), df["Gene 2"].tolist()
    del df
    nodes = set(src + dst)
    net = nx.Graph()
    net.add_edges_from(zip(src, dst), verbose=False)
    gene_2_idx = {}

    for idx, node in enumerate(nodes):
        gene_2_idx[node] = idx
    print("Num nodes: ", len(nodes))
    print("Num edges: ", len(net.edges()))
    print()
    return net, gene_2_idx


def load_targets(
    filepath="bio-decagon-targets/bio-decagon-targets.csv", randomize_dpi=False
):
    """
    Loads the drug-target interaction graph from the Bio-decagon dataset.
    """
    df = pd.read_csv(filepath)
    if randomize_dpi:
        df = randomize_dataframe_col2_values(df, "Gene")
    print("Load Drug-Target Interaction Graph")
    print("Num of interaction: ", df.shape[0])
    print()

    stitch_ids = df["STITCH"].tolist()  # drug
    genes = df["Gene"].tolist()  # target
    stitch_2_gene = DefaultDict(set)  # drug -> target
    for stitch_id, gene in zip(stitch_ids, genes):
        stitch_2_gene[stitch_id].add(gene)

    return stitch_2_gene  # drug -> target


def load_categories(
    filepath="bio-decagon-effectcategories/bio-decagon-effectcategories.csv",
):
    """
    Loads the side effect categories from the Bio-decagon dataset.
    """
    df = pd.read_csv(filepath)

    side_effects = df["Side Effect"]
    side_effect_names = df["Side Effect Name"]
    disease_classes = df["Disease Class"]

    side_effect_2_name = {}  # side effect -> name
    side_effect_2_class = {}  # side effect -> disease class

    for side_effect, name, class_ in zip(
        side_effects, side_effect_names, disease_classes
    ):
        side_effect_2_name[side_effect] = name
        side_effect_2_class[side_effect] = class_
    # Returns a dictionary mapping side effects to their names and
    # a dictionary mapping side effects to their disease classes.
    return side_effect_2_class, side_effect_2_name


def load_combo_side_effect(filepath="bio-decagon-combo/bio-decagon-combo.csv"):
    """
    Loads the drug combination side effect graph from the Bio-decagon dataset.
    Returns: combo_2_side_effect containing drug combinations and their side effects,
    side_effect_2_combo containing side effects and their drug combinations,
    side_effect_2_name containing side effects and their names,
    combo_2_stitch containing drug combinations and their drugs,
    stitch_2_idx containing drugs and their unique indices.
    """
    df = pd.read_csv(filepath)
    print("Load Combination Side Effect Graph")
    combo_2_side_effect = defaultdict(set)  # drug combination -> side effect
    side_effect_2_combo = defaultdict(set)  # side effect -> drug combination
    side_effect_2_name = {}
    combo_2_stitch = {}

    stitch_ids_1 = df["STITCH 1"].tolist()  # drug 1
    stitch_ids_2 = df["STITCH 2"].tolist()  # drug 2
    side_effects = df["Polypharmacy Side Effect"].tolist()  # side effect
    side_effect_names = df["Side Effect Name"].tolist()  # side effect name
    combos = (df["STITCH 1"] + "_" + df["STITCH 2"]).tolist()  # drug combination
    del df
    stitch_set = set()  # drug set
    items = zip(stitch_ids_1, stitch_ids_2, side_effects, side_effect_names, combos)
    # drug 1, drug 2, side effect, side effect name, drug combination
    for stitch_id_1, stitch_id_2, side_effect, side_effect_name, combo in items:
        # loop through all the drug combinations and side effects
        combo_2_side_effect[combo].add(side_effect)  # drug combination -> side effect
        side_effect_2_combo[side_effect].add(combo)  # side effect -> drug combination
        side_effect_2_name[side_effect] = side_effect_name  # side effect -> name
        combo_2_stitch[combo] = [
            stitch_id_1,
            stitch_id_2,
        ]  # drug combination -> drug1, drug2
        stitch_set.add(stitch_id_1)  # add drug1 to drug set
        stitch_set.add(stitch_id_2)  # add drug2 to drug set

    stitch_set = list(stitch_set)
    stitch_set.sort()
    stitch_2_idx = {}
    idx = 0
    for stitch in stitch_set:
        # map drug to unique index
        stitch_2_idx[stitch] = idx
        idx += 1

    num_interactions = sum(len(v) for v in combo_2_side_effect.values())
    print("Number of drug combinations: ", len(combo_2_stitch))
    print("Number of side effects: ", len(side_effect_2_name))
    print("Number of interactions: ", num_interactions)
    print()
    return (
        combo_2_side_effect,
        side_effect_2_combo,
        side_effect_2_name,
        combo_2_stitch,
        stitch_2_idx,
    )


def convert_combo_side_effect_to_edge_index_list(
    side_effect_2_combo, combo_2_stitch, stitch_2_idx
):
    """
    Input: side_effect_2_combo: Containing side effects and their drug combinations,
    combo_2_stitch: containing drug combinations and their drugs,
    stitch_2_idx: containing drugs and their unique indices.

    Returns: stitch_2_idx: containing drugs and their unique indices,
    edge_index_dict: maps side effects types to their drug combinations
    This code is building a dictionary called edge_index_dict that represents the edge indices in a graph,
    where the nodes are drugs and the edges represent the side effects between the drug pairs.
    """
    edge_index_dict = defaultdict(list)
    for se in side_effect_2_combo.keys():  # loop through all the side effects
        for combo in side_effect_2_combo[
            se
        ]:  # loop through all the drug combinations for each side effect
            s1, s2 = combo_2_stitch[combo]  # get the drugs for each drug combination
            edge_index_dict[("drug", se, "drug")].append(
                [stitch_2_idx[s1], stitch_2_idx[s2]]
            )

        if len(edge_index_dict[("drug", se, "drug")]) < 500:
            del edge_index_dict[("drug", se, "drug")]
            # delete the side effect if the number of interactions is less than 500
        else:
            # convert the edge indices to tensor
            edge_index_dict[("drug", se, "drug")] = (
                torch.tensor(edge_index_dict[("drug", se, "drug")]).long().T
            )
    # Returns a dictionary edge_index_dict mapping side effects types to their drug combinations
    # and a dictionary stitch_2_idx mapping drugs to their unique indices.

    return stitch_2_idx, edge_index_dict


def load_mono_side_effect(filepath=MONO_SIDE_EFFECT_PATH):
    df = pd.read_csv(filepath)
    print("Load Mono Side Effect\n")
    stitch_ids = df["STITCH"]
    side_effects = df["Individual Side Effect"]
    side_effect_names = df["Side Effect Name"]

    items = zip(stitch_ids, side_effects, side_effect_names)

    stitch_2_side_effect = defaultdict(set)
    side_effect_2_name = {}

    for stitch_id, side_effect, side_effect_name in items:
        stitch_2_side_effect[stitch_id].add(side_effect)
        side_effect_2_name[side_effect] = side_effect_name
    # dictionary edge_index_dict represents the edge indices in a graph,
    # where the nodes are drugs and the edges represent the side effects between the drug pairs.
    return stitch_2_side_effect, side_effect_2_name


def generate_morgan_fingerprint(stitch_2_smile, stitch_2_idx):
    num_drugs = len(stitch_2_idx)
    x = np.identity(num_drugs)
    features = [0 for i in range(num_drugs)]
    for stitch, idx in stitch_2_idx.items():
        features[idx] = featurize.molconvert.smiles2ECFP4(stitch_2_smile[stitch])
    augment = np.array(features)
    return augment
