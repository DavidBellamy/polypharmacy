import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as pyg_T
import torch_geometric.utils as pyg_utils

from polypharmacy.data import load_data


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=16):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def prepare_data(self):
        self.data = load_data()

    def setup(self, stage=None):
        if self.data is None:
            self.data = load_data()

        self.edge_types = self.data.edge_types
        rev_edge_types = []

        for src, relation, dst in self.edge_types:
            rev_relation = f"rev_{relation}"
            rev_edge_types.append((dst, rev_relation, src))

        self.transform = pyg_T.Compose(
            [
                pyg_T.AddSelfLoops(),
                pyg_T.RandomLinkSplit(
                    num_val=0.1,
                    num_test=0.1,
                    is_undirected=True,
                    edge_types=self.data.edge_types,
                    rev_edge_types=rev_edge_types,
                    neg_sampling_ratio=0.0,
                    disjoint_train_ratio=0.2,
                ),
            ]
        )
        self.train_data, self.valid_data, self.test_data = self.transform(self.data)

        for node in self.data.node_types:
            self.train_data[node].x = self.train_data[node].x.to_sparse().float()
            self.valid_data[node].x = self.valid_data[node].x.to_sparse().float()
            self.test_data[node].x = self.test_data[node].x.to_sparse().float()

        for edge_type in rev_edge_types:
            del self.train_data[edge_type]
            del self.valid_data[edge_type]
            del self.test_data[edge_type]

        for edge_type in self.edge_types:
            if edge_type[0] == edge_type[2]:
                self.train_data[edge_type].edge_index = pyg_utils.to_undirected(
                    self.train_data[edge_type].edge_index
                )
                self.valid_data[edge_type].edge_index = pyg_utils.to_undirected(
                    self.valid_data[edge_type].edge_index
                )
                self.test_data[edge_type].edge_index = pyg_utils.to_undirected(
                    self.test_data[edge_type].edge_index
                )

    def train_dataloader(self):
        return DataLoader([self.train_data], batch_size=1, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader([self.valid_data], batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader([self.test_data], batch_size=1, num_workers=self.num_workers)

