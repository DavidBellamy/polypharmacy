import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler

from polypharmacy.data_pl import DataModule
from polypharmacy.models.hetero_vgae_pl import HeteroVGAE_PL


def main():
    parser = argparse.ArgumentParser(description="Polypharmacy Side Effect Prediction")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--num_epoch", type=int, default=300, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--chkpt_dir", type=str, default="./checkpoint/", help="checkpoint directory"
    )
    parser.add_argument(
        "--latent_encoder_type", type=str, default="linear", help="latent encoder type"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument(
        "--num_bases",
        type=int,
        default=None,
        help="number of basis matrices for weight sharing (0 for no weight sharing)",
    )

    args = parser.parse_args()

    pl.seed_everything(42)

    data_module = DataModule(batch_size=1, num_workers=0)
    data_module.prepare_data()

    decoder_2_relation = {
        "bilinear": ["interact", "has_target", "get_target"],
        "dedicom": [
            relation
            for (_, relation, _) in data_module.data.edge_types
            if relation not in ["interact", "has_target", "get_target"]
        ],
    }

    relation_2_decoder = {
        "interact": "bilinear",
        "has_target": "bilinear",
        "get_target": "bilinear",
    }

    for _, relation, _ in data_module.data.edge_types:
        if relation not in ["interact", "has_target", "get_target"]:
            relation_2_decoder[relation] = "dedicom"

    input_dim = {
        node_type: data_module.data[node_type].num_features
        for node_type in data_module.data.node_types
    }

    model = HeteroVGAE_PL(
        hidden_dims=[64, 32],
        out_dim=16,
        node_types=data_module.data.node_types,
        edge_types=data_module.data.edge_types,
        decoder_2_relation=decoder_2_relation,
        relation_2_decoder=relation_2_decoder,
        num_bases=args.num_bases,
        input_dim=input_dim,
        latent_encoder_type=args.latent_encoder_type,
        dropout=args.dropout,
        lr=args.lr,
        kl_lambda={
            "drug": 0.9,
            "gene": 0.9,
        },
    )

    profiler = AdvancedProfiler(dirpath=".", filename="profiler_report")

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger("lightning_logs/"),
        callbacks=[ModelCheckpoint(monitor="val_roc_auc", mode="max")],
        profiler=profiler,
        log_every_n_steps=1,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
