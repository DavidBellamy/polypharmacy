import argparse
import pickle

from polypharmacy.train_hetero_gae import run_experiment


parser = argparse.ArgumentParser(description="Polypharmacy Side Effect Prediction")
parser.add_argument(
    "--num_runs", type=int, default=20, help="number of runs with different seeds"
)
parser.add_argument("--num_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--chkpt_dir", type=str, default="./", help="checkpoint directory")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--device", type=str, default="cpu", help="training device")
parser.add_argument(
    "--pretrained", type=str, default=None, help="pretrained model checkpoint path"
)
parser.add_argument(
    "--num_bases", type=int, default=None, help="number of basis functions"
)
parser.add_argument(
    "--patience", type=int, default=20, help="patience for early stopping"
)
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument(
    "--randomize_ppi", action="store_true", help="randomize protein interactions"
)
parser.add_argument(
    "--randomize_dpi", action="store_true", help="randomize drug protein interactions"
)
args = parser.parse_args()

results = {}
input_seed = args.seed
for i in range(input_seed, args.num_runs + input_seed):
    seed = i
    result = run_experiment(seed, args)
    results[seed] = result
    print(f"Run {i + 1}: {result['auroc']:.4f}", end=None)
    print(f"Run {i + 1}: {result['auprc']:.4f}", end=None)
    print(f"Run {i + 1}: {result['ap50']:.4f}", end=None)
# Save or process results as desired
if args.num_bases is None:
    if args.randomize_ppi:
        if args.randomize_dpi:
            with open("results_randomized_both.pkl", "wb") as f:
                pickle.dump(results, f)
        else:
            with open("results_randomized_ppi.pkl", "wb") as f:
                pickle.dump(results, f)

    else:
        with open("results.pkl", "wb") as f:
            pickle.dump(results, f)


else:
    if args.randomize_ppi:
        if args.randomize_dpi:
            with open("results_randomized_both_shared.pkl", "wb") as f:
                pickle.dump(results, f)
        else:
            with open("results_randomized_ppi_shared.pkl", "wb") as f:
                pickle.dump(results, f)
    else:
        with open("results_shared_basis.pkl", "wb") as f:
            pickle.dump(results, f)
