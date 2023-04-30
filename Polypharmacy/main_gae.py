# main.py

import argparse
from train_hetero_gae import run_experiment
import pickle
parser = argparse.ArgumentParser(description="Polypharmacy Side Effect Prediction")
parser.add_argument("--num_runs", type=int, default=20, help="number of runs with different seeds")
parser.add_argument("--num_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--chkpt_dir", type=str, default="./", help="checkpoint directory")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--device", type=str, default="cpu", help="training device")
parser.add_argument("--pretrained", type=str, default=None, help="pretrained model checkpoint path")
parser.add_argument("--num_bases", type=int, default=None, help="number of basis functions")
parser.add_argument("--patience", type=int, default=20, help="patience for early stopping")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--randomize_ppi", action="store_true", help="randomize protein interactions")
parser.add_argument("--randomize_dpi", action="store_true", help="randomize drug protein interactions")
args = parser.parse_args()
results = {}
input_seed = args.seed
for i in range(input_seed, args.num_runs+input_seed):
    seed = i
    result = run_experiment(seed, args)
    results[seed] = result
    # print(f"Run {i + 1}: {result}")
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

# Data (indepedent weights)
# Aucroc, Auprc, Ap50
# .861, .832 .800 (seed 8)
# .861, .832, .802 (seed 7)
# .844, .817, .787 (seed 6)
# .847, .820, .788 (seed 5)
# .846, .821, .783 (seed 4)
# .871, .846, .840 (seed 3)

# Data (shared weights, num_bases=20)
# Aucroc, Auprc, Ap50
# .930, .909, .901 (seed 7),
# .931, .909, .901 (seed 6)
# .933, .913, .914 (seed 5)
# .928, .908, .910 (seed 4)
# .932, .910, .899 (seed 3)
# .927, .906, .910 (seed 2)

# Data (shared weights, num_bases=1, seed 5)
# 0.8863, 0.8686 0.8801
# Data (shared weights, num_bases=5, seed 5)
# 0.9354, 0.9141 0.9127
# Data (shared weights, num_bases=50, seed 5)
# 0.8859, 0.8591 0.8562
# Data (shared weights, num_bases=10, seed 5)
# 0.9375, 0.9164, 0.9138
# Data (shared weights, num_bases=12, seed 5)
# 0.9372, 0.9158, 0.9142
# Data (shared weights, num_bases=15, seed 5)
# 0.9411, 0.9198, 0.9168
# Randomized PPI (shared weights, num_bases=15, seed 5)
# 0.9348, 0.9127, 0.9067
# Randomized PPI and PDI (No shared weights)
# 0.8988, 0.8728, 0.8712

