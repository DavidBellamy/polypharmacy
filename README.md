# Drug-Drug Side-effect Prediction With Efficient Weight Sharing

Contributors:
* Bhawesh Kumar
* David R. Bellamy


Paper:
* Predicting Polypharmacy Side-effects with Efficient Weight Sharing https://drive.google.com/file/d/1AcFKcBVm-eqZEPA62uHcjAnxZkYBGPNL/view?usp=sharing

## Setup

Using Python 3.11, create a virtual environment and install the required packages:
```bash
python3.11 -m venv .env
source .env/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Downloading evaluation data
The evaluation data comes from [Decagon](http://snap.stanford.edu/decagon/). Run the following commands to download the data:
```bash
python setup_data.py
```

Make sure the `data/` folder is created and the `test/` folder within it.
  ```
    .
    ├── data
    │   ├── bio-decagon-combo.csv
    │   ├── bio-decagon-mono.csv
    │   ├── bio-decagon-ppi.csv
    │   ├── bio-decagon-targets.csv
    │   └── test
    │       ├── bio-decagon-combo.csv
    │       ├── bio-decagon-mono.csv
    │       ├── bio-decagon-ppi.csv
    │       └── bio-decagon-targets.csv
    ├── polypharmacy
    │   ├── __init__.py
    │   ├── models
    │   │   ├── decoder_module.py
    │   │   ├── hetero_gae.py
    │   │   └── hetero_gae_shared.py
    │   ├── train_hetero_gae.py
    │   ├── data.py
    │   ├── main_gae.py
    │   ├── metrics.py
    │   └── utils.py
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    └── setup_data.py
   ```

- Train GAE without shared basis
  ```bash
    cd Polypharmacy/
    python main_gae.py --num_epoch 1000 --lr 3e-3 --num_runs 1 --chkpt_dir trained_models --patience 25 --seed 5
  ```
- Train GAE without shared basis with randomization of Protein-Protein Interaction and Protein-Drug Interaction data
  ```bash
    cd Polypharmacy/
    python main_gae.py --num_epoch 1000 --lr 3e-3 --num_runs 1 --chkpt_dir trained_models --patience 25 --seed 5 --randomize_ppi --randomize_dpi
  ```
  Train GAE with shared basis (15 basis vectors here)
  ```bash
    cd Polypharmacy/
    python main_gae.py  --num_bases 15 --num_epoch 1000 --lr 3e-3 --num_runs 1 --chkpt_dir trained_models_shared --patience 25 --seed 5 
  ```
  Train GAE with shared basis (15 basis vectors here) with randomization of Protein-Protein Interaction and Protein-Drug Interaction data
  ```bash
    cd Polypharmacy/
    python main_gae.py  --num_bases 15 --num_epoch 1000 --lr 3e-3 --num_runs 1 --chkpt_dir trained_models_shared --patience 25 --seed 5 --randomize_ppi --randomize_dpi
  ```

## Citations
```bibtex
@misc{ngo2022predicting,
      title={Predicting Drug-Drug Interactions using Deep Generative Models on Graphs}, 
      author={Nhat Khang Ngo and Truong Son Hy and Risi Kondor},
      year={2022},
      eprint={2209.09941},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```

```bibtex
@article{Zitnik2018,
  title={Modeling polypharmacy side effects with graph convolutional networks},
  author={Zitnik, Marinka and Agrawal, Monica and Leskovec, Jure},
  journal={Bioinformatics},
  volume={34},
  number={13},
  pages={457–466},
  year={2018}
}

```


