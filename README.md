# Drug-Drug Side-effect Prediction With Efficient Weight Sharing

Contributors:
* Bhawesh Kumar
Other Contributors (Bhawesh implemented weight sharing on their codebase):
* Ngo Nhat Khang 
* Hy Truong Son 

Papers:
* Predicting Polypharmacy Side-effects with Efficient Weight Sharing https://drive.google.com/file/d/11St8wHn8CI8TCckoX8qm2OJuryKvPuIW/view?usp=sharing
## Requirements
- [Pytorch](https://pytorch.org/)
- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)\
Recommend using Conda for easy installation. 
## Data
Make sure a Data folder is created in each data's subfolder. Then, you should donwload data from the links below and locate them into the Data folders as:
  ```
   
    └── Polypharmacy                
    │   ├── Data 
    │   │   ├── bio-decagon-combo.csv
    │   │   ├── bio-decagon-mono.csv
    │   │   ├── bio-decagon-ppi.csv
    │   │   ├── bio-decagon-targets.csv
    │   ├── models
    │   │   ├── trained_models
    │   │   │   ├──...
    │   │   ├── decoder_module.py
    │   │   ├── hetero_gae.py
    │   │   ├── hetero_gae_shared.py
    │   ├── data.py
    │   ├── main_gae.py
    │   ├── metrics.py
    │   ├── train_hetero_gae.py
    │   ├── utils.py
    └── README.md
   ```
#### Polypharmacy 
Download from [Decagon](https://github.com/mims-harvard/decagon)

#### Polypharmacy
- Train GAE without shared basis
  ```bash
    cd Polypharmacy/
    python main_gae.py --num_epoch 1000 --lr 3e-3 --num_runs 1 --chkpt_dir ./models/trained_models --patience 25 --seed 5
  ```
- Train GAE without shared basis with randomization of Protein-Protein Interaction and Protein-Drug Interaction data
  ```bash
    cd Polypharmacy/
    python main_gae.py --num_epoch 1000 --lr 3e-3 --num_runs 1 --chkpt_dir ./models/trained_models --patience 25 --seed 5 --randomize_ppi --randomize_dpi
  ```
  Train GAE with shared basis (15 basis vectors here)
  ```bash
    cd Polypharmacy/
    python main_gae.py  --num_bases 15 --num_epoch 1000 --lr 3e-3 --num_runs 1 --chkpt_dir ./models/trained_models_shared --patience 25 --seed 5 
  ```
  Train GAE with shared basis (15 basis vectors here) with randomization of Protein-Protein Interaction and Protein-Drug Interaction data
  ```bash
    cd Polypharmacy/
    python main_gae.py  --num_bases 15 --num_epoch 1000 --lr 3e-3 --num_runs 1 --chkpt_dir ./models/trained_models_shared --patience 25 --seed 5 --randomize_ppi --randomize_dpi
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


