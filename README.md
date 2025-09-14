# SecureFedPROM: A Zero-Trust Federated Learning Approach with Multi-Criteria Client Selection

This repository contains a part of the experiments conducted for paper:

**SecureFedPROM: A Zero-Trust Federated Learning Approach with Multi-Criteria Client Selection**   

Published in IEEE Journal on Selected Areas in Communications 2025

Our code is built on top of the [LEAF](https://github.com/TalwalkarLab/leaf) framework and extends it to support client selection strategies based on multiple criteria such as hardware, network, cost, etc.

## Features
- Extension of LEAF with multi-criterion client ranking and selection
- Implementation of the algorithms described in the paper
- Configurable selection criteria and budget constraints

## Installation

```bash
git clone https://github.com/mehreentahir16/securefedprom.git
cd securefedprom
pip install -r requirements.txt
```

## Usage

Example command to run client selection experiments:

```bash
python3 main.py -dataset femnist -model cnn --eval-every 1 --num-rounds 50 --clients-per-round 20 --client-selection-strategy random -lr 0.004 --metrics-name femnist_experiment_random
```

## Citation

If you use this code, please cite our paper:

@ARTICLE{10966024,

  author={Tahir, Mehreen and Mawla, Tanjila and Awaysheh, Feras and Alawadi, Sadi and Gupta, Maanak and Intizar Ali, Muhammad},

  journal={IEEE Journal on Selected Areas in Communications}, 

  title={SecureFedPROM: A Zero-Trust Federated Learning Approach With Multi-Criteria Client Selection}, 

  year={2025},

  volume={43},

  number={6},

  pages={2025-2041},

  keywords={Training;Data models;Computational modeling;Servers;Protocols;Data privacy;Registers;MCDM;Federated learning;Convergence;Federated learning;multi-criteria client selection;access control;zero-trust federated learning},

  doi={10.1109/JSAC.2025.3560008}
  
  }


## Acknowledgments

This code is based on the [LEAF framework](https://github.com/TalwalkarLab/leaf).