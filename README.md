# Multi-Criterion Client Selection for Efficient Federated Learning

This repository contains the official implementation of the paper:

**Multi-Criterion Client Selection for Efficient Federated Learning**  
Published in Proceedings of the AAAI Symposium Series 2024  

Our code is built on top of the [LEAF](https://github.com/TalwalkarLab/leaf) framework and extends it to support client selection strategies based on multiple criteria such as hardware, network, cost, etc.

## Features
- Extension of LEAF with multi-criterion client ranking and selection
- Implementation of the algorithms described in the paper
- Configurable selection criteria and budget constraints

## Installation

```bash
git clone https://github.com/mehreentahir16/multi-criteria-client-selection-FL.git
cd multi-criteria-client-selection-FL
pip install -r requirements.txt
```

## Usage

Example command to run client selection experiments:

```bash
python3 main.py -dataset femnist -model cnn --eval-every 1 --num-rounds 50 --clients-per-round 20 --client-selection-strategy random -lr 0.004 --metrics-name femnist_experiment_random
```

## Citation

If you use this code, please cite our paper:

@article{Tahir_Ali_2024, 
  title={Multi-Criterion Client Selection for Efficient Federated Learning}, 
  volume={3}, 
  url={https://ojs.aaai.org/index.php/AAAI-SS/article/view/31227}, 
  DOI={10.1609/aaaiss.v3i1.31227}, 
  abstractNote={Federated Learning (FL) has received tremendous attention as a decentralized machine learning (ML) framework that allows distributed data owners to collaboratively train a global model without sharing raw data. Since FL trains the model directly on edge devices, the heterogeneity of participating clients in terms of data distribution, hardware capabilities and network connectivity can significantly impact the overall performance of FL systems. Optimizing for model accuracy could extend the training time due to the diverse and resource-constrained nature of edge devices while minimizing training time could compromise the model’s accuracy. Effective client selection thus becomes crucial to ensure that the training process is not only efficient but also capitalizes on the diverse data and computational capabilities of different devices. To this end, we propose FedPROM, a novel framework that tackles client selection in FL as a multi-criteria optimization problem. By leveraging the PROMETHEE method, FedPROM ranks clients based on their suitability for a given FL task, considering multiple criteria such as system resources, network conditions, and data quality. This approach allows FedPROM to dynamically select the most appropriate set of clients for each learning round, optimizing both model accuracy and training efficiency. Our evaluations on diverse datasets demonstrate that FedPROM outperforms several state-of-the-art FL client selection protocols in terms of convergence speed, and accuracy, highlighting the framework’s effectiveness and the importance of multi-criteria client selection in FL.}, 
  number={1}, 
  journal={Proceedings of the AAAI Symposium Series}, 
  author={Tahir, Mehreen and Ali, Muhammad Intizar}, 
  year={2024}, 
  month={May}, 
  pages={318-322} 
}

## Acknowledgments

This code is based on the [LEAF framework](https://github.com/TalwalkarLab/leaf).