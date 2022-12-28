# Bi-Level Optimization Augmented with Conditional Variational Autoencoder for Autonomous Driving in Dense Traffic

This repository contains the source code to reproduce the experiments in our RAL submission [Bi-Level Optimization Augmented with Conditional Variational Autoencoder for Autonomous Driving in Dense Traffic](https://arxiv.org/abs/2212.02224).

![Pipeline](https://user-images.githubusercontent.com/38403732/209846154-865812a0-e1c4-474c-ba78-8dab36c4ac21.png)

## Getting Started

1. Clone this repository:

```
git clone https://github.com/jatan12/MPC-Bi-Level.git
cd MPC-Bi-Level
```
2. Create a conda environment and install the dependencies:

```
conda create -n bilevel python=3.8
conda activate bilevel
pip install -r requirements.txt
```
3. Download [CVAE Initialization Models](https://drive.google.com/file/d/19JP3UuHlVCR6XZB9mB5krZNmJXttPvvL/view?usp=share_link) and extract the zip file to the weights directory. 

## Reproducing our main experimental results

![Eval](https://user-images.githubusercontent.com/38403732/209851177-1d56bef3-8e77-4452-a9d1-f1a5c80f2260.png)

### MPC Baselines

```
python baseline_vanilla.py --episodes 50 --density 3.0 --four_lane True --record False --render False
```

## Learning Good Initialization Distribution

![CVAE](https://user-images.githubusercontent.com/38403732/209850972-7171caa7-6aff-48ab-aa32-dbdf5f5a1ffc.png)

## Citation

If you found this repository useful, please consider citing our work:

```
@misc{https://doi.org/10.48550/arxiv.2212.02224,
  doi = {10.48550/ARXIV.2212.02224},
  url = {https://arxiv.org/abs/2212.02224},
  author = {Singh, Arun Kumar and Shrestha, Jatan and Albarella, Nicola},
  keywords = {Robotics (cs.RO), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Bi-Level Optimization Augmented with Conditional Variational Autoencoder for Autonomous Driving in Dense Traffic},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
