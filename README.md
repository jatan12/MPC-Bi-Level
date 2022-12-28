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
3. Download [CVAE Initialization Models](https://drive.google.com/file/d/19JP3UuHlVCR6XZB9mB5krZNmJXttPvvL/view?usp=share_link) and extract the zip file to the Weights directory. 

## Reproducing our main experimental results

![Full Scenarios](https://user-images.githubusercontent.com/38403732/209849602-bda62e58-2949-4873-bb44-9095eacf851d.png)

```
python main.py 
```

## Learning Good Initialization Distribution

![Training](https://user-images.githubusercontent.com/38403732/209849729-7c21c549-b118-4dad-abda-f375e22fbba4.png)

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
