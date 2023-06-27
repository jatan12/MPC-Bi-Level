# Bi-Level Optimization Augmented with Conditional Variational Autoencoder for Autonomous Driving in Dense Traffic

This repository contains the source code to reproduce the experiments in our IEEE CASE 2023 paper [Bi-Level Optimization Augmented with Conditional Variational Autoencoder for Autonomous Driving in Dense Traffic](https://arxiv.org/abs/2212.02224).

![CASE2023_Overview_page-0001](https://github.com/jatan12/MPC-Bi-Level/assets/38403732/8be3088a-bd03-4acb-b83b-6298b57417ce)

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
3. Download [CVAE Initialization Models](https://drive.google.com/file/d/1nOQq6EGnEdUtq1nuBOqsYGJwmw6M47dJ/view?usp=share_link) and extract the zip file to the weights directory. 

## Reproducing our main experimental results

![Eval](https://user-images.githubusercontent.com/38403732/209851177-1d56bef3-8e77-4452-a9d1-f1a5c80f2260.png)

### MPC-Bi-Level

```
python main_bilevel.py --density ${select} --four_lane ${True / False for two lane}
```

### MPC Baselines

To run a baseline {vanilla, grid, random, batch}:

```
python main_baseline.py --baseline ${select} --density ${select} --four_lane ${True / False for two lane}
```
**Note**: Default number of episodes is 50. To record / render the environment:

```
python main_baseline.py --episodes ${select} --record True --render True
```

## Learning Good Initialization Distribution

![CASE2023 Pipeline_page-0001](https://github.com/jatan12/MPC-Bi-Level/assets/38403732/ee172b4c-f4a3-4153-85f5-5aa4e03b974a)

1. Clone the [Deep Declarative Networks](https://arxiv.org/abs/1909.04866) repository:

```
cd MPC-Bi-Level
git clone https://github.com/anucvml/ddn.git
```

2. Download the [training dataset](https://drive.google.com/file/d/1tfXn11uwGwqS23hOH1oKlfIVJya9-hvE/view?usp=share_link) and extract the zip file to the dataset directory. 

3. The training example is shown in the [Jupyter Notebook](https://github.com/jatan12/MPC-Bi-Level/blob/main/Beta%20cVAE%20DDN%20Training.ipynb) and can also be viewed using [Notebook Viewer](https://nbviewer.org/github/jatan12/MPC-Bi-Level/blob/main/Beta%20cVAE%20DDN%20Training.ipynb).

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
