# Optimizing Reachability in Graph-based Recommender Systems

This repository contains the code and data necessary to reproduce the results from our study titled **"Optimizing Reachability in Graph-based Recommender Systems."** Our work focuses on enhancing item accessibility in recommendation graphs to ensure a fairer and more diverse recommendation experience.

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Contact](#contact)

## Introduction

While accuracy has long been prioritized as the primary metric for Recommender Systems (RSs), it is increasingly accepted that the system's overall quality is not solely determined by this factor. Reachability, the ease with which users can navigate the whole content catalog through recommendations, emerges as a pivotal yet under-explored concept: not only  it ensures a smooth experience for users, but it also provides more equitable exposure for the items, avoiding that only a small fraction of popular items get the bulk of the attention.
Despite its importance, the few existing studies analyze reachability without attempting a proper optimization. 

In this paper, we study the problem of optimizing the overall reachability of a RS while maintaining high-quality recommendations. We model a user browsing session as a random walk on a recommendation graph, where the links and the transition probabilities are defined based on the relevance score of the recommendation list that the user gets at every step. In this setting, reachability is modeled as the expected length of a path to reach a given item. We introduce two optimization problems, one discrete and one continuous, and characterize their theoretical properties. We then devise two algorithms that outperform non-trivial baseline methods in enhancing reachability while maintaining a high normalized Discounted Cumulative Gain (nDCG) score. Our experimental results show that, in some settings, our methods are able to improve the reachability metric by 80% while only compromising nDCG by 5%. Moreover, our empirical analysis shows that optimizing for reachability provides positive effects also on other prevalent *beyond-accuracy* metrics.

## Repository Structure

- **data/**: Contains the scripts to pre-process tha data used in our experiments.
- **compute_SLSQP_rewirings-reweighting.py**: Implements the SLSQP optimization method for reweighting edges to improve reachability.
- **compute_baseline1_greedy.py**: Implements a greedy baseline method.
- **compute_baseline2_random.py**: Implements a random baseline method.
- **compute_baseline3_diversify.py**: Implements a diversification baseline method.
- **compute_greedy_rewirings.py**: Implements the greedy rewiring method, BGS in our paper.
- **utils.py**: Utility functions that support the main scripts.
- **requirements.txt**: Lists the Python packages required to run the code.

## Installation

1. **Create a virtual environment:**
   ```bash
   python3 -m venv reachability

2. **Activate virtual environment:**
   ```bash
   source reachability/bin/activate
   
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

## Data Preparation
Our experiments utilize the NELAGT-2022 dataset. Please download the dataset from the [NELAGT-2022 original source](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AMCV2H) and place it in the ```data/nelagt``` directory of this repository. After placing the dataset, run the preprocessing notebook to prepare the data for analysis: ```data-preprocessing.ipynb```. This will create and store the relevancy matrix and transition probability matrices.

## Usage
Once the environment is set up and the data is prepared, you can run the scripts corresponding to the different methods and baselines.

1. **SLSQP**: Uses Sequential Least Squares Quadratic Programming to update all edge probabilities simultaneously
   ```bash
   python3 compute_SLSQP_rewirings-reweighting.py

2. **BGS**: Implements a batch greedy search to find the optimal rewiring at each iteration
   ```bash
   python3 compute_greedy_rewirings.py
3. **Greedy Baseline**: Serves as an ablation baseline of the BGS
   ```bash
   python3 compute_baseline1_greedy.py

4. **Random Baseline Method**: Implement completely random rewirings
   ```bash
   python3 compute_baseline2_random.py

5. **Diversification Baseline Method**: Implements rewirings the maximize diversification across recommendation lists
   ```bash
   python3 compute_greedy_rewirings.py

## Contact
For questions or further information, please contact:
- **Alex Martinez**
- alexmartinez.m97@gmail.com
