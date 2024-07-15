# Exploring the Connection Between Predictive Processing and Mental Wellbeing Through Reinforcement Learning

## Bachelor Thesis

- **Title**: Exploring the Connection Between Predictive Processing and Mental Wellbeing Through Reinforcement Learning
- **Author**: Manuel Arnau Fernández
- **Tutor**: Jordi González Sabaté, Antonio Lozano Bagen
- **Institution**: Universitat Autònoma de Barcelona (UAB)

## Overview

This project is focused on the field of psychology, exploring new methodologies to obtain insights into mental wellbeing using reinforcement learning (RL). The aim is to leverage RL techniques to analyze decision-making processes and their connection to predictive processing. This innovative approach has the potential to provide new perspectives and improve our understanding of mental health, which could lead to more effective clinical practices.

## Collaboration and Dataset

This is a collaborative project with the Cognitive and Affective Science Laboratory at the Universitat Autònoma de Barcelona (UAB). The dataset used in this study has been provided by members of the department and contains sensitive and private information. Due to privacy concerns, the dataset cannot be shared publicly.

## Repository Structure

The repository is organized into several main components, each focusing on different aspects of the project:

1. **rl_predictiveprocessing.ipynb**:
   - Implements Behavioral Cloning (BC) to analyze decision-making processes.
   
2. **BIRL_PredictiveProcessing.ipynb** and **Updated_BIRL_PredictiveProcessing.py**:
   - Implement Inverse Reinforcement Learning (IRL), particularly Bayesian IRL with several modifications, to understand the dynamic evolution of decision-making.

3. **Analysis_RL.ipynb**:
   - Conducts statistical analysis using ANOVA and machine learning analysis using XGBoost and RandomForest to find relationships between the variables.

## Installation

To run the code in this repository, you need to have Python installed along with the following packages:
- `torch`
- `pandas`
- `numpy`
- `time`
- `openpyxl` (for handling Excel files)

## Methodology

The methodology section describes the approach used to explore the relationship between predictive processing and mental wellbeing data. It includes feature engineering, Imitation Learning models, and the optimization process using Bayesian Inverse Reinforcement Learning.

- **Feature Engineering**: Extract meaningful features from the predictive processing dataset.
- **Imitation Learning Models**: Implement Behavioral Cloning and Inverse Reinforcement Learning to learn from expert trajectories.
- **Optimization**: Use BIRL to infer the hidden reward functions that drive decision-making.

### Feature Engineering

The first step involves extracting meaningful features from the predictive processing dataset. This helps in understanding the relationship between the variables and the decision-making processes.

### Imitation Learning Models

1. **Behavioral Cloning (BC)**:
    - Implemented in `RL_PredictiveProcessing.ipynb`.
    - A supervised learning approach that mimics the decision-making process of experts by learning a policy from the state-action pairs.

2. **Inverse Reinforcement Learning (IRL)**:
    - Implemented in `BIRL_PredictiveProcessing.ipynb` and `Updated_BIRL_PredictiveProcessing.py`.
    - A Bayesian IRL approach with several modifications to capture the dynamic evolution of decision-making over time.

### Statistical Analysis

    - Implemented in `Analysis_RL.ipynb`.
    - Conducts statistical analysis using ANOVA to determine the significance of the relationships between the extracted features and psychological scales.
    - Uses machine learning algorithms like XGBoost and RandomForest to find and validate relationships between variables.


## Contributors

- **Manuel Arnau Fernández** - Implementation and analysis
- **Cognitive and Affective Science Laboratory, UAB** - Provided dataset and the objective to solve


---


