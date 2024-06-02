# Iris Pipeline

This is a prototypical Machine Learning pipeline for Iris dataset classifier training.
It uses MLFlow as a model registry to manage and track experiments and models.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
  - [Clone the Repository](#clone-the-repository)
  - [Create Conda Environment](#create-conda-environment)
  - [Install Dependencies](#install-dependencies)
  - [Run MLFlow](#run-mlflow)
- [Usage](#usage)
  - [Running the Pipeline](#running-the-pipeline)
  - [Tracking Experiments with MLFlow](#tracking-experiments-with-mlflow)
- [Acknowledgments](#acknowledgments)

## Overview

This project is designed to demonstrate a machine learning pipeline using the Iris dataset. It incorporates MLFlow for tracking experiments, logging model parameters, metrics, and artifacts, and managing model versions.


# Setup

Clone the repository
```bash
git clone https://github.com/your-username/iris_pipeline.git
cd iris_pipeline
conda create -n iris-pipeline python=3.10
conda activate iris-pipeline
# install dependencies
pip install -r requirements.txt
```

# Usage
In order to `simulate` an orchestration, please run pipeline.py. It will execute each step in src/steps
```bash
python src/pipeline.py
````

To assess the outputs, run the MLFlow UI

```bash
mlflow ui -p 5010
```

got to 127.0.0.1:5010 and navigate through the experiment.
