# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

- This dataset contains data about client profiles at a bank, and the problem statement is to predict whether a client profile will subscribe (yes/no) to a bank term deposit.

- The best performing model was VotingEnsemble (accuracy 0.9189)

## Scikit-learn Pipeline

- **Pipeline architecture**:
    - Data: Clean, one-hot encode, split train-test (80:20)
    - Hyperparameter tuning: inverse of regularization strength C, maximum iteration max_iter
    - Classification algorithm: Logistic Regression
- **Benefits of RandomParameterSampling**: 
    - Flexible to handle both discrete (max_iter) and continuous (C) param values
    - Save computational cost while still randomly exploring diverse values within the range
    - Help find best combination of parameters for highest accuracy
- **Benefits of BanditPolicy**:
    - Terminates poorly performing runs (compared to best run) early, saving computational resources and time.

## AutoML

AutoML generated over 30 modesl, with the best performing one to be VotingEnsemble. 

Some parameters are task="classification", primary_metric="accuracy", n_cross_validations=4.

## Pipeline comparison

I used the same data preprocessing steps before fitting training data to both pipelines. AutoML outperforms the traditional Logistic Regression with HyperDrive (LR 0.9083 vs VE 0.9189) by a small margin. The data is clean, with no missing value, and the problem is quite simple so the accuracy gap between the simple Logistic Regression and using AutoML is small, but for more complex problem I think the gap will be much greater.

## Future work

I would try to experiment maximizing other metrics that takes into account the imbalance in the labels, such as F1, AUC.

I may also try to run the AutoML for longer to see if it finds a better model with higher metric.
