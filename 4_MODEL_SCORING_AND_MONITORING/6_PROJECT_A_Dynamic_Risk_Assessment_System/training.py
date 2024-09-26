"""
This script trains a logistic regression model on the data in finaldata.csv
and saves the trained model in a file called trainedmodel.pkl

Author: Derrick Lewis
Date: 2023-01-28
"""
import os
import pickle
import json
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def train_model(data_pth: str, model_pth: str) -> None:
    """
    Trains a logistic regression model on the data in finaldata.csv

    Parameters
    ---
    data_path: str
        Path to the directory containing finaldata.csv
    model_path: str
        Path to the directory where the trained model will be saved

    Returns
    ---
    None
    """
    logger.info("Training model")
    try:
        dff = pd.read_csv(data_pth + '/finaldata.csv')
        y = dff.pop('exited')
        X = dff.drop('corporation', axis=1)
    except Exception as e:
        logger.error("Error reading data %s", e)
        raise e
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # fit the logistic regression to your data
    logger.info('Fitting model')
    model = model.fit(X, y)

    # write the trained model in a file called trainedmodel.pkl
    logger.info('Writing model to file')
    try:
        with open(model_pth + '/trainedmodel.pkl', 'wb') as mod:
            pickle.dump(model, mod)
    except FileNotFoundError as fnf:
        logger.error(
            "File %s/trainedmodel.pkl not found, check config.json: %s",
            model_pth, fnf)
        raise fnf
    except Exception as e:
        logger.error("Error writing model to file %s", e)
        raise e
    logger.info('Model training complete')
    return None


if __name__ == "__main__":
    logging.basicConfig(
        filename="./logs/training.log",
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )
    # Load config.json and get path variables
    with open('config.json', 'r', encoding='utf8') as file:
        config = json.load(file)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])

    train_model(dataset_csv_path, model_path)
