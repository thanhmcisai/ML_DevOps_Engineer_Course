"""
This script should load a trained model, load test data, and calculate an
F1 score for the model relative to the test data it should write the result
to the latestscore.txt file

Author: Derrick Lewis
Date: 2023-01-28
"""
import json
import os
import pickle
import logging
import pandas as pd
from sklearn import metrics

logger = logging.getLogger(__name__)

# Function for model scoring
def score_model(
    output_folder_path: str,
    test_data_path: str,
    output_model_path: str
        ) -> float:
    """
    This function should take a trained model, load test data, and calculate an
    F1 score for the model relative to the test data it should write the result
    to the latestscore.txt file

    Parameters
    ---
    output_folder_path: str
        Path to the dataset csv file
    test_data_path: str
        Path to the test data csv file including the filename
        ie. '/data/testdata.csv'
    output_model_path: str
        Path to the model pickle file

    Returns
    ---
    F1 score: float

    """
    logger.info("Scoring model")
    try:
        dff = pd.read_csv(test_data_path)

        y = dff.pop('exited')
        X = dff.drop('corporation', axis=1)
    except FileNotFoundError as fnf:
        logger.error("File %s/testdata.csv not found, check config.json: %s",
                      test_data_path, fnf)
        raise fnf
    except Exception as e:
        logger.error("Error reading data %s", e)
        raise e
    model = pickle.load(open(output_model_path + '/trainedmodel.pkl', 'rb'))

    preds = model.predict(X)

    f1 = metrics.f1_score(y, preds)

    with open(output_folder_path + '/latestscore.txt', 'w', encoding='utf8'
              ) as file:
        file.write(str(f1))
    return f1


if __name__ == '__main__':
    logging.basicConfig(
        filename="./logs/scoring.log",
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )
    # Path variables are stored in config.json
    with open('config.json', 'r', encoding='utf8') as f:
        config = json.load(f)

    OUTPUT = os.path.join(config['output_folder_path'])
    TEST_DATA = os.path.join(config['test_data_path'], 'testdata.csv')
    MODEL = os.path.join(config['output_model_path'])

    score_model(OUTPUT, TEST_DATA, MODEL)
