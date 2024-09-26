"""
A collection of functions to check the health of the deployed model

Author: Derrick Lewis
Date: 2023-01-28
"""
import os
import json
import logging
import timeit
import subprocess
import pickle
import pandas as pd

logger = logging.getLogger(__name__)

# Function to get model predictions
def model_predictions(
    dff: pd.DataFrame,
    prod_deployment_path: str
        ) -> list:
    """
    read the deployed model and a test dataset, calculate predictions

    Parameters
    ---
    dff: pd.DataFrame
        A dataframe containing the test data
    prod_deployment_path: str
        Path to the directory containing the deployed model

    Returns
    ---
    list
        A list containing all predictions
    """
    logger.info("Getting model predictions")
    try:
        assert dff.shape[0] > 0
    except AssertionError as assert_error:
        logger.error("No data to make predictions on %s", assert_error)
        raise assert_error
    try:
        model = pickle.load(
            open(prod_deployment_path + '/trainedmodel.pkl', 'rb'))
        predictions = model.predict(dff.drop(['corporation', 'exited'],
                                    axis=1))
    except Exception as e:
        logger.error("Error getting predictions %s", e)
        raise e
    return predictions


# Function to get summary statistics
def dataframe_summary(
    output_folder_path: str
) -> dict:
    """
    Calculate summary statistics for numerical columns in the dataset

    Parameters
    ---
    output_folder_path: str
        Path to the directory containing the dataset

    Returns
    ---
    dict
        A dict containing all summary statistics
    """
    logger.info("Getting summary statistics")
    try:
        dff = pd.read_csv(output_folder_path + '/finaldata.csv')
    except Exception as e:
        logger.error("Error reading data %s", e)
        raise e
    try:
        # Divert from instructions to use dict instead of list
        summary_stats = dff.describe().to_dict()
    except Exception as e:
        logger.error("Error getting summary statistics %s", e)
        raise e
    return summary_stats


# Function to get summary statistics
def missing_data(
    output_folder_path: str
) -> dict:
    """
    Calculate missing for columns in the dataset

    Parameters
    ---
    output_folder_path: str
        Path to the directory containing the dataset

    Returns
    ---
    dict
        A dict containing all summary statistics
    """
    logger.info("Getting missing data")
    try:
        dff = pd.read_csv(output_folder_path + '/finaldata.csv')
    except Exception as e:
        logger.error("Error reading data %s", e)
        raise e
    try:
        # Divert from instructions to use dict instead of list
        nan_counts = dff.isna().sum().to_dict()
    except Exception as e:
        logger.error("Error getting summary statistics %s", e)
        raise e
    logger.info('Collected stats on missing data.')
    return nan_counts


# Function to get timings
def execution_time() -> dict:
    """
    Calculate execution time for ingestion and training

    Parameters
    ---
    None

    Returns
    ---
    dict
        A dict containing execution time for ingestion and training
    """
    logger.info("Getting execution time")
    timings = {}
    start = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start
    timings['ingestion_time'] = ingestion_time
    logger.info('Collected stats on ingestion time.')

    start = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start
    timings['training_time'] = training_time
    logger.info('Collected stats on training time.')

    return timings


# Function to check dependencies
def outdated_packages_list():
    """
    Get a list of outdated packages
    """
    logger.info("Getting outdated packages")
    outdated = subprocess.check_output(['pip', 'list', '-o'])
    print(outdated.decode())
    logger.info('Collected stats on outdated packages.')
    return outdated.decode()


if __name__ == '__main__':
    logging.basicConfig(
        filename="./logs/diagnostics.log",
        level=logger.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )

    # Path and environment variables are in config.json
    with open('config.json', 'r', encoding='utf8') as config_file:
        config = json.load(config_file)

    DEPLOY = os.path.join(config['prod_deployment_path'])
    OUTPUT = os.path.join(config['output_folder_path'])
    TEST_DATA = os.path.join(config['test_data_path'])
    INPUT_DATA = os.path.join(config['input_folder_path'])
    DFF = pd.read_csv(TEST_DATA + '/testdata.csv')

    model_predictions(DFF, DEPLOY)
    dataframe_summary(OUTPUT)
    execution_time()
    outdated_packages_list()
