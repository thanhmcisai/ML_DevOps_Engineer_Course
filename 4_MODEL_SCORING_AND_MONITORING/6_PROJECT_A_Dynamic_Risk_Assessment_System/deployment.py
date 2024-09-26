"""
This script copies the latest pickle file, the latestscore.txt value, and the
ingestfiles.txt file into the deployment directory

Author: Derrick Lewis
Date: 2023-01-28
"""
import os
import json
import logging

logger = logging.getLogger(__name__)


# Function for deployment
def store_model_into_pickle(
    output_folder_path: str,
    output_model_path: str,
    prod_deployment_path: str
        ) -> None:
    """
    Copy the latest pickle file, the latestscore.txt value, and the
    ingestfiles.txt file into the deployment directory

    Parameters
    ---
    output_folder_path: str
        Path to the directory containing the
        latestscore.txt value and the ingestfiles.json file
    output_model_path: str
        path to the latest model pickle file
    prod_deployment_path: str
        Path to the directory where the latest pickle file, the latestscore.txt
        value, and the ingestfiles.txt file will be copied

    Returns
    ---
    None
    """
    logger.info("Storing model into pickle")
    try:
        # Copy the latest pickle file
        os.system("cp " + output_model_path + "/" + "trainedmodel.pkl "
                  + prod_deployment_path + "/" + "trainedmodel.pkl")
        # Copy the latestscore.txt value
        os.system("cp " + output_folder_path + "/latestscore.txt "
                  + prod_deployment_path + "/latestscore.txt")
        # Copy the ingestfiles.txt file
        os.system("cp " + output_folder_path + "/ingestedfiles.json "
                  + prod_deployment_path + "/ingestedfiles.json")
    except FileNotFoundError as fnf:
        logger.error(
            "File not found, check config.json: %s", fnf)
        raise fnf
    except Exception as e:
        logger.error("Error in storing model into pickle")
        logger.error(e)
        raise e
    logger.info(
        "Model, latest score, and latest file list stored into \
prod_deployment_path: %s", prod_deployment_path)
    return None


if __name__ == "__main__":
    logging.basicConfig(
        filename="./logs/deployment.log",
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )
    # Path variables are stored in config.json
    with open('config.json', 'r', encoding='utf8') as f:
        config = json.load(f)

    OUTPUT = os.path.join(config['output_folder_path'])
    DEPLOYMENT = os.path.join(config['prod_deployment_path'])
    MODEL = os.path.join(config['output_model_path'])

    store_model_into_pickle(OUTPUT, DEPLOYMENT, MODEL)
