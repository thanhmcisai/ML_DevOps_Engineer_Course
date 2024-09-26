"""
This script is the main script that runs the entire process.
It will be scheduled to run on a regular basis.

Author: Derrick Lewis
Date: 2023-01-29
"""
import sys
import os
import logging
import json
import glob
from ingestion import merge_multiple_dataframe
from training import train_model
from scoring import score_model
from deployment import store_model_into_pickle
from reporting import create_plots
from apicalls import get_data

# Set up working directory of this file so that cron can run it from anywhere
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

logging.basicConfig(
    filename=os.path.join(os.getcwd() + "/logs/fullprocess.log"),
    level=logging.INFO,
    filemode='a',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

with open('config.json', 'r', encoding='utf8') as file:
    config = json.load(file)

DEPLOY_PATH = os.path.join(config['prod_deployment_path'])
INPUT_PATH = os.path.join(config['input_folder_path'])
OUTPUT_PATH = os.path.join(config['output_folder_path'])
OUTPUT_MODEL_PATH = os.path.join(config['output_model_path'])
TEST_DATA_PATH = os.path.join(config['test_data_path'])
EXT = config['input_file_extension']
URL = config['url']


# First, read ingestedfiles.json
with open(
    os.path.join(DEPLOY_PATH, 'ingestedfiles.json'),
    'r',
    encoding="utf8"
          ) as file:
    ingestedfiles = json.load(file)

# Second, determine whether the source data folder has new files
newfiles = glob.glob(f'./{INPUT_PATH}/*.{EXT}')

if len(newfiles) > 0:
    newfiles = [os.path.basename(file) for file in newfiles]
    newfiles = [file for file in newfiles if file not in ingestedfiles]
    if len(newfiles) > 0:
        logging.info("Found new files: %s", newfiles)
        df = merge_multiple_dataframe(INPUT_PATH, OUTPUT_PATH, EXT)
        OUTPUT_DATA_PATH = os.path.join(
            config['output_folder_path'],
            'finaldata.csv')
    else:
        logging.info("No new files found")
        sys.exit()
else:
    logging.info("Zero files in the source data folder with extension %s",
                 EXT)
    sys.exit()

# Read in the score from the deployed model
with open(
    os.path.join(DEPLOY_PATH, 'latestscore.txt'),
    'r',
    encoding="utf8"
          ) as f1:
    og_f1_score = float(f1.read())

# train the model on the new data and get the f1 score
new_f1_score = score_model(OUTPUT_PATH, OUTPUT_DATA_PATH, DEPLOY_PATH)

if new_f1_score < og_f1_score:
    logging.info("Model drift detected")
    model_drift = True

else:
    logging.info("No model drift detected")
    model_drift = False
    sys.exit()

# Re-deployment
if model_drift is True:
    train_model(OUTPUT_PATH, OUTPUT_MODEL_PATH)
    store_model_into_pickle(OUTPUT_PATH, OUTPUT_MODEL_PATH, DEPLOY_PATH)
    get_data(OUTPUT_MODEL_PATH, URL)
    create_plots(df, OUTPUT_MODEL_PATH, DEPLOY_PATH)
