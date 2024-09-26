"""
This script merges multiple csv files into one dataframe, de-duplicates records
and writes it to a csv file

Data is read from the input folder and written to the output folder

Authort: Derrick Lewis
Date: 2023-01-28
"""
import os
import glob
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)


# Function for data ingestion
def merge_multiple_dataframe(
    input_folder_path: str,
    output_folder_path: str,
    ext: str
        ) -> pd.DataFrame:
    """Merges multiple csv files into one dataframe, de-duplicates records
    and writes it to a csv file

    Parameters
    ---
    input_folder_path: str
        Path to the folder containing the csv files

    output_folder_path: str
        Path to the folder where the output file will be written

    ext: str
        Extension of the files to be read ie. csv, txt, etc.

    Returns
    ---
    final_dataframe: pd.DataFrame

    """

    final_dataframe = pd.DataFrame(
        columns=["corporation", "lastmonth_activity", "lastyear_activity",
                 "number_of_employees", "exited"])

    # check for datasets, compile them together, and write to an output file
    # check if the input folder is empty
    result = glob.glob(f'./{input_folder_path}/*.{ext}')
    try:
        assert len(result) > 0
    except AssertionError as e:
        logger.error("Input folder %s does not contain .csv files",
                     input_folder_path)
        raise e
    logger.info("Found (%i) files in the input folder", len(result))
    for filename in result:
        logger.info("Reading file: %s", filename)
        try:
            df = pd.read_csv(filename)
        except Exception as e:
            logger.error("Could not read file: %s due to %s", filename, e)
            raise e
        logger.info("Successfully read file: %s", filename)
        final_dataframe = pd.concat([final_dataframe, df], axis=0)
    try:
        start_len = len(final_dataframe)
        final_dataframe = final_dataframe.drop_duplicates()
        logger.info("Removed %i duplicates", start_len - len(final_dataframe))
    except Exception as e:
        logger.error("Could not remove duplicates due to %s", e)
        raise e
    logger.info("Writing output file to %s", output_folder_path)
    final_dataframe.to_csv("./" + output_folder_path + '/finaldata.csv',
                           index=False)
    logger.info("Successfully wrote output file to %s", output_folder_path)

    # Save ingested file names as a python json file
    logger.info('Saving ingested file names as json file')
    with open('./' + output_folder_path + '/ingestedfiles.json', 'w',
              encoding='utf8') as file:
        json.dump([os.path.basename(file) for file in result], file)
    return final_dataframe


if __name__ == '__main__':
    logging.basicConfig(
        filename="./logs/data_ingestion.log",
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )
    # Load config.json and get input and output paths
    with open('config.json', 'r', encoding="utf8") as f:
        config = json.load(f)

    INPUT_PATH = config['input_folder_path']
    OUTPUT_PATH = config['output_folder_path']
    EXT = 'csv'

    merge_multiple_dataframe(INPUT_PATH, OUTPUT_PATH, EXT)
