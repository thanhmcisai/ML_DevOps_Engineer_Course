"""
Script to call the API endpoints of the model and store the responses
"""
import os
import json
import requests


def get_data(output_folder_path: str, url: str) -> None:
    """
    Call each API endpoint and store the responses

    Parameters
    ---
    output_folder_path: str
        Path to the folder where the responses will be stored

    Returns
    ---
    None
    """
    response1 = requests.get(
        url + "prediction",
        params={'filepath': 'testdata/testdata.csv'},
        timeout=10).json()
    response2 = requests.get(url + "scoring", timeout=10).content.decode()
    response3 = requests.get(url + "summarystats", timeout=10).json()
    response4 = requests.get(url + "diagnostics", timeout=10).json()

    apicalls = {'prediction': response1, 'F1_score': response2,
                'data_summary': response3, 'diagnostics': response4}
    responses = apicalls

    with open(output_folder_path + '/apireturns.json', 'w', encoding='utf8'
              ) as resp:
        json.dump(responses, resp, indent=4)
    return None


if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf8') as f:
        config = json.load(f)

    OUTPUT_PATH = os.path.join(config['output_folder_path'])
    URL = config['url']

    get_data(OUTPUT_PATH, URL)
