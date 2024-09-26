"""
Script to run the Flask app

Author: Derrick Lewis
Date: 2023-01-29
"""
import json
import os
import pandas as pd
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from flask import request
import plotly.io as pio
import pandas as pd
from diagnostics import model_predictions, dataframe_summary
from diagnostics import execution_time, missing_data, outdated_packages_list
import logging

api_log = logging.basicConfig(
    filename=os.path.join(os.getcwd() + "/logs/api.log"),
    level=logging.INFO,
    filemode='a',
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

external_stylesheets = [dbc.themes.BOOTSTRAP]
CELL_PADDING = 5
DATA_PADDING = 5
TABLE_PADDING = 1
FONTSIZE = 12

# ---------------------------------------------------------------------
# read in the data from the json file
try:
    with open(f'models/apireturns.json', 'r') as outfile:
        data_check = json.load(outfile)
    dff = pd.DataFrame(data_check['data_summary'])
except FileNotFoundError:
    api_log.warning('No json diagnostics file found')
    dff = pd.DataFrame()
    data_check = {
        'F1_score': {},
        'diagnostics': {
            'timings':{'ingestion_time': {},
                       'training_time': {}
            }
        }
    }
# ---------------------------------------------------------------------
# convert pip outdated stdout into a dataframe 
try:
    outdated = data_check['diagnostics']['outdated_pckgs']

    column_widths = [len(dashes) for dashes in outdated.split("\n")[1].split()]
    column_names = outdated.split("\n")[0].split()
    data = {key: list() for key in column_names}
    for line in outdated.split("\n")[2:]:
        start = 0
        end = 0
        for column_name, column_width in zip(column_names, column_widths):
            end += column_width + 1
            data[column_name].append(line[start:end])
            start = end
        
    outdated = pd.DataFrame(data)
except KeyError:
    outdated = pd.DataFrame()
    api_log.warning('No json diagnostics for outdated PIP packages found')
# ---------------------------------------------------------------------
# Read in the confusion matrix and AUC plotly figures from json files
with open(f'models/confusionmatrix.json', 'r') as outfile:
    cm = pio.from_json(json.load(outfile))
with open(f'models/auc.json', 'r') as outfile:
    auc = pio.from_json(json.load(outfile))

app = Dash(
    url_base_pathname="/dashboard/",
    external_stylesheets=external_stylesheets,
    meta_tags=[{
        "name": "viewport",
        "content": "width=device-width"
    }])
server = app.server

server.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['prod_deployment_path'])

prediction_model = None


@server.route("/prediction", methods=['GET', 'OPTIONS'])
def predict() -> str:
    """
    Calls the prediction function on the data in the config file
    under 'output_folder_path'
    """
    filepath = request.args.get('filepath')
    dff = pd.read_csv(filepath)
    preds = model_predictions(dff, model_path)
    return str(list(preds))


@server.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """Check the score of the deployed model"""
    with open(dataset_csv_path + '/latestscore.txt', 'r', encoding='utf8'
              ) as f1:
        latestscore = f1.read()
    return latestscore


@server.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary():
    """
    Calls the summary function on the data in the config file
    """
    sumamry_dict = dataframe_summary(dataset_csv_path)
    return sumamry_dict


@server.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnose():
    """
    Calls the remaingin diagnostics functions and returns the results
    """
    timings = execution_time()
    missing = missing_data(dataset_csv_path)
    outdated = outdated_packages_list()
    dianostics = {'timings': timings,
                  'missing_data': missing,
                  'outdated_pckgs': outdated}
    return dianostics


app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    dcc.Markdown(
                        """
                        # Dashboard
                        ---

                        Summary Statistics and Model Diagnostics for the latest dataset
                        """),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    dcc.Markdown(
                        """
                        ### Data Summary
                        
                        Below is a summary of the numerical data used to train the model.
                        
                        ---
                        """),
                    html.Br(),
                    dash_table.DataTable(
                        data=dff.reset_index().to_dict('records'),
                        columns=[
                            {"name": i, "id": i} for i in dff.reset_index().columns
                            ],
                        id='datatable-main',
                        virtualization=True,
                        page_action='none',
                        editable=True,
                        style_cell={
                            'whiteSpace': 'normal',
                            'height': '50px',
                            'fontSize': FONTSIZE,
                            'padding': CELL_PADDING,
                        },
                        style_header={
                            'backgroundColor': 'white',
                            'fontWeight': 'bold',
                            'font-family': 'plain'
                        },
                        style_cell_conditional=[
                            {'if':{'column_id':'index'}, 'fontWeight': 'bold'}
                            ],
                        style_data={
                            
                            'font-family': 'plain light',
                            'font-weight': 'light',
                            'color': 'grey',
                            'padding': DATA_PADDING,
                            'minWidth': 10,
                        },
                        style_table={
                            'height': '400px',
                            'minWidth': 10,
                            'padding': TABLE_PADDING
                        },
                        fixed_rows={'headers': True},
                        style_as_list_view=True,
                        ),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    dcc.Markdown(
                        """
                        ### Model Performance
                        The following graphs show the model performance on the latest dataset.
                        
                        ---
                        """
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(
                            dcc.Markdown(
                                f"""
                                ##### F1 Score:
                                    {data_check['F1_score']}
                                """), width={"size": 3, "offset": 1}),
                        dbc.Col(
                            dcc.Markdown(
                                f"""
                                ##### Ingestion Time:
                                    {data_check['diagnostics']['timings']['ingestion_time']}
                                """), width=3),
                        dbc.Col(
                            dcc.Markdown(
                                f"""
                                ##### Training Time:
                                    {data_check['diagnostics']['timings']['training_time']}
                                """), width=3)
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Confusion Matrix"),
                            dcc.Graph(
                                id='example-graph',
                                figure=cm
                                ),
                            ], width=6),
                        dbc.Col([
                            html.H5("AUC"),
                            dcc.Graph(
                                id='example-graph2',
                                figure=auc
                                )], width=6)
                        ]),
                    html.Br(),
                    dcc.Markdown(
                        """
                        ---
                        #### Outdated PIP Packages
                        The following table shows the packages that are out of date.
                        

                        """
                    ),
                    html.Br(),
                    dash_table.DataTable(
                        data=outdated.to_dict('records'),
                        columns=[
                            {"name": i, "id": i} for i in outdated.columns
                            ],
                        id='datatable-pip',
                        virtualization=True,
                        page_action='none',
                        editable=True,
                        style_cell={
                            'whiteSpace': 'normal',
                            'height': '50px',
                            'fontSize': FONTSIZE,
                            'padding': CELL_PADDING,
                        },
                        style_header={
                            'backgroundColor': 'white',
                            'fontWeight': 'bold',
                            'font-family': 'plain'
                        },
                        style_cell_conditional=[
                            {'if':{'column_id': 'Package'}, 
                            'fontWeight': 'bold',
                            'textAlign': 'left',
                            'font-family': 'plain bold'}
                            ],
                        style_data={
                            
                            'font-family': 'plain light',
                            'font-weight': 'light',
                            'color': 'grey',
                            'padding': DATA_PADDING,
                            'minWidth': 10,
                        },
                        style_table={
                            'height': '400px',
                            'minWidth': 10,
                            'padding': TABLE_PADDING
                        },
                        fixed_rows={'headers': True},
                        style_as_list_view=True,
                        ),
                ]
            ),
        )
    ],
)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
