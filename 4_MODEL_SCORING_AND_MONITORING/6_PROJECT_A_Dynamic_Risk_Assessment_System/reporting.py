"""
This script is used to generate a confusion matrix using the test data
and the deployed model. The confusion matrix is saved to the workspace

Author: Derrick Lewis
Date: 2023-01-29
"""
import json
import os
import logging
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from diagnostics import model_predictions

logger = logging.getLogger(__name__)


def create_plots(
    dff: pd.DataFrame,
    output_folder_path: str,
    prod_deployment_path: str
        ) -> None:
    """calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    Parameters
    ---
    dff: pd.DataFrame
        A dataframe containing the test data
    output_folder_path: str
        Path to the directory containing the output folder
    prod_deployment_path: str
        Path to the directory containing the deployed model

    Returns
    ---
    fig: plotly.graph_objects.Figure
        A plotly figure containing the confusion matrix

    """
    logging.info('Generating confusion matrix')
    try:
        preds = model_predictions(
            dff, prod_deployment_path)
    except Exception as e:
        logging.error('Error loading model predictions: %s', e)
        raise e
    logging.info('loaded model predictions')

    logging.info('Generating confusion matrix in Plotly')
    print('dff exited type: ', dff['exited'].dtype)
    print('preds type: ', preds.dtype)

    y_test = dff['exited'].astype(int)

    fig_cm = go.Figure()
    fig_cm.add_trace(go.Heatmap(
        z=confusion_matrix(y_test, preds),
        x=['Predicted Not Exited', 'Predicted Exited'],
        y=['Actual Not Exited', 'Actual Exited'],
        text=confusion_matrix(y_test, preds),
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale='YlGnBu')
        )
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual'
        )

    logging.info('Saving confusion matrix to %s', output_folder_path)
    fig_cm.write_image(
        os.path.join(output_folder_path, 'confusionmatrix.png'),
        format='png',
        width=800, height=600)

    with open(f'{output_folder_path}/confusionmatrix.json', 'w') as outfile:
        json.dump(fig_cm.to_json(), outfile)

    logging.info('Finished generating confusion matrix')
    logging.info('Generating ROC Curve')
    fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)
    fig_auc = px.area(
            x=fpr,
            y=tpr,
            title=f'Logistic Regression<br>\
                <sub>ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate',
                        y='True Positive Rate'),
            width=600,
            height=800
    )
    fig_auc.add_shape(
            type='line',
            line=dict(dash='dash'),
            x0=0,
            x1=1,
            y0=0,
            y1=1
    )

    fig_auc.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_auc.update_xaxes(constrain='domain')
    logging.info('Created ROC Curve')

    # TODO: write json format for dashboard
    fig_auc.write_image(
        os.path.join(output_folder_path, 'auc.png'),
        format='png',
        width=800, height=600)

    with open(f'{output_folder_path}/auc.json', 'w') as outfile:
        json.dump(fig_auc.to_json(), outfile)
    logging.info('Saved ROC/AUC to %s', output_folder_path)
    return fig_cm, fig_auc


if __name__ == '__main__':
    logging.basicConfig(
        filename="./logs/reporting.log",
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )
    # Load config.json and get path variables
    with open('config.json', 'r', encoding="utf8") as f:
        config = json.load(f)

    OUTPUT = os.path.join(config['output_folder_path'])
    MODEL = os.path.join(config['prod_deployment_path'])
    TEST_DATA = os.path.join(config['test_data_path'])
    DFF = pd.read_csv(TEST_DATA + '/testdata.csv')

    FIG, FIG2 = create_plots(DFF, OUTPUT, MODEL)
    FIG.show()
    FIG2.show()
