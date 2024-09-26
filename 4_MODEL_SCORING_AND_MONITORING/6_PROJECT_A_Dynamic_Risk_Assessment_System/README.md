# ML_DevOps_ML_Model_Scoring_and_Monitoring

- Project 4 for Udacity Machine Learning DevOps Engineer Nanodegree

The purpose of this project is to imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Creating and deploying the model isn't the ultimate purpose of this project, though. The industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. This project sets up regular monitoring of the model to ensure that it remains accurate and up-to-date. The project processes the scripts to re-train, re-deploy, monitor, and report on your ML model, so that the company can get risk assessments that are as accurate as possible and minimize client attrition.

Components of this project
1. **Data ingestion**. Automatically checks a database for new data that can be used for model training. Compiles all training data to a training dataset and saves it to persistent storage. Writes metrics related to the completed data ingestion tasks to persistent storage.
2. **Training, scoring, and deploying**. Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.
3. **Diagnostics**. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.
4. **Reporting**. Automatically generate plots and documents that report on model metrics. Script builds API endpoint and visual dashboard that return model predictions and metrics.
5. **Process Automation**. Project sets up a cron job that automatically run all previous steps at regular intervals.


## To set up environment

The file `environment.yml` in this repository can be used to create install the libraries I've been using to run these test scripts. With conda installed on your machine: 

```
conda env create -f environment.yml
```

Then 

```
conda activate proj4

```


## Files and data description
Overview of the files and data present in the root directory. 

### Data
Data for this project was supplied by Udacity is very simple and small as this project is focused on the automation 
more than the challenge of modeling. 

There are several files found in the project with the following formats. They need to be de-duplicated and combined into 
a single training file before training the model. 

- `./sourcedata/dataset4.csv`: 
    - `corporation`: str - Name of the target client to be predicted
    - `lastmonth_activity`: int -  volume of activity last month
    - `lastyear_activity`: int - volume of activity last year 
    - `number_of_employees`: int - number of employees
    - `exited`: int - target classification of churn for the model to predict 

### config
- The config files sets all parameters of the project to be run. It is in a `.json` format and should contain `key:value` pairs in a string format. 
- The following values should be present to train and deploy this project

```
"input_folder_path": "sourcedata",
"output_folder_path": "ingesteddata",
"test_data_path": "testdata",
"output_model_path": "models",
"prod_deployment_path": "production_deployment",
"input_file_extension": "csv",
"url": "http://127.0.0.1:8000/"
```

### Directories
- [practicedata](practicedata): This folder that contains some data to use in development of the main functions. 
- [ingesteddata](ingesteddata): This is a directory that will contain the compiled datasets after the ingestion script.
- [sourcedata](sourcedata): This is the primary directory that the project will look for new data files. In a larger production setting this could be a cloud storage bucket with regulary deposited data. 
- [testdata](testdata): Directory containing test data for development
- [models](models): This directory contains ML models that are created during production
- [production_deployment](production_deployment): The directory that contains the final deployed models.
- 
- [/logs](/logs/): Directory containing log files of either training or testing of the project. 
- [/models](/models/): Serialized models after training. 
- [/data](/data/): directory to store data file to be used in model training.

### Files

- [**training.py**](training.py): a Python script meant to train an ML model
- [**scoring.py**](scoring.py): a Python script meant to score an ML model
- [**deployment.py**](deployment.py): a Python script meant to deploy a trained ML model
- [**ingestion.py**](ingestion.py): a Python script meant to ingest new data
- [**diagnostics.py**](diagnostics.py): a Python script meant to measure model and data diagnostics
- [**reporting.py**](reporting.py): a Python script meant to generate reports about model metrics
- [**app.py**](app.py): a Python script meant to contain API endpoints
- [**wsgi.py**](wsgi.py): a Python script to help with API deployment
- [**apicalls.py**](apicalls.py): a Python script meant to call your API endpoints
- [**fullprocess.py**](fullprocess.py): a script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed
- [**cronjob.txt**](cronjob.txt): a text file with the configurations to set up the reoccurring process to check for new data and model drift.
  
## Running Files

1. The model must first have the API being served on the url in the `config.json` file. To start the API locally you can
   simply run `python app.py` from your current working directory.

2. Then you can run the full process script which checks for new data, determines if the model has drifted from the 
   new data, and retrains the model if so. To do this, simply run `python fulprocess.py` from the command line of this 
   directory. Be sure the API is running before hand. 

3. The intended set up is to use a Crontab Job in the linux system to automate the process. 
   The Udacity course brief requested a reoccuring check every 10 min 
   
   To setup the orccuring process. In the command line of your workspace, run the following command: `service cron start`
   Open your workspace's crontab file by running `crontab -e` in your workspace's command line. Edit the file by appending 
   the lines from [**cronjob.txt**](cronjob.txt) replacing `/home/derricklewis/` with your home directory or `~` if possible.

The structure of the script:

![diagram of model steps](images/fullprocess.jpg)
## Author 
-   **Udacity** (Main functions and .ipynb)  [Udacity Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)
-   **Derrick Lewis** (Documenation, logging, tests)  [Portfolio Site](https://www.derrickjameslewis.com) - [linkedin](https://www.linkedin.com/in/derrickjlewis/)

## Dashboard

With the app.py file being served, a visual dashboard is available at the url + `/dashboard/`. This dashboard will present
a front end visual of the model performance and current data summary used to train the most recent model.

![Full page screenshot of project dashboard](images/Screenshot%202023-02-11%20at%2011-40-29%20Updating.png)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

I would like to thank [Udacity](https://eu.udacity.com/) for this amazing project
