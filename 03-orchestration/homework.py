from asyncio import tasks
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import datetime
import os, requests, pickle

from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):

    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):

    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date = None):
    if date == None:
        date = datetime.now()
    elif type(date) == datetime:
        date = date
    else:
        date_format = "%Y-%m-%d"
        date = datetime.strptime(date, date_format)

    train_date = date - pd.DateOffset(months = 2)
    val_date = date - pd.DateOffset(months = 1)

    train_file = f"fhv_tripdata_{train_date.year}-{train_date.month:02d}.parquet"
    val_file = f"fhv_tripdata_{val_date.year}-{val_date.month:02d}.parquet"

    if not os.path.exists('./data'):
        os.makedirs('./data')

    train_path = f"./data/{train_file}"
    val_path = f"./data/{val_file}"

    if not os.path.exists(train_path):
        response = requests.get(f"https://nyc-tlc.s3.amazonaws.com/trip+data/{train_file}")
        open(f"{train_path}", "wb").write(response.content)
    if not os.path.exists(val_path):
        response = requests.get(f"https://nyc-tlc.s3.amazonaws.com/trip+data/{val_file}")
        open(f"{val_path}", "wb").write(response.content)

    return train_path, val_path


@flow
def main(date = None):
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # Train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # Save the model.
    with open(f'./models/model-{date}.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)

    # Save the dictionary vectorizer.
    with open(f'./models/dv-{date}.bin', 'wb') as f_out:
        pickle.dump(dv, f_out)

DeploymentSpec(
    flow = main,
    name = "model-deployment",
    parameters = {'date': '2021-08-15'},
    schedule = CronSchedule(cron = "0 9 15 * *"),
    flow_runner = SubprocessFlowRunner(),
    tags = (['models'])
)