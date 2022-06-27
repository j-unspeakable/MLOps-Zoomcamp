# Import libraries.
import argparse
import logging
import pickle
import pandas as pd
import logging
import os

# Configure the logger.
logging.basicConfig(level=logging.INFO)

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def ride_predictions(year, month):

    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    logging.info(f'The Mean taxi ride duration predicted for {year:04d}/{month:02d} is: {y_pred.mean()}')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['predictions'] = y_pred

    df_result = df[['ride_id', 'predictions']].copy()

     # Write results to a parquet file.
    if not os.path.exists('./data'):
        os.makedirs('./data')

    output_file = f'data/{year:04d}-{month:02d}-predictions.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)

    # Parse the argument
    args = parser.parse_args()
    if args.month < 1 and args.month > 12:
        logging.info("Invalid details, please provide a valid month ranging from 1 to 12")
    else:
        ride_predictions(args.year, args.month)