import json
import os
from datetime import datetime
import dill
import pandas as pd


path = os.environ.get('PROJECT_PATH', '.')
# path = os.path.expanduser('~/airflow_hw')

def predict():
    latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]

    with open(f'{path}/data/models/{latest_model}', 'rb') as f:
        model = dill.load(f)
    test_cars = os.listdir(f'{path}/data/test')
    preds = {'car_id': [], 'pred': []}

    for car_id in test_cars:
        with open(f'{path}/data/test/{car_id}', 'rb') as file:
            car = json.load(file)

        df = pd.DataFrame(car, index=[0])
        y = model.predict(df)
        preds['car_id'].append(car_id.split('.')[0])
        preds['pred'].append(y[0])

    df_preds = pd.DataFrame(preds)
    now = datetime.now().strftime("%Y%m%d%H%M")
    df_preds.to_csv(f'{path}/data/predictions/{now}.csv', index=False)



if __name__ == '__main__':
    predict()
