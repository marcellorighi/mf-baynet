from flask import Flask, request, jsonify
from scalers import MinMaxScaler as MMS
from bnn import BNN
import torch
import argparse
import pandas as pd
import yaml

with open("Settings/server_settings.yaml", "r") as file:
    server_settings = yaml.safe_load(file)
with open("Settings/model_settings.yaml", "r") as file:
    model_settings = yaml.safe_load(file)
with open("Settings/training_settings.yaml", "r") as file:
    training_settings = yaml.safe_load(file)
with open(training_settings["DATASET_YAML_PATH"], "r") as file:
    dataset_settings = yaml.safe_load(file)

MODEL_PATH = server_settings["MODEL_PATH"]
NORM_DATA_PATH = server_settings["NORM_DATA_PATH"]
DEVICE = server_settings["DEVICE"]

INPUT_LABELS = dataset_settings["INPUT_LABELS"]
OUTPUT_LABELS = dataset_settings["OUTPUT_LABELS"]

MU = model_settings["MU"]
STD = model_settings["STD"]
UNITS = model_settings["UNITS"]
MODEL_NAME = model_settings["MODEL_NAME"]

scaler = MMS()
scaler.load(NORM_DATA_PATH)

model = BNN(
    in_dim=len(INPUT_LABELS),
    out_dim=len(OUTPUT_LABELS),
    mu=MU,
    std=STD,
    units=UNITS,
    device=DEVICE,
    model_name=MODEL_NAME
)
model.load(MODEL_PATH)

def pred_bnn(aoa:float, 
             aos:float, 
             u_inf:float, 
             PP:int, 
             FR:int, 
             FL:int, 
             RR:int, 
             RL:int, 
             attempt:int=10) -> dict:
    x = [aoa, aos, u_inf, PP, FR, FL, RR, RL]
    input_tensor = torch.tensor(scaler.scaleArray(x, columns=INPUT_LABELS), dtype=torch.float32)
    mean_df, std_df = model.predict([input_tensor],
                         scaler=scaler,
                         output_labels=OUTPUT_LABELS,
                         attempt=attempt,
                         returnDataFrame=True
                         )
    mean_df = mean_df.add_prefix('mean_')
    std_df = std_df.add_prefix('std_')
    pred = pd.concat([mean_df, std_df], axis=1)
    return pred.iloc[0].to_dict()

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_params = ['aoa', 'aos', 'u_inf', 'PP', 'FR', 'FL', 'RR', 'RL']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'Missing parameter: {param}'}), 400
        
        try:
            aoa = float(data['aoa'])
            aos = float(data['aos'])
            u_inf = float(data['u_inf'])
        except ValueError:
            return jsonify({'error': 'Invalid type for aoa, aos, or u_inf; expected float'}), 400
        
        try:
            PP = int(data['PP'])
            FR = int(data['FR'])
            FL = int(data['FL'])
            RR = int(data['RR'])
            RL = int(data['RL'])
        except ValueError:
            return jsonify({'error': 'Invalid type for PP, FR, FL, RR, or RL; expected int'}), 400

        attempt = data.get('attempt', 100)
        
        if not isinstance(attempt, int) or attempt <= 0:
            return jsonify({'error': 'Invalid value for attempt; expected a positive integer'}), 400

        result = pred_bnn(aoa, aos, u_inf, PP, FR, FL, RR, RL, attempt)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with open("Settings/server_settings.yaml", "r") as file:
        server_settings = yaml.safe_load(file)
    app.run(host=server_settings["HOST_ADDRESS"], port=server_settings["PORT_NO"])
