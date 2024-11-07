import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import copy
import os
import yaml
import sys
from utils import Color, show_continue_cancel, create_data_dump
from bnn import BNN, BNNDataset, test_multiple_models
from cokriging import CK
from scalers import MinMaxScaler as MMS

# Function to import settings from YAML configuration files
def import_settings():
    """
    Imports model, training, and dataset settings from YAML configuration files.

    Returns:
        tuple: A tuple containing dictionaries for model settings, training settings, and dataset settings.
    """
    with open("Settings/model_settings.yaml", "r") as file:
        model_settings = yaml.safe_load(file)
    with open("Settings/training_settings.yaml", "r") as file:
        training_settings = yaml.safe_load(file)
    DATASET_YAML_PATH = training_settings["DATASET_YAML_PATH"]
    with open(DATASET_YAML_PATH, "r") as file:
        dataset_settings = yaml.safe_load(file)
    return model_settings, training_settings, dataset_settings

# Function to create directories if they do not exist
def setup_model_directory(MODEL_PATH):
    """
    Creates the directory for saving models if it does not exist.

    Args:
        MODEL_PATH (str): The path where the model should be saved.

    Returns:
        bool: True if the directory already exists, False if it was created.
    """
    path_exists = False
    try: 
        os.makedirs(MODEL_PATH)
    except FileExistsError: 
        resp = show_continue_cancel(f"The model {MODEL_PATH} already exists. All the models will be overwritten. Do you want to continue?")
        if not resp:
            sys.exit()
        path_exists = True
    except Exception:
        print("Error")
    return path_exists

# Function to load and normalize datasets
def load_and_normalize_datasets(dataset_settings, path_exists, model_path):
    """
    Loads the dataset and applies normalization using MinMaxScaler.

    Args:
        dataset_settings (dict): Settings related to the dataset.
        path_exists (bool): Indicates whether the model path already exists.
        MODEL_PATH (str): The path where normalization data should be saved/loaded.

    Returns:
        tuple: A tuple containing a list of normalized datasets and the scaler used (the order is defined by the list parameter 'FIDELITIES' of dataset_settings.yaml).
    """
    dataset = pd.read_csv(dataset_settings["DATASET_LOCATION"], sep=dataset_settings["SEP"])
    in_out_col = dataset_settings["OUTPUT_LABELS"] + dataset_settings["INPUT_LABELS"]
    all_dataset_columns = dataset.columns.tolist()
    drop_col_list = [col for col in all_dataset_columns if col not in in_out_col]
    if not path_exists:
        scaler = MMS(dataset.drop(columns=drop_col_list, inplace=False), interval=(1, 2))
        scaler.save(path=f"{model_path}NormalizationData", filename=f"normdata.pkl")
    else:
        scaler = MMS()
        scaler.load(f"{model_path}NormalizationData/normdata.pkl")
    
    normalized_datasets = []
    for fid in dataset_settings["FIDELITIES"]:
        dataset_fidelity = dataset[dataset[dataset_settings["FIDELITY_COLUMN_NAME"]] == fid]
        dataset_fidelity_norm = scaler.scaleDataframe(dataset_fidelity.drop(columns=drop_col_list, inplace=False))
        normalized_datasets.append(dataset_fidelity_norm)
    return normalized_datasets, scaler

# Function to train the single-fidelity models
def train_single_fidelity_model(fidelity, INPUT_LABELS, OUTPUT_LABELS, model_settings, training_settings, MODEL_PATH, train_data, valid_data, verbose=True, showPlotHistory=True, show_loadbar=True):
    """
    Trains the Bayesian Neural Network (BNN) model.

    Args:
        fidelity (str): The fidelity level to train the model (e.g. 'HF', 'LF' or 'MF')
        INPUT_LABELS (list): List of input feature names.
        OUTPUT_LABELS (list): List of output feature names.
        model_settings (dict): Settings for the BNN model.
        training_settings (dict): Training hyperparameters.
        MODEL_PATH (str): Path to save the trained model.
        train_data (BNNDataset): Training dataset.
        valid_data (BNNDataset): Validation dataset.

    Returns:
        BNN: Trained model.
    """
    bnn_model = BNN(
        in_dim=len(INPUT_LABELS),
        out_dim=len(OUTPUT_LABELS),
        mu=model_settings[f"MU_{fidelity}"],
        std=model_settings[f"STD_{fidelity}"],
        units=model_settings[f"UNITS_{fidelity}"],
        denseOut=False,
        dropout=False,
        device=model_settings[f"DEVICE"],
        activation=nn.LeakyReLU(),
        model_name=model_settings[f"MODEL_NAME_{fidelity}"]
    )
    if training_settings[f"TRAIN_{fidelity}"]:
        bnn_model.train(
            train_data,
            valid_data,
            patience=training_settings[f"PATIENCE_{fidelity}"],
            n_epochs=training_settings[f"N_EPOCHS_{fidelity}"],
            batch_size=training_settings[f"BATCH_SIZE_{fidelity}"],
            lr=training_settings[f"LR_{fidelity}"],
            history_plot_saving_path=MODEL_PATH,
            verbose=verbose, showPlotHistory=showPlotHistory, show_loadbar=show_loadbar
        )
        bnn_model.save(f"{MODEL_PATH}{bnn_model.model_name}.pt", subfolder="")
    return bnn_model

# Function to perform transfer learning
def transfer_learning(fidelity, model_settings, training_settings, MODEL_PATH, train_hf, valid_hf, bnn_lf_model, verbose=True, showPlotHistory=True, show_loadbar=True):
    """
    Performs transfer learning by fine-tuning the lower-fidelity model on the higher-fidelity dataset.

    Args:
        fidelity (str): The fidelity level to train the model (e.g. 'TL_HF', 'TL_MF')
        model_settings (dict): Settings for the BNN model.
        training_settings (dict): Training hyperparameters.
        MODEL_PATH (str): Path to save the fine-tuned model.
        train_mf (BNNDataset): Training dataset for higher-fidelity model.
        valid_mf (BNNDataset): Validation dataset for higher-fidelity model.
        bnn_lf_model (BNN): Pre-trained lower-fidelity model to be fine-tuned.

    Returns:
        BNN: Fine-tuned model.
    """
    bnn_df_model = copy.deepcopy(bnn_lf_model) 
    bnn_df_model.model_name = model_settings[f"MODEL_NAME_{fidelity}"]
    bnn_df_model.setModelGradients(False)
    all_layers = bnn_df_model.getAllLayersName()
    bnn_df_model.setModelGradients(True, layers=all_layers[-training_settings[f"N_LAYER_TO_UNFREEZE_{fidelity}"]:])
    if training_settings[f"TRAIN_{fidelity}"]:
        bnn_df_model.train(
            train_hf,
            valid_hf,
            n_epochs=training_settings[f"N_EPOCHS_{fidelity}"],
            lr=training_settings[f"LR_{fidelity}"],
            restoreBestModel=True,
            patience=training_settings[f"PATIENCE_{fidelity}"],
            batch_size=training_settings[f"BATCH_SIZE_{fidelity}"],
            history_plot_saving_path=MODEL_PATH,
            verbose=verbose, showPlotHistory=showPlotHistory, show_loadbar=show_loadbar
        )
        bnn_df_model.save(f"{MODEL_PATH}{bnn_df_model.model_name}.pt", subfolder="")
    return bnn_df_model

# Function to train the Co-Kriging model
def train_co_kriging_model(training_settings, MODEL_PATH, model_settings, train_mf, bnn_lf_model):
    """
    Trains the Co-Kriging model using the low-fidelity model as a basis.

    Args:
        training_settings (dict): Training hyperparameters.
        MODEL_PATH (str): Path to save the trained model.
        model_settings (dict): Settings for the CK model.
        train_mf (BNNDataset): Training dataset for mid-fidelity model.
        bnn_lf_model (BNN): Pre-trained low-fidelity model.

    Returns:
        CK: Trained Co-Kriging model.
    """
    ck_df_model = CK(bnn_lf_model)
    if training_settings["TRAIN_CK"]:
        print("Training Co-Kriging...")
        ck_df_model.train_model(mid_fid_data=train_mf, low_fid_attempt=100)
        ck_df_model.save(f"{MODEL_PATH}{model_settings['MODEL_NAME_CK']}.pkl", subfolder="")
    return ck_df_model

# Function to test all models
def test_models(models_to_test, test_mf, scaler, OUTPUT_LABELS, MODEL_PATH):
    """
    Tests all trained models on the mid-fidelity test dataset.

    Args:
        models_to_test (list): List of models to be tested.
        test_mf (BNNDataset): Test dataset for mid-fidelity model.
        scaler (MMS): Scaler used for normalization.
        OUTPUT_LABELS (list): List of output feature names.
        MODEL_PATH (str): Path to save the test results.
    
    Returns:
        pd.DataFrame: DataFrame containing the error metrics for each model.
    """
    print("Testing models...")
    return test_multiple_models(models_to_test, test_mf, scaler, OUTPUT_LABELS, path=f"{MODEL_PATH}/error_results.csv", save_test_results=True)

# Function to save the model settings and trained models
def save_model_data(model_settings, dataset_settings, training_settings, MODEL_PATH, bnn_lf_model=None, bnn_mf_model=None, bnn_df_model=None, ck_df_model=None):
    """
    Saves the model and training settings, along with the trained models, to a YAML file.

    Args:
        model_settings (dict): Settings for the BNN and CK models.
        dataset_settings (dict): Settings for the dataset.
        training_settings (dict): Training hyperparameters.
        MODEL_PATH (str): Path to save the YAML file.
        bnn_lf_model (BNN): Trained low-fidelity model.
        bnn_mf_model (BNN): Trained mid-fidelity model.
        bnn_df_model (BNN): Fine-tuned model.
        ck_df_model (CK): Trained Co-Kriging model.
    """
    data_dump = create_data_dump(model_settings, dataset_settings, training_settings, bnn_lf_model, bnn_mf_model, bnn_df_model, ck_df_model)
    print("Dumping data...")
    with open(f"{MODEL_PATH}/model_info.yaml", "w") as file:
        yaml.dump(data_dump, file, default_flow_style=True)

# Function to plot results and compare model predictions against validation data
def plot_results(predict_data, 
                 validation_dataset, 
                 input_labels, 
                 output_labels, 
                 x_lab, 
                 y_lab,
                 show_test_set = False,
                 show_ck = True,
                 out_prefix = "T",
                 std_adj_factor = 3,
                 showfig = True,
                 MODEL_PATH = "",
                 bnn_lf_model=None,
                 bnn_mf_model=None,
                 bnn_df_model=None,
                 ck_df_model=None,
                 scaler=None):
    """
    Plots the predictions of various models and compares them against the validation dataset.

    Args:
        predict_data (pd.DataFrame): DataFrame containing input data for prediction.
        validation_dataset (pd.DataFrame): Validation dataset for comparison.
        input_labels (list): List of input feature names.
        output_labels (list): List of output feature names.
        x_lab (str): Label for the x-axis.
        y_lab (str): Label for the y-axis.
        show_test_set (bool): If True, shows the test set points in the plot (default: False).
        show_ck (bool): If True, includes the Co-Kriging model in the plot (default: True).
        out_prefix (str): Prefix for output labels (default: "T").
        std_adj_factor (int): Factor to adjust the standard deviation in the plot (default: 3).
        showfig (bool): If True, displays the plot (default: True).
        MODEL_PATH (str): Path to save the plot.
        bnn_lf_model (BNN): Trained low-fidelity model.
        bnn_mf_model (BNN): Trained mid-fidelity model.
        bnn_df_model (BNN): Fine-tuned model.
        ck_df_model (CK): Trained Co-Kriging model.
        scaler (MMS): Scaler used for normalization.
    """
    x = validation_dataset[input_labels]
    y = validation_dataset[output_labels]
    predict_data_scaled = scaler.scaleDataframe(predict_data)
    pred_lf, _ = bnn_lf_model.predict(predict_data_scaled[input_labels].values, scaler=scaler, output_labels=output_labels, returnDataFrame=True)
    pred_mf, _ = bnn_mf_model.predict(predict_data_scaled[input_labels].values, scaler=scaler, output_labels=output_labels, returnDataFrame=True)
    pred_df, std_df = bnn_df_model.predict(predict_data_scaled[input_labels].values, scaler=scaler, output_labels=output_labels, returnDataFrame=True)
    pred_ck, std_ck = ck_df_model.predict(predict_data_scaled[input_labels].values, scaler=scaler, output_labels=output_labels, returnDataFrame=True)
    std_df *= std_adj_factor
    std_ck *= std_adj_factor
    pred_upper_bnn = np.array(pred_df[y_lab]) + (np.array(std_df[y_lab]))
    pred_lower_bnn = np.array(pred_df[y_lab]) - (np.array(std_df[y_lab]))
    pred_upper_ck = np.array(pred_ck[y_lab]) + (np.array(std_ck[y_lab]))
    pred_lower_ck = np.array(pred_ck[y_lab]) - (np.array(std_ck[y_lab]))
    x_pred = np.abs(predict_data[x_lab])
    plt.figure(figsize=(6,4))
    plt.title(f"[aoa={predict_data['aoa'][0]}, aos={predict_data['aos'][0]}, uinf={predict_data['u_inf'][0]}]")
    if show_test_set: 
        plt.scatter(np.abs(x[x_lab]), y[y_lab], marker="*", s=50, label="Test-set Points", color="black")
    plt.plot(x_pred, pred_lf[y_lab], label=bnn_lf_model.model_name)
    plt.plot(x_pred, pred_mf[y_lab], markersize=1, label=bnn_mf_model.model_name)
    plt.fill_between(x_pred, pred_lower_bnn, pred_upper_bnn, alpha=0.3, zorder=5, color="red")
    plt.plot(np.abs(predict_data[x_lab]), pred_df[y_lab], label=bnn_df_model.model_name, color="red")
    if show_ck:
        plt.fill_between(x_pred, pred_lower_ck, pred_upper_ck, alpha=0.3, zorder=5, color="orange")
        plt.plot(np.abs(predict_data[x_lab]), pred_ck[y_lab], label=ck_df_model.model_name, color="orange")
    if out_prefix == "T": 
        plt.ylabel(f"THRUST {x_lab}")
    if out_prefix == "Q": 
        plt.ylabel(f"TORQUE {x_lab}")
    plt.xlabel("RPM")
    plt.legend()
    plt.savefig(f"{MODEL_PATH}validation_plot.pdf")
    if showfig: 
        plt.show()