import torch
import yaml
from bnn import BNN, BNNDataset, test_multiple_models
from cokriging import CK
from scalers import MinMaxScaler as MMS
from utils import create_data_dump
import pandas as pd

# Load the settings from the YAML files
def load_settings():
    with open("Settings/model_settings.yaml", "r") as file:
        model_settings = yaml.safe_load(file)
    with open("Settings/training_settings.yaml", "r") as file:
        training_settings = yaml.safe_load(file)
    DATASET_YAML_PATH = training_settings["DATASET_YAML_PATH"]
    with open(DATASET_YAML_PATH, "r") as file:
        dataset_settings = yaml.safe_load(file)
    return model_settings, training_settings, dataset_settings

# Load models from the specified paths
def load_models(model_settings, training_settings, dataset_settings):
    models_to_test = []

    # Load the low-fidelity model
    if training_settings["TRAIN_LF"]:
        bnn_lf_model = BNN(
            in_dim=len(dataset_settings["INPUT_LABELS"]),
            out_dim=len(dataset_settings["OUTPUT_LABELS"]),
            mu=model_settings["MU"],
            std=model_settings["STD"],
            units=model_settings["UNITS"],
            device=model_settings["DEVICE"],
        )
        bnn_lf_model.load(f"{training_settings['MODEL_PATH']}{model_settings['MODEL_NAME_LF']}.pt", device=model_settings["DEVICE"])
        models_to_test.append(bnn_lf_model)

    # Load the mid-fidelity model
    if training_settings["TRAIN_MF"]:
        bnn_mf_model = BNN(
            in_dim=len(dataset_settings["INPUT_LABELS"]),
            out_dim=len(dataset_settings["OUTPUT_LABELS"]),
            mu=model_settings["MU_MF"],
            std=model_settings["STD_MF"],
            units=model_settings["UNITS_MF"],
            device=model_settings["DEVICE"],
        )
        bnn_mf_model.load(f"{training_settings['MODEL_PATH']}{model_settings['MODEL_NAME_MF']}.pt", device=model_settings["DEVICE"])
        models_to_test.append(bnn_mf_model)

    # Load the transfer learning model
    if training_settings["TRAIN_TL"]:
        bnn_df_model = BNN(
            in_dim=len(dataset_settings["INPUT_LABELS"]),
            out_dim=len(dataset_settings["OUTPUT_LABELS"]),
            mu=model_settings["MU"],
            std=model_settings["STD"],
            units=model_settings["UNITS"],
            device=model_settings["DEVICE"],
        )
        bnn_df_model.load(f"{training_settings['MODEL_PATH']}{model_settings['MODEL_NAME']}.pt", device=model_settings["DEVICE"])
        models_to_test.append(bnn_df_model)

    # Load the Co-Kriging model
    if training_settings["TRAIN_CK"]:
        ck_df_model = CK(
            low_fidelity_model=bnn_lf_model,
            device=model_settings["DEVICE"],
        )
        ck_df_model.load(f"{training_settings['MODEL_PATH']}{model_settings['MODEL_NAME_CK']}.pkl")
        models_to_test.append(ck_df_model)

    return models_to_test

# Main function to execute the testing process
def main():
    # Load all settings
    model_settings, training_settings, dataset_settings = load_settings()

    # Load all trained models
    models_to_test = load_models(model_settings, training_settings, dataset_settings)

    # Load the dataset and scaler
    dataset = pd.read_csv(dataset_settings["DATASET_LOCATION"], sep=dataset_settings["SEP"])
    scaler = MMS()
    scaler.load(f"{training_settings['MODEL_PATH']}NormalizationData/normdata.pkl")

    # Prepare the mid-fidelity test dataset
    test_mf = BNNDataset(
        dataset[dataset[dataset_settings["FIDELITY_COLUMN_NAME"]] == dataset_settings["FIDELITIES"][1]],
        dataset_settings["INPUT_LABELS"],
        dataset_settings["OUTPUT_LABELS"],
        device=model_settings["DEVICE"]
    )

    # Test all models
    print("Testing models...")
    test_multiple_models(
        models_to_test,
        test_mf,
        scaler,
        dataset_settings["OUTPUT_LABELS"],
        path=f"{training_settings['MODEL_PATH']}/error_results.csv",
        save_test_results=True
    )

    # Dump the model and training settings, along with trained models, to a YAML file
    data_dump = create_data_dump(
        model_settings,
        dataset_settings,
        training_settings,
        *models_to_test
    )

    print("Dumping data...")
    with open(f"{training_settings['MODEL_PATH']}/model_info.yaml", "w") as file:
        yaml.dump(data_dump, file, default_flow_style=True)

    print("Test completed successfully.")

if __name__ == "__main__":
    main()
