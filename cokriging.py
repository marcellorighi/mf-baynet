# AUTHOR: Andrea Vaiuso
MODEL_VERSION = "1.3"

import GPy
import numpy as np
from bnn import BNN, BNNDataset
from scalers import MinMaxScaler as MMS
from utils import PrCol
import pandas as pd
import torch
from time import time
import pickle
import os

Color = PrCol()

class CK():
    """
    Co-Kriging model class that combines predictions from a low-fidelity Bayesian Neural Network (BNN)
    with a Gaussian Process (GP) regression model for multi-fidelity modeling.

    Attributes:
        version (str): Version of the CK model.
        model_name (str): Name of the CK model.
        low_fidelity_model (BNN): The low-fidelity Bayesian Neural Network model.
        device (str): Device on which the model is trained and evaluated.
        model (GPy.models.GPRegression): Trained Gaussian Process model.
        tot_time (float): Total time taken for training the CK model.
    """

    def __init__(self,
                 low_fidelity_model: BNN,
                 device="cpu",
                 model_name="COKriging"):
        """
        Initializes the CK model with the low-fidelity model, device, and model name.

        Args:
            low_fidelity_model (BNN): The low-fidelity Bayesian Neural Network model.
            device (str): Device on which the model will be trained and evaluated (default: "cpu").
            model_name (str): Name of the CK model (default: "COKriging").
        """
        super().__init__()
        self.version = MODEL_VERSION
        self.model_name = model_name
        self.low_fidelity_model = low_fidelity_model
        self.device = device
        self.model = None
        self.tot_time = 0
    
    def train_model(self,
                    mid_fid_data: BNNDataset,
                    low_fid_attempt: int = 100):
        """
        Trains the CK model by using the low-fidelity model predictions as additional inputs
        to a Gaussian Process regression model.

        Args:
            mid_fid_data (BNNDataset): Mid-fidelity dataset used for training the GP model.
            low_fid_attempt (int): Number of stochastic forward passes for generating low-fidelity predictions (default: 100).
        """
        t_1 = time()
        LF_PRED_COL_NAME = "lf_pred"
        means, _ = self.low_fidelity_model.predict(np.array(mid_fid_data.x), attempt=low_fid_attempt, scaler=None, output_labels=mid_fid_data.output_labels)
        means = np.array(means)
        dataset = mid_fid_data.data
        lf_out_col_names = []
        for i, out_lab in enumerate(mid_fid_data.output_labels):
            col_name = LF_PRED_COL_NAME + "_" + out_lab
            dataset[col_name] = means[:, i]
            lf_out_col_names.append(col_name)
        input_labels = mid_fid_data.input_labels + lf_out_col_names
        X_ck = dataset[input_labels].values
        y_ck = dataset[mid_fid_data.output_labels].values
        
        # Define the kernel and train the GP model
        kernel = GPy.kern.Linear(input_dim=X_ck.shape[1], variances=1.0)
        gp_model = GPy.models.GPRegression(X_ck, y_ck, kernel=kernel)
        gp_model.optimize(optimizer='lbfgsb', max_iters=1000)
        self.model = gp_model
        self.tot_time = time() - t_1

    def predict(self, x, output_labels: str, attempt: int = 100, scaler: MMS = None, returnDataFrame=False):
        """
        Generates predictions using the trained CK model, combining the low-fidelity model's predictions
        with the Gaussian Process model.

        Args:
            x (array-like): Input data for prediction.
            output_labels (str): List of output labels for the prediction.
            attempt (int): Number of stochastic forward passes for uncertainty estimation in the low-fidelity model (default: 100).
            scaler (MMS): Scaler object for reversing normalization (default: None).
            returnDataFrame (bool): If True, returns results as pandas DataFrames (default: False).

        Returns:
            Tuple: Mean and standard deviation of predictions, optionally as DataFrames.
        """
        if self.model is None:
            raise RuntimeError("GP model not trained")
        predictions, _ = self.low_fidelity_model.predict(x, attempt=attempt, scaler=None, output_labels=output_labels)
        if isinstance(x[0], torch.Tensor):
            x = np.array([tensor.numpy() for tensor in x]) 
        predictions = np.array(predictions)
        input_data_cokg = np.column_stack((x, predictions))
        y_mean_gp, pred_std_gp = self.model.predict(input_data_cokg)
        pred_std_gp_new = []
        for i in range(len(pred_std_gp)):
            new_val = pred_std_gp[i] * np.ones(len(output_labels), dtype=int)
            pred_std_gp_new.append(new_val)
        
        if scaler is not None and output_labels is not None:
            for i in range(len(y_mean_gp)):
                y_mean_gp[i] = scaler.reverseArray(y_mean_gp[i], columns=output_labels)

        means = y_mean_gp
        stds = np.array(pred_std_gp_new) * 10

        if returnDataFrame and output_labels is not None:
            return pd.DataFrame(means, columns=output_labels), pd.DataFrame(stds, columns=output_labels)
        else:
            return means, stds

    def _mae(self, gt: np.ndarray, pred: np.ndarray):
        """
        Computes the Mean Absolute Error (MAE) between the ground truth and predictions.

        Args:
            gt (np.ndarray): Ground truth values.
            pred (np.ndarray): Predicted values.

        Returns:
            np.ndarray: The MAE between ground truth and predictions.
        """
        return np.abs(gt - pred)

    def testModel(self, test_set: BNNDataset, scaler: MMS, output_labels: list, attempt: int = 10, skip=[]):
        """
        Tests the CK model on a test dataset and returns prediction errors and uncertainties.

        Args:
            test_set (BNNDataset): The test dataset.
            scaler (MMS): Scaler object to reverse normalization.
            output_labels (list): List of output labels.
            attempt (int): Number of stochastic forward passes for uncertainty estimation (default: 10).
            skip (list): Indices to skip during testing (default: []).

        Returns:
            Tuple[list, list]: Prediction errors and standard deviations.
        """
        errors = []
        cov = []
        for i, data in enumerate(test_set):
            if i in skip:
                continue
            x, y = data
            y = scaler.reverseArray(y, columns=output_labels)
            preds, stds = self.predict([x], scaler=scaler, output_labels=output_labels, attempt=attempt)
            preds = np.array(preds[0])
            stds = stds[0]
            errors.append(self._mae(y, preds))
            cov.append(stds)
        return errors, cov

    def save(self, path: str, subfolder: str = "AIModels/"):
        """
        Saves the CK model, including the trained Gaussian Process model and metadata.

        Args:
            path (str): The path where the model will be saved.
            subfolder (str): Subfolder to save the model in (default: "AIModels/").
        """
        full_path = os.path.join(subfolder, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Save the Gaussian Process model
        with open(full_path + "_gp_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # Save metadata and low-fidelity model
        metadata = {
            'version': self.version,
            'model_name': self.model_name,
            'tot_time': self.tot_time,
            'low_fidelity_model_state_dict': self.low_fidelity_model.state_dict(),
            'device': self.device
        }

        with open(full_path + "_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"{Color.green}[{Color.end}{self.model_name}{Color.green}] >> Model Saved{Color.end}")

    def load(self, path: str):
        """
        Loads the CK model, including the trained Gaussian Process model and metadata.

        Args:
            path (str): The path from where the model will be loaded.
        """
        # Load the Gaussian Process model
        with open(path + "_gp_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load metadata and low-fidelity model state
        with open(path + "_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.version = metadata['version']
        self.model_name = metadata['model_name']
        self.tot_time = metadata['tot_time']
        self.device = metadata['device']

        # Load the state dictionary into the low-fidelity model
        self.low_fidelity_model.load_state_dict(metadata['low_fidelity_model_state_dict'])
        self.low_fidelity_model.to(self.device)

        print(f"{Color.green}[{Color.end}{self.model_name}{Color.green}] >> Model Loaded from {Color.blue}{path}{Color.end}")
