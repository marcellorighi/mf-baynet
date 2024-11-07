# AUTHOR: Andrea Vaiuso
MODEL_VERSION = "4.16"

import torch
import torch.nn as nn
import torchbnn as bnn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import PrCol, CircularBuffer, LoadBar, seconds_to_hhmmss
from time import time
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from scalers import MinMaxScaler as MMS
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import List, Tuple
import random

Color = PrCol()

class BNNDataset(Dataset):
    """
    A custom PyTorch Dataset for Bayesian Neural Networks (BNN).

    Attributes:
        data (pd.DataFrame): The dataset.
        input_labels (list): List of column names for input features.
        output_labels (list): List of column names for output features.
        device (str): The device to use for tensors (default: "cpu").
        dtype (torch.dtype): The data type for tensors (default: torch.float32).
    """
    def __init__(self, data: pd.DataFrame, input_labels: list, output_labels: list, device = "cpu", dtype = torch.float32):
        """
        Initializes the BNNDataset with data and labels.

        Args:
            data (pd.DataFrame): The dataset.
            input_labels (list): List of column names for input features.
            output_labels (list): List of column names for output features.
            device (str): The device to use for tensors (default: "cpu").
            dtype (torch.dtype): The data type for tensors (default: torch.float32).
        """
        self.x = data[input_labels]
        self.y = data[output_labels]
        self.data = data
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.device = device
        self.dtype = dtype
        self.tot_time = 0
        self.best_epoch = 0

    def __add__(self, other):
        """
        Combines two BNNDataset instances.

        Args:
            other (BNNDataset): Another instance of BNNDataset.

        Returns:
            BNNDataset: A new BNNDataset instance combining both datasets.
        """
        if not isinstance(other, BNNDataset):
            raise TypeError("Both operands must be of type BNNDataset")
        combined_data = pd.concat([self.data, other.data], ignore_index=True)
        return BNNDataset(combined_data, 
                          self.input_labels, 
                          self.output_labels, 
                          self.device, 
                          self.dtype)

    def __str__(self):
        """
        Returns the string representation of the dataset.

        Returns:
            str: String representation of the dataset.
        """
        return self.data.__str__()
    
    def __sizeof__(self) -> int:
        """
        Returns the size of the dataset in bytes.

        Returns:
            int: Size of the dataset in bytes.
        """
        return self.data.__sizeof__()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and output tensors.
        """
        return torch.tensor(self.x.iloc[idx].values, device=self.device, dtype=self.dtype), torch.tensor(self.y.iloc[idx].values, device=self.device, dtype=self.dtype)
    
    def train_val_test_split(self, train_size: float = 0.7, val_size: float = 0.15, seed: int = 42):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            train_size (float): Proportion of the dataset to include in the training set.
            val_size (float): Proportion of the dataset to include in the validation set.
            seed (int): Random seed for shuffling the dataset.

        Returns:
            Tuple[BNNDataset, BNNDataset, BNNDataset]: The training, validation, and test datasets.
        """
        total_size = len(self.data)
        train_len = int(train_size * total_size)
        val_len = int(val_size * total_size)
        data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_data, val_data, test_data = data[:train_len], data[train_len:train_len + val_len], data[train_len + val_len:]
        train_dataset = BNNDataset(train_data, self.input_labels, self.output_labels)
        val_dataset = BNNDataset(val_data, self.input_labels, self.output_labels)
        test_dataset = BNNDataset(test_data, self.input_labels, self.output_labels)
        return train_dataset, val_dataset, test_dataset
    
    def train_val_split(self, train_size: float = 0.7, seed: int = 42):
        """
        Splits the dataset into training and validation sets.

        Args:
            train_size (float): Proportion of the dataset to include in the training set.
            seed (int): Random seed for shuffling the dataset.

        Returns:
            Tuple[BNNDataset, BNNDataset]: The training and validation datasets.
        """
        total_size = len(self.data)
        train_len = int(train_size * total_size)
        data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_data, val_data = data[:train_len], data[train_len:]
        train_dataset = BNNDataset(train_data, self.input_labels, self.output_labels)
        val_dataset = BNNDataset(val_data, self.input_labels, self.output_labels)

        return train_dataset, val_dataset
    
    def remove_random(self, num_to_remove: int, seed: int = None):
        """
        Removes a specified number of random elements from the dataset.

        Args:
            num_to_remove (int): The number of elements to remove from the dataset.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            BNNDataset: A new BNNDataset instance with the specified number of elements removed, or an empty BNNDataset if num_to_remove is greater than or equal to the dataset size.
        """
        if seed is not None:
            random.seed(seed)
        
        if num_to_remove == 0:
            # Return a copy of the current dataset
            return BNNDataset(self.data.copy(), self.input_labels, self.output_labels, self.device, self.dtype)
        
        if num_to_remove >= len(self.data):
            # Return an empty dataset
            empty_data = pd.DataFrame(columns=self.data.columns)
            return BNNDataset(empty_data, self.input_labels, self.output_labels, self.device, self.dtype)
        
        indices_to_remove = random.sample(range(len(self.data)), num_to_remove)
        remaining_data = self.data.drop(self.data.index[indices_to_remove]).reset_index(drop=True)
        
        return BNNDataset(remaining_data, self.input_labels, self.output_labels, self.device, self.dtype)

class BNN(nn.Module):
    """
    A Bayesian Neural Network (BNN) implementation using PyTorch and torchbnn.

    Attributes:
        in_layer (bnn.BayesLinear): Input layer of the BNN.
        hidden_layers (nn.ModuleList): List of hidden layers in the BNN.
        out_layer (nn.Module): Output layer of the BNN.
        activation (nn.Module): Activation function used in the BNN.
        dropout_flag (bool): Indicates if dropout is used.
        dropout_norm (nn.Dropout): Dropout layer for input.
        hidden_dropout (nn.ModuleList): Dropout layers for hidden layers.
        device (str): Device to run the model on.
        model_name (str): Name of the model.
        version (str): Model version.
    """
    def __init__(self, 
                 in_dim:int, 
                 out_dim:int, 
                 mu:float=0, 
                 std:float=0.5, 
                 units:list=[100], 
                 denseOut:bool = False,
                 dropout:bool = False,
                 device:str = "cpu",
                 activation:nn.Module = nn.LeakyReLU(),
                 model_name:str = "BNN DEFAULT"
                 ):
        """
        Initializes the Bayesian Neural Network (BNN).

        Args:
            in_dim (int): Number of input features.
            out_dim (int): Number of output features.
            mu (float): Mean for the prior distribution.
            std (float): Standard deviation for the prior distribution.
            units (list): List of units for hidden layers.
            denseOut (bool): If True, use a deterministic output layer; otherwise, use a Bayesian output layer.
            dropout (bool): If True, use dropout layers.
            device (str): Device to run the model on (default: "cpu").
            activation (nn.Module): Activation function to use (default: nn.LeakyReLU()).
            model_name (str): Name of the model (default: "BNN DEFAULT").
        """
        super().__init__()
        self.version = MODEL_VERSION
        self.model_name = model_name
        self.device = device
        self.to(self.device)
        self.dropout_flag = dropout
        self.activation = activation.to(self.device)
        
        if self.dropout_flag: self.dropout_norm = nn.Dropout(0.2)
        self.in_layer = bnn.BayesLinear(prior_mu=mu, prior_sigma=std, in_features=in_dim, out_features=units[0]).to(self.device)
        self.hidden_layers = nn.ModuleList([
            bnn.BayesLinear(prior_mu=mu, prior_sigma=std, in_features=units[i-1], out_features=units[i]).to(self.device)
            for i in range(1,len(units))
        ])
        if self.dropout_flag: self.hidden_dropout = nn.ModuleList([
            nn.Dropout(0.2)
            for _ in range(1,len(units))
        ])
        if denseOut:
            self.out_layer = nn.Linear(units[-1],out_dim).to(self.device)
        else:
            self.out_layer = bnn.BayesLinear(prior_mu=mu, prior_sigma=std, in_features=units[-1], out_features=out_dim).to(self.device)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.activation(self.in_layer(x))
        if self.dropout_flag: x = self.dropout_norm(x)
        for i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[i]
            if self.dropout_flag: dropout = self.hidden_dropout[i]
            x = self.activation(layer(x))
            if self.dropout_flag: x = dropout(x)
        return self.out_layer(x)
    
    def train(self, 
            train_data: BNNDataset,
            valid_data: BNNDataset,
            n_epochs: int = 1000, 
            patience: int = 20, 
            lr: float = 0.001, 
            batch_size:int = 1, 
            earlyStopping:bool = True, 
            shuffle:bool = True, 
            restoreBestModel:bool = True, 
            verbose:bool = True, 
            show_loadbar:bool = True,
            showPlotHistory:bool = True,
            lrdecay:tuple = None, #(step_size, gamma)
            lrdecay_limit:float = 0.00005,
            history_plot_saving_path: str = None,
            batchnorm:int = 0
            ):
        """
        Trains the Bayesian Neural Network (BNN).

        Args:
            train_data (BNNDataset): The training dataset.
            valid_data (BNNDataset): The validation dataset.
            n_epochs (int): Number of epochs to train (default: 1000).
            patience (int): Number of epochs to wait for improvement before stopping (default: 20).
            lr (float): Learning rate (default: 0.001).
            batch_size (int): Batch size (default: 1).
            earlyStopping (bool): If True, use early stopping (default: True).
            shuffle (bool): If True, shuffle the data (default: True).
            restoreBestModel (bool): If True, restore the best model at the end of training (default: True).
            verbose (bool): If True, print training progress (default: True).
            show_loadbar (bool): If True, show a progress bar during training (default: True).
            showPlotHistory (bool): If True, display the loss history plot (default: True).
            lrdecay (tuple): Tuple containing step size and decay factor for learning rate scheduler (default: None).
            lrdecay_limit (float): The lower limit for learning rate decay (default: 0.00005).
            history_plot_saving_path (str): Path to save the loss history plot (default: None).
            batchnorm (int): Type of batch normalization to apply (default: 0).
        """
        self.earlyStopping = earlyStopping
        train_data.device = self.device
        valid_data.device = self.device
        if verbose: print(f"{Color.yellow}Creating dataloader (train size: {len(train_data)}, valid size: {len(valid_data)})...{Color.end}")
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle)
        if verbose: print(f"{Color.yellow}Initializing model architecture...{Color.end}")
        if verbose: print(self)
        if not self.earlyStopping:
            patience = np.inf
        
        self.kl_weight = 1 / len(train_data)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        if lrdecay:
            if isinstance(lrdecay, tuple): 
                if len(lrdecay) == 2: 
                    self.scheduler = StepLR(self.optimizer, step_size=lrdecay[0], gamma=lrdecay[1]) #30, 0.9
                else: raise TypeError("lrdecay parameter must be a bidimensional tuple: (step_size, gamma)")
            else: raise TypeError("lrdecay parameter must be a bidimensional tuple: (step_size, gamma)")
        else: self.scheduler = None

        self.best_val_loss = np.inf
        patience_count = 0
        best_model = self
        self.best_epoch = np.nan
        self.train_loss = np.inf
        self.valid_loss = np.inf
        self.train_loss_history = []
        self.valid_loss_history = []
        self.timeBuffer = CircularBuffer(50)
        loadBar = LoadBar()
        n_grd_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_tot_params = sum(p.numel() for p in self.parameters())
        if verbose: print(f"{Color.green}[{Color.end}{self.model_name}{Color.green}] >> Model initialized ({n_grd_params} learnable / {n_tot_params} total parameters){Color.end}\n")
        if verbose: print(f"{Color.green}Start Training{Color.end}\n")
        try:
            tot_t1 = time()
            for epoch in range(n_epochs):
                self.actual_epoch = epoch
                loadBar.tick()
                t1 = time()
                current_train_loss = 0
                current_valid_loss = 0

                with torch.no_grad():
                    if batchnorm==1 or batchnorm==2:
                        for param in self.parameters():
                            param.data = F.normalize(param.data, p=batchnorm, dim=0)

                if show_loadbar: print(f'{loadBar.loadBar(epoch + 1, n_epochs)} ' +
                        f'MSE (TRAIN) : {Color.yellow}{self.train_loss:.6f}{Color.end}, ' +
                        f'MSE (VAL) : {Color.cyan}{self.valid_loss:.6f}{Color.end} -- ' +
                        f'BEST Val Loss : {Color.green}{self.best_val_loss:.6f}{Color.end} ' +
                        f'(at epoch {self.best_epoch}) ' +
                        (f'Eary Stopping in: {str(patience - patience_count).zfill(4)}   ' if self.earlyStopping else '    '), end="\r")

                for i, data in enumerate(train_data_loader):
                    x,y = data
                    pre = self(x)
                    mse = self.mse_loss(pre, y)
                    kl = self.kl_loss(self)
                    cost = mse + self.kl_weight * kl

                    self.optimizer.zero_grad()
                    cost.backward()
                    self.optimizer.step()
                    current_train_loss += cost.item()

                self.train_loss = current_train_loss

                # Validation loop with batches
                for i, data in enumerate(valid_data_loader):
                    x,y = data
                    pre = self(x)
                    mse = self.mse_loss(pre, y)
                    kl = self.kl_loss(self)
                    cost = mse + self.kl_weight * kl
                    current_valid_loss += cost.item()

                self.valid_loss = current_valid_loss

                if self.optimizer.param_groups[0]['lr'] > lrdecay_limit and lrdecay:
                    self.scheduler.step()

                if self.valid_loss <= self.best_val_loss:
                    self.best_val_loss = self.valid_loss
                    patience_count = 0
                    best_model = self
                    self.best_epoch = epoch
                else:
                    patience_count += 1

                self.train_loss_history.append(self.train_loss)
                self.valid_loss_history.append(self.valid_loss)
                t2 = time() - t1
                self.timeBuffer.add_element(t2)
                loadBar.tock()
                if patience_count >= patience:
                    if verbose: print(f"\n{Color.yellow}Early stopping at epoch {self.best_epoch}{Color.end}")
                    self.tot_time = time() - tot_t1
                    if verbose: self.tot_time = time() - tot_t1; print(f"Total enlapsed time: {(self.tot_time):.2f} sec")
                    break

            self._plotHistory(save_path=history_plot_saving_path,showplot=showPlotHistory)
            if restoreBestModel:
                if verbose: print(f"{Color.green}Saving best model at epoch {self.best_epoch}{Color.end}")
                self = best_model
            if verbose: self.tot_time = time() - tot_t1; print(f"Total enlapsed time: {(self.tot_time):.2f} sec")
            return 

        except KeyboardInterrupt:
            if verbose: print(f"\n{Color.magenta}Interrupting training at epoch {self.best_epoch}...{Color.end}")
            self._plotHistory(save_path=history_plot_saving_path,showplot=showPlotHistory)
            if restoreBestModel:
                if verbose: print(f"{Color.green}Saving best model at epoch {self.best_epoch}{Color.end}")
                self = best_model
            if verbose: self.tot_time = time() - tot_t1; print(f"Total enlapsed time: {(self.tot_time):.2f} sec")
            return
        
    def k_fold_cross_validation(self, train_data:BNNDataset, k:int=5, **train_args):
        """
        Performs k-fold cross-validation on the BNN.

        Args:
            train_data (BNNDataset): The dataset to perform cross-validation on.
            k (int): Number of folds (default: 5).
            **train_args: Additional arguments for the train method.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Average training and validation losses across folds.
        """
        kf = KFold(n_splits=k)
        fold_train_losses = []
        fold_valid_losses = []

        for train_idx, valid_idx in kf.split(train_data):
            train_fold = torch.utils.data.Subset(train_data, train_idx)
            valid_fold = torch.utils.data.Subset(train_data, valid_idx)

            self.reset()  # Reset model parameters for each fold

            # Train the model on the current fold
            self.train(train_fold, valid_fold, **train_args)

            # Save training and validation losses for this fold
            fold_train_losses.append(self.train_loss_history)
            fold_valid_losses.append(self.valid_loss_history)

        # Aggregate losses across folds
        avg_train_losses = np.mean(np.array(fold_train_losses), axis=0)
        avg_valid_losses = np.mean(np.array(fold_valid_losses), axis=0)

        return avg_train_losses, avg_valid_losses
    
    def getAllParametersName(self):
        """
        Returns the names of all parameters in the model.

        Returns:
            list: List of parameter names.
        """
        par_name = []
        for name, _ in self.named_parameters():
            name_splitted = name.split(".")
            if len(name_splitted) > 2:
                name_splitted = [name_splitted[0] + "." + name_splitted[1], name_splitted[2]]
            par_name.append(name_splitted[-1])
        return list(set(par_name))
    
    def getAllLayersName(self):
        """
        Returns the names of all layers in the model.

        Returns:
            list: List of layer names.
        """
        par_name = []
        for name, _ in self.named_parameters():
            name_splitted = name.split(".")
            if len(name_splitted) > 2:
                name_splitted = [name_splitted[0] + "." + name_splitted[1], name_splitted[2]]
            par_name.append(name_splitted[0])
        return list(dict.fromkeys(par_name))

    def setModelGradients(self, 
                          requires_grad:bool, 
                          params:list = None,
                          layers:list = None):
        """
        Sets the requires_grad attribute for the specified parameters and layers.

        Args:
            requires_grad (bool): Whether the gradients should be computed.
            params (list): List of parameter names to set requires_grad for (default: None).
            layers (list): List of layer names to set requires_grad for (default: None).
        """
        if params is None: params = self.getAllParametersName()
        if layers is None: layers = self.getAllLayersName()
        for name, param in self.named_parameters():
            name_splitted = name.split(".")
            if len(name_splitted) > 2:
                name_splitted = [name_splitted[0] + "." + name_splitted[1], name_splitted[2]]
            if name_splitted[-1] in params and name_splitted[0] in layers:
                param.requires_grad = requires_grad
    
    def _plotHistory(self,save_path:str=None,showplot:bool=True):
        """
        Plots the training and validation loss history.

        Args:
            save_path (str): Path to save the loss history plot (default: None).
            showplot (bool): If True, display the plot (default: True).
        """
        lim_max = max([max(self.train_loss_history), max(self.valid_loss_history)])
        lim_min = min([min(self.train_loss_history), min(self.valid_loss_history)])
        half = (lim_max - lim_min) / 2

        x = range(len(self.train_loss_history))
        min_y_val = min(self.valid_loss_history)

        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        axs[0].plot(x, self.train_loss_history, color="orange", label="train loss")
        axs[0].plot(x, self.valid_loss_history, color="blue", label="valid loss")
        
        if self.earlyStopping:
            axs[0].axvline(self.best_epoch, linestyle="--", color="red")
        
        axs[1].plot(x, self.train_loss_history, color="orange",  label="train loss")
        axs[1].plot(x, self.valid_loss_history, color="blue", label="valid loss")

        if self.earlyStopping:
            axs[1].axvline(self.best_epoch, linestyle="--", color="red")
            axs[1].scatter(self.best_epoch, min_y_val, edgecolor='black', marker='o', s=40)
        span_x = int(self.actual_epoch * 30 / 100)
        span_y = min_y_val * 300 / 100
        axs[1].set_xlim([self.best_epoch-span_x, self.best_epoch+span_x])
        axs[1].set_ylim([min_y_val-span_y, min_y_val+span_y*2])
        plt.legend()
        plt.suptitle(f"{self.model_name}: Training loss") 
        if save_path is not None: plt.savefig(f"{save_path}history_{self.model_name}.pdf")
        if showplot: plt.show()

    def predict(self, x, attempt: int = 100, scaler: MMS = None, output_labels: list = None, returnDataFrame: bool = False):
        """
        Generates predictions with uncertainty estimates using the Bayesian model.

        Args:
            x (array-like): Input data for prediction.
            attempt (int): Number of stochastic forward passes for uncertainty estimation (default: 100).
            scaler (MMS): Scaler object for reversing normalization (default: None).
            output_labels (list): List of output labels (default: None).
            returnDataFrame (bool): If True, returns results as pandas DataFrames (default: False).

        Returns:
            Tuple: Mean and standard deviation of predictions, optionally as DataFrames.
        """
        means = []
        stds = []

        for data in x:
            # Ensure the input data is on the same device as the model
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32).to(self.device)
            else:
                data = data.to(self.device)
            
            predictions = np.zeros((attempt, len(output_labels)))

            for i in range(attempt):
                # Make sure the prediction is also done on the same device
                p = self(data).detach().cpu().numpy()
                if scaler is not None and output_labels is not None:
                    p = scaler.reverseArray(p, columns=output_labels)
                predictions[i] = p
            
            output_mean = np.mean(predictions, axis=0)
            output_stds = np.std(predictions, axis=0)

            means.append(output_mean)
            stds.append(output_stds)
        
        if returnDataFrame and output_labels is not None:
            return pd.DataFrame(means, columns=output_labels), pd.DataFrame(stds, columns=output_labels)
        else:
            return means, stds
    
    def _removeZeros(self, gt: np.ndarray, pred:np.ndarray = None, offset:float = 1.5):
        """
        Adds a small offset to the ground truth and predictions to avoid zero values.

        Args:
            gt (np.ndarray): Ground truth array.
            pred (np.ndarray): Predicted values array (default: None).
            offset (float): Value to add to avoid zero values (default: 1.5).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Adjusted ground truth and prediction arrays.
        """
        if pred is None: pred = gt
        assert len(gt) == len(pred), "Array length mismatch"
        for i in range(len(gt)):
            gt[i] += offset; pred[i] += offset
        return gt, pred

    def _smape(self, gt: np.ndarray, pred: np.ndarray):
        """
        Computes the Symmetric Mean Absolute Percentage Error (SMAPE) between the ground truth and predictions.

        Args:
            gt (np.ndarray): Ground truth array.
            pred (np.ndarray): Predicted values array.

        Returns:
            list: SMAPE values as a percentage.
        """
        abs_diff = np.abs(pred - gt)
        abs_sum = np.abs(pred) + np.abs(gt)
        gt_modified = gt.copy()
        gt_modified[gt_modified == 0] = abs_diff[gt_modified == 0] * 100
        return list((abs_diff / abs_sum) * 100)
    
    def _mae(self, gt: np.ndarray, pred: np.ndarray):
        """
        Computes the Mean Absolute Error (MAE) between the ground truth and predictions.

        Args:
            gt (np.ndarray): Ground truth array.
            pred (np.ndarray): Predicted values array.

        Returns:
            np.ndarray: MAE values.
        """
        return np.abs(gt - pred)

    def testModel(self, test_set:BNNDataset, scaler:MMS, output_labels:list, attempt:int = 10, skip:list = []):
        """
        Tests the model on a test dataset and returns the prediction errors.

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
            if i in skip: continue
            x,y = data
            y = scaler.reverseArray(y,columns=output_labels)
            preds, stds = self.predict([x], scaler=scaler, output_labels=output_labels, attempt=attempt)
            preds = np.array(preds[0])
            stds = stds[0]
            #y, preds = self._removeZeros(y, preds)
            errors.append(self._mae(y, preds))
            cov.append(stds)
        return errors, cov
    
    def save(self, path:str, subfolder:str = "AIModels/", save_model_resume:bool = True):
        """
        Saves the model state to a file.

        Args:
            path (str): The file path to save the model.
            subfolder (str): Subfolder to save the model in (default: "AIModels/").
            save_model_resume (bool): If True, prints the model summary (default: True).
        """
        try: os.makedirs(subfolder)
        except: pass
        torch.save(self.state_dict(), path)
        if save_model_resume:
            self.__str__()
        print(f"{Color.green}[{Color.end}{self.model_name}{Color.green}] >> Model Saved{Color.end}")

    def load(self, path: str, device:str="cpu"):
        """
        Loads the model state from a file.

        Args:
            path (str): The file path to load the model from.
            device (str): The device to load the model onto (default: "cpu").
        """
        self.load_state_dict(torch.load(path))
        print(f"{Color.green}[{Color.end}{self.model_name}{Color.green}] >> Model Loaded on {Color.blue}{device}{Color.end} ({path})")


def test_multiple_models(models_to_test: List[BNN], test_set: BNNDataset, scaler: MMS, output_labels: list, attempt: int = 100, path: str = "test_results.csv", save_test_results: bool = True) -> pd.DataFrame:
    """
    Tests multiple BNN models on the same test set and logs their performance.

    Args:
        models_to_test (List[BNN]): List of models to test.
        test_set (BNNDataset): The test dataset.
        scaler (MMS): Scaler object to reverse normalization.
        output_labels (list): List of output labels.
        attempt (int): Number of stochastic forward passes for uncertainty estimation (default: 100).
        path (str): File path to save the test results (default: "test_results.csv").
        save_test_results (bool): If True, save the test results to a CSV file (default: True).

    Returns:
        pd.DataFrame: DataFrame containing the error metrics for each model.
    """
    error_data = {"Model Name": []}
    for out_val in output_labels:
        error_data[out_val + " ERR%"] = []
        error_data[out_val + " STD%"] = []
        #error_data[out_val + " Max/Min Err %"] = []
        #error_data[out_val + " Max/Min CoV %"] = []
    error_data["ERR_TOT%"] = []
    error_data["STD_TOT%"] = []
    for i, model in enumerate(models_to_test):
        error_data["Model Name"].append(model.model_name)
        errors, cov = model.testModel(test_set=test_set,
                                scaler=scaler,
                                output_labels=output_labels,
                                attempt=attempt)
        total_error = []
        total_std =  []
        for out_val in range(len(output_labels)):
            error_on_output_i = np.array([t[out_val] for t in errors])
            cov_on_output_i = np.array([t[out_val] for t in cov])
            cov_on_output_i = (cov_on_output_i / scaler.offset[output_labels[out_val]]) * 100
            error = np.mean(error_on_output_i) / scaler.offset[output_labels[out_val]] * 100
            total_error.append(error)
            total_std.append(cov_on_output_i)
            error_data[output_labels[out_val] + " ERR%"].append(f"{error:.2f}")
            error_data[output_labels[out_val] + " STD%"].append(f"{np.mean(cov_on_output_i):.2f}")
            #error_data[output_labels[out_val] + " Max/Min Err %"].append(f"{np.max(error_on_output_i):.2f} / {np.min(error_on_output_i):.2f}")
            #error_data[output_labels[out_val] + " Max/Min CoV %"].append(f"{np.max(cov_on_output_i):.2f} / {np.min(cov_on_output_i):.2f}")
        error_data["ERR_TOT%"].append(np.sqrt(np.mean(np.array(total_error)**2)))
        error_data["STD_TOT%"].append(np.sqrt(np.mean(np.array(total_std)**2)))
        
    error_table = pd.DataFrame(error_data)
    if save_test_results: error_table.to_csv(path, index=False)
    return error_table