# AUTHOR: Andrea Vaiuso
VERSION = "1.5"

import pandas as pd
import numpy as np
import pickle
import torch as pt
import os

class MinMaxScaler():
    def __init__(self, dataframe: pd.DataFrame = None, interval: tuple = (0, 1)) -> None:
        if dataframe is None: return
        self.scalers = {}
        self.offset = {}
        self.interval = interval
        for c in dataframe.columns:
            self.scalers[c] = (min(dataframe[c]), max(dataframe[c]))
            self.offset[c] = abs(max(dataframe[c]) - min(dataframe[c]))
        self.column_names = list(dataframe.columns)

    def scaleDataframe(self, dataframe: pd.DataFrame):
        assert all(col in  list(self.scalers.keys()) for col in list(dataframe.columns)), "Column names mismatch"
        scaled_df = dataframe.copy()
        for col in dataframe.columns:
            min_val, max_val = self.scalers[col]
            scaled_df[col] = (dataframe[col] - min_val) / (max_val - min_val) * (self.interval[1] - self.interval[0]) + self.interval[0]
        return scaled_df

    def reverseDataframe(self, normalized_dataframe: pd.DataFrame):
        assert set(normalized_dataframe.columns) == set(self.scalers.keys()), "Column names mismatch"
        reversed_df = normalized_dataframe.copy()
        for col in self.column_names:
            min_val, max_val = self.scalers[col]
            reversed_df[col] = (normalized_dataframe[col] - self.interval[0]) * (max_val - min_val) / (self.interval[1] - self.interval[0]) + min_val
        return reversed_df

    def scaleArray(self, array, columns: list = None):
        array = self._precomputeArray(array)
        if array is None: raise ValueError("Input array not valid")
        assert len(array.shape) == 1 or (len(array.shape) == 2 and array.shape[0] == 1), "Invalid array shape"
        if columns is None:
            scaled_array = (array - np.array([self.scalers[col][0] for col in self.column_names])) / np.array([self.scalers[col][1] - self.scalers[col][0] for col in self.column_names]) * (self.interval[1] - self.interval[0]) + self.interval[0]
        else:
            assert len(array) == len(columns), "Array size and columns size mismatch"
            scaled_array = np.zeros_like(array)
            for i, col in enumerate(columns):
                min_val, max_val = self.scalers[col]
                scaled_array[i] = (array[i] - min_val) / (max_val - min_val) * (self.interval[1] - self.interval[0]) + self.interval[0]
        return scaled_array

    def reverseArray(self, normalized_array, columns: list = None):
        normalized_array = self._precomputeArray(normalized_array)
        if normalized_array is None: raise ValueError("Input array not valid")
        if columns is None:
            reversed_array = (normalized_array - self.interval[0]) * np.array([self.scalers[col][1] - self.scalers[col][0] for col in self.column_names]) / (self.interval[1] - self.interval[0]) + np.array([self.scalers[col][0] for col in self.column_names])
        else:
            assert len(normalized_array) == len(columns), "Array size and columns size mismatch"
            reversed_array = np.zeros_like(normalized_array)
            for i, col in enumerate(columns):
                min_val, max_val = self.scalers[col]
                reversed_array[i] = (normalized_array[i] - self.interval[0]) * (max_val - min_val) / (self.interval[1] - self.interval[0]) + min_val
        return reversed_array
    
    def scaleMatrix(self, matrix: np.ndarray, columns: list = None):
        scaled_matrix = None
        for i in range(len(matrix)):
            if scaled_matrix is None:
                scaled_matrix = self.scaleArray(matrix[i], columns)
            else: scaled_matrix = np.vstack((scaled_matrix, self.scaleArray(matrix[i], columns)))
        return scaled_matrix

    def reverseMatrix(self, normalized_matrix: np.ndarray, columns: list = None):
        reversed_matrix = None
        for i in range(len(normalized_matrix)):
            if reversed_matrix is None:
                reversed_matrix = self.reverseArray(normalized_matrix[i], columns)
            else: reversed_matrix = np.vstack((reversed_matrix, self.reverseArray(normalized_matrix[i], columns)))
        return reversed_matrix
    
    def save(self, filename:str, path:str="."):
        try: os.makedirs(path)
        except FileExistsError as e: pass
        with open(path+"/"+filename, 'wb') as file:
            pickle.dump((self.scalers, self.interval, self.offset), file)

    def load(self, filename:str):
        with open(filename, 'rb') as file:
            self.scalers, self.interval, self.offset = pickle.load(file)

    def _precomputeArray(self, array):
        if isinstance(array, np.ndarray): return array.astype(np.float32)
        elif isinstance(array, pt.Tensor): return array.detach().cpu().numpy()
        elif isinstance(array, list): return np.array(array, dtype=np.float32)
        else: return None
