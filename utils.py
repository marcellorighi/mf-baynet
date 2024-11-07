import numpy as np
from time import time
import tkinter as tk
from tkinter import messagebox

class PrCol:
    def __init__(self):
        self.black = "\033[30m"
        self.red = "\033[91m"
        self.green = "\033[92m"
        self.yellow = "\033[93m"
        self.blue = "\033[94m"
        self.magenta = "\033[95m"
        self.cyan = "\033[96m"
        self.white = "\033[97m"
        self.bright_black = "\033[90m"
        self.bright_red = "\033[91;1m"
        self.bright_green = "\033[92;1m"
        self.bright_yellow = "\033[93;1m"
        self.bright_blue = "\033[94;1m"
        self.bright_magenta = "\033[95;1m"
        self.bright_cyan = "\033[96;1m"
        self.bright_white = "\033[97;1m"
        self.end = "\033[0m"

Color = PrCol()

class CircularBuffer:
    def __init__(self, dimension):
        self.dimension = dimension
        self.buffer = []

    def add_element(self, val):
        if len(self.buffer) < self.dimension:
            self.buffer.append(val)
        else:
            self.buffer = self.buffer[1:] + [val]
    
    def get_mean(self):
        if len(self.buffer) == 0:
            return np.nan
        return np.mean(self.buffer)
    

class LoadBar():
    t1 = np.inf

    def __init__(self) -> None:
        self.timeBuffer = CircularBuffer(50)

    def getLoadBar(self, actual, total, startchar="|", loadchar=f"{Color.green}█{Color.end}", uncompletedchar=f"{Color.black}█{Color.end}", endchar="|", length=20):
        percent_completed = int(actual / total * 100)
        completed_length = int(length * actual / total)
        uncompleted_length = length - completed_length
        load_bar = (
            startchar +
            loadchar * completed_length +
            uncompletedchar * uncompleted_length +
            endchar
        )
        return load_bar

    def loadBar_ij(self, i, end_i, j, end_j):
        eta = self.timeBuffer.get_mean() * (end_i - i)
        return f'{self.getLoadBar(i + 1,end_i)}{self.getLoadBar(j + 1, end_j, startchar="",loadchar=f"{Color.yellow}█{Color.end}",length=7)} ' + \
                            f'({i + 1}/{end_i}) ({self.timeBuffer.get_mean():.2f}s avg time) ' + \
                            f'ETA: {seconds_to_hhmmss(eta)} - '
    
    def loadBar(self,i,end_i):
        eta = self.timeBuffer.get_mean() * (end_i - i)
        return f'{self.getLoadBar(i + 1,end_i)}' + \
                            f'({i + 1}/{end_i}) ({self.timeBuffer.get_mean():.2f}s avg time) ' + \
                            f'ETA: {seconds_to_hhmmss(eta)} - '

    def tick(self):
        self.t1 = time()

    def tock(self):
        self.timeBuffer.add_element(time() - self.t1)

def darken(hex_color, factor=0.5):
    # Ensure the factor is within the valid range [0, 1]
    factor = max(0, min(1, factor))

    # Convert hex to RGB
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)

    # Darken each RGB component
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))

    # Convert back to hex and return
    darkened_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)
    return darkened_hex

def seconds_to_hhmmss(secs, minFormat=True):
    try:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        mm = m + 60 * h
        ss = s + 60 * mm
        if minFormat: 
            if mm==0 and h==0: return "~{:02d} sec".format((ss // 10 + 1) * 10)
            elif h==0: return "~{:02d} min".format(mm+1)
            else: return "~{:02d} h, {:02d} m".format(h,(mm+1) % 60)
        else: return "{:02d}:{:02d}:{:02d}".format(h, m, s)
    except ValueError:
        return np.nan
    
def mpe(predictions, ground_truth):
    errors = np.abs(predictions - ground_truth) / np.abs(ground_truth)
    mean_error = np.mean(errors) * 100
    return mean_error

def show_warning(message):
    root = tk.Tk()
    root.withdraw()
    top = tk.Toplevel(root)
    top.withdraw()
    top.attributes('-topmost', True)
    top.grab_set()
    messagebox.showwarning("Warning", message, parent=top)
    top.grab_release() 
    top.destroy() 

def show_continue_cancel(message):
    root = tk.Tk()
    root.withdraw()
    top = tk.Toplevel(root)
    top.withdraw() 
    top.attributes('-topmost', True)  
    top.grab_set()  
    response = messagebox.askquestion("Warning", message, icon='warning', parent=top)
    top.grab_release() 
    top.destroy() 
    if response == 'no':
        return False
    return True

def create_data_dump(model_settings,dataset_settings,training_settings,bnn_lf_model,bnn_mf_model,bnn_df_model,ck_df_model):
    data_dump = {
        "MODEL_0_NAME": model_settings["MODEL_NAME"],
        "DATASET_PATH": dataset_settings["DATASET_LOCATION"],
        "DATASET_DESCRIPTION": dataset_settings["DATASET_DESCRIPTION"],
        "MODEL_HYPERPARAMETER_MU": model_settings["MU"],
        "MODEL_HYPERPARAMETER_MU_MF": model_settings["MU_MF"],
        "MODEL_HYPERPARAMETER_STD": model_settings["STD"],
        "MODEL_HYPERPARAMETER_STD_MF": model_settings["STD_MF"],
        "MODEL_HYPERPARAMETER_UNITS": model_settings["UNITS"],
        "MODEL_HYPERPARAMETER_UNITS_MF": model_settings["UNITS_MF"],
        "MODEL_DEVICE": model_settings["DEVICE"],
        "MODEL_SEED": model_settings["SEED"],
        "MODEL_HYPERPARAMETER_PATIENCE_LF": training_settings["PATIENCE_LF"],
        "MODEL_HYPERPARAMETER_N_EPOCHS_LF": training_settings["N_EPOCHS_LF"],
        "MODEL_HYPERPARAMETER_BATCH_SIZE_LF": training_settings["BATCH_SIZE_LF"],
        "MODEL_HYPERPARAMETER_LR_LF": training_settings["LR_LF"],
        "FLAG_TRAIN_LF": training_settings["TRAIN_LF"],
        "MODEL_HYPERPARAMETER_PATIENCE_MF": training_settings["PATIENCE_MF"],
        "MODEL_HYPERPARAMETER_N_EPOCHS_MF": training_settings["N_EPOCHS_MF"],
        "MODEL_HYPERPARAMETER_BATCH_SIZE_MF": training_settings["BATCH_SIZE_MF"],
        "MODEL_HYPERPARAMETER_LR_MF": training_settings["LR_MF"],
        "FLAG_TRAIN_MF": training_settings["TRAIN_MF"],
        "MODEL_HYPERPARAMETER_PATIENCE_TL": training_settings["PATIENCE_TL"],
        "MODEL_HYPERPARAMETER_N_EPOCHS_TL": training_settings["N_EPOCHS_TL"],
        "MODEL_HYPERPARAMETER_BATCH_SIZE_TL": training_settings["BATCH_SIZE_TL"],
        "MODEL_HYPERPARAMETER_LR_TL": training_settings["LR_TL"],
        "FLAG_TRAIN_TL": training_settings["TRAIN_TL"],
        "FLAG_TRAIN_CK": training_settings["TRAIN_CK"],
        "MODEL_HYPERPARAMETER_N_LAYER_TO_UNFREEZE": training_settings["N_LAYER_TO_UNFREEZE"],
    }
    if training_settings["TRAIN_LF"]: 
        data_dump["MODEL_TRAININFO_TIME_TRAIN_LF"] = bnn_lf_model.tot_time
        data_dump["MODEL_TRAININFO_BEST_EPOCH_LF"] = bnn_lf_model.best_epoch
    if training_settings["TRAIN_MF"]: 
        data_dump["MODEL_TRAININFO_TIME_TRAIN_MF"] = bnn_mf_model.tot_time
        data_dump["MODEL_TRAININFO_BEST_EPOCH_MF"] = bnn_mf_model.best_epoch
    if training_settings["TRAIN_TL"]:
        data_dump["MODEL_TRAININFO_TIME_TRAIN_TL"] = bnn_df_model.tot_time
        data_dump["MODEL_TRAININFO_BEST_EPOCH_TL"] = bnn_df_model.best_epoch
    if training_settings["TRAIN_CK"]:
        data_dump["MODEL_TRAININFO_TIME_TRAIN_CK"] = ck_df_model.tot_time
    return data_dump