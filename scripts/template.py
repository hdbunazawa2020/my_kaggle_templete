import os, gc, glob, pickle, warnings
import random, math
import joblib, pickle, itertools
from pathlib import Path

import numpy as np
import scipy as sp
import polars as pl
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import hydra
from omegaconf import DictConfig

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import warnings
# warnings.filterwarnings('ignore')

# original
import sys
sys.path.append(r"..")
import utils
from utils.data import *

from datetime import datetime
date = datetime.strptime("20241116", "%Y%m%d").date().strftime("%Y%m%d")
print(f"TODAY is {date}")

# ===================================
# utils
# ===================================
# TODO: config_pathをこのスクリプトからの相対パスにする
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """description
    Args:
        cfg (DictConf): config
    """
    # from config
    input_dir  = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    # get config of this file
    config = cfg["template"]
    exp = config["exp"]
    debug = config["debug"]
    if debug:
        exp = "xxx_debug" # TODO: ファイルの連番を入れる
    # set config

    # make savedir
    savedir = output_dir / exp
    os.makedirs(savedir, exist_ok=True)

    # ==================
    # Save file
    # ==================

if __name__ == "__main__":
    main()
