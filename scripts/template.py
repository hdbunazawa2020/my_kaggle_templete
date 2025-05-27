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
from omegaconf import DictConfig, OmegaConf

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
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")

# ===================================
# utils
# ===================================


# ===================================
# main
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
        exp = "000_filename__debug" # TODO: ファイルの連番を入れる
    # set config

    # make savedir
    savedir = output_dir / exp
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir / "yaml", exist_ok=True)

    # ==================
    # Save file
    # ==================
    # -- 必要なファイル保存 --

    # YAMLとして保存
    output_path = Path(savedir/"yaml"/"config.yaml")
    with open(output_path, "w") as f:
        OmegaConf.save(config=config, f=f.name)
    print(f"Config saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
