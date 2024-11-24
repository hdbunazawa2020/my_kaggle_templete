# from Ipython.display import display
import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from IPython.core.display import display

def sep(word, num=80):
    print("="*num); print(word); print("="*80)

def show_df(df, num=3, showtail=False):
    """
    データの概要を出力するメソッド
    """
    print(df.shape)
    display(df.head(num))
    if showtail:
        display(df.tail(num))

def glob_walk(root: Path, glob_str: str) -> list:
    """データpathのリストを作る関数
    Args:
        root (Path): データフォルダが入っているpath
        glob_str (str): globでマッチする文字列
    Returns:
        list[Path]: データpathのリスト
    """
    path = Path(root)
    walker = sorted(list(path.glob(glob_str)))

    return walker

import random

import numpy as np
import torch

def seed_everything(seed, GPU=False):
    """
    Seeds basic parameters for reproductibility of results.
    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if GPU:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_datapaths(input_dir:str, glob_str:str):
    """データのPathリストを取得
    Args:
        input_dir (str): データディレクトリのパス
        glob_str (str): 拡張子
    Returns:
        data_paths(list): データのPathリスト
    """
    # CAN300
    data_paths = glob_walk(input_dir, glob_str)
    return data_paths

def check_allnullcolumns(data_paths):
    """全てNullのカラムを確認し、そうでないカラムのリストを返す
    Args:
        data_paths (list): データPathのリスト
    Returns:
        cols: 全てがNullではないカラムのリスト
    """
    df = pl.read_parquet(data_paths[0])
    columns = df.columns
    # columns.remove("vehicle_model")
    print(columns)
    print()
    cols = []
    for col in columns:
        if (df[col].null_count() != len(df)) and (df[col].min() != df[col].max()):
            cols.append(col)

    cols += ["vehicle_model"]
    print(f"ALL Not NULL columns: {cols}")
    return cols

def normalize(y):
    """Normalize data 
    Args:
        y : data
        
    Returns:
        y : normalized data
    """
    mean = y[:,0].mean().item()
    std  = y[:,0].std().item()
    y[:,0] = (y[:,0]-mean)/(std+1e-16)
    mean = y[:,1].mean().item()
    std  = y[:,1].std().item()
    y[:,1] = (y[:,1]-mean)/(std+1e-16)
    return y

def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df

def to_pickle(filename, obj):
    with open(filename, mode="wb") as f:
        pickle.dump(obj, f)

def unpickle(filename):
    with open(filename, mode="rb") as fo:
        p = pickle.load(fo)
    return p