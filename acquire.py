import pandas as pd
import numpy as np
import os

def get_map_data():
    filename = "match_map_stats.csv"
    return pd.read_csv(filename)
