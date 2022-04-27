"""Codes for regressionion model"""

# packages
import pandas as pd

# Load the data
df = pd.read_feather("../../../data/mtg.feather")

# Remove NAs
df = df.dropna(subset = ["edhrec_rank"]).reset_index(drop=True)
