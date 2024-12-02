import torch
import numpy as np
import pandas as pd
import json

############################################################
### Create tables
############################################################

df = pd.read_json("./results/encoder_mnist/encoder_mnist.json")
df = df.drop("fscore", axis=1)
df = df.groupby(by=["normalized", "model"]).agg(["mean", "std"])
print(df)
