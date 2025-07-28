import glob
import json

import numpy as np
import pandas as pd

# |%%--%%| <53amjXKjlp|7QPjohcs3S>

files = glob.glob("results/encoder_new_large**/**.json")

results = []
for file in files:
    with open(file) as f:
        results.extend(json.load(f))

# fmt: off
df = (
        pd.DataFrame(results)
        .drop("normalized", axis=1)
)
# df[["model_name","x", 'y', "num_pts","dataset"]] = df["model"].str.split('_',expand=True)
#
df["MMD-CD"] *= 1e4
df["MMD-EMD"] *= 1e3
#
# df = df[["dataset",'num_pts','MMD-CD',"MMD-EMD"]].set_index(['dataset','num_pts']).sort_index()
df.to_markdown("rebuttal_larger.md")
# fmt: on
