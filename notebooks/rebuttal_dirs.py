import glob
import json

import numpy as np
import pandas as pd

# |%%--%%| <53amjXKjlp|7QPjohcs3S>

files = glob.glob("results/encoder_new_numdir**/**.json")

results = []
for file in files:
    with open(file) as f:
        results.extend(json.load(f))

df = pd.DataFrame(results).drop("normalized", axis=1)
df[["model_name", "x", "y", "num_dirs", "dataset"]] = df["model"].str.split(
    "_", expand=True
)


df["MMD-CD"] *= 1e4
df["MMD-EMD"] *= 1e3
df = (
    df[["dataset", "num_pts", "MMD-CD", "MMD-EMD"]]
    .set_index(["dataset", "num_pts"])
    .sort_index()
)
df = df.unstack(level=0).reorder_levels([1, 0], axis=1).sort_index(axis=1)
# fmt: on

df_res

# |%%--%%| <7QPjohcs3S|D5NZX1VWkr>

# df.reset_index(col_level=1).to_markdown()

# df.to_markdown("rebuttal_lower.md")
