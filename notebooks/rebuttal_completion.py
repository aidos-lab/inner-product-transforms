import glob
import json

import numpy as np
import pandas as pd

# |%%--%%| <53amjXKjlp|7QPjohcs3S>

files = glob.glob("results/encoder_completion**/**.json")

results = []
for file in files:
    with open(file) as f:
        results.extend(json.load(f))

# fmt: off
df = (
        pd.DataFrame(results)
        .drop("normalized", axis=1)
)
df[["model_name", 'y', "dataset"]] = df["model"].str.split('_',expand=True)
#
df["MMD-CD"] *= 1e4
df["MMD-EMD"] *= 1e3
#

df = (
    df[["dataset",'MMD-CD',"MMD-EMD"]]
    .set_index(['dataset'])
    .groupby(['dataset'])
    .agg(["mean"])
    .reorder_levels([1,0],axis=1)
    .sort_index()
)
# df.to_markdown("rebuttal_completion.md",format="rst")
df.to_markdown(tablefmt="rst")
# df
# fmt: on
