# """
# Make the figure using the results from evaluate_rotation.py
# """
import matplotlib.pyplot as plt
import torch 
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")

# Load results 
results = torch.load("./results/rotation.pt")

df = pd.DataFrame(results, columns=["Direction", "Class", "MMD-CD"])
df = df.astype({"Direction": int, "Class":int, "MMD-CD": float})

print(df.head())


# Plot the responses for different events and regions
line_plot = sns.lineplot(x="Direction", y="MMD-CD",
             hue="Class",
             data=df)
sns.move_legend(line_plot, "upper left", bbox_to_anchor=(1, 1))
fig = line_plot.get_figure()
fig.savefig("./figures/rotation.png", bbox_inches='tight',dpi=500)
