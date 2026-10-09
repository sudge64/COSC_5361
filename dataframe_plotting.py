import sys
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def set_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--data",
    type=str,
    help="Path to data frame stored in csv"
  )
  parser.add_argument(
    "--plot",
    action="store_true",
    help="Plot data?"
  )
  return parser.parse_args()

def read_dataframes(dir):
  dfs = {}
  saved_dirs = {}

  for root, _dirs, files in os.walk(dir):
      for file in files:
          if file.endswith('.csv'):
              csv = Path(root) / file
              key = csv.stem
              dfs[key] = pd.read_csv(csv)
              saved_dirs[key] = csv.parent
  return dfs, saved_dirs

def show_plot(name, df, out):
  path = out / f"{name}.png"

  fig, ax = plt.subplots(figsize=(9, 5))

  df.plot(
    x="Epoch",
    y=["Train Loss", "Val Loss"],
    kind="line",
    marker="o",
    xlabel="Epoch",
    ylabel="Loss",
    title="Epoch vs. Loss",
    ax=ax,
  )

  fig.savefig(path, dpi=300, bbox_inches="tight")
  print(f"Saved Figure! -> {path}")

  plt.show()
  plt.close(fig)

def main():
  args = set_args()

  if args.data:
    dfs, dirs = read_dataframes(args.data)
    recorded_values = []
    for key, df in dfs.items():
      max_value = df['Val Acc'].max()
      mean_value = df['Val Acc'].mean()
      recorded_values.append({"Name": key, "Max": max_value, "Mean": mean_value})
      if args.plot:
        show_plot(key, df, dirs[key])
  else:
    print("FLAGRANT SYSTEM ERROR: Provide the checkpoints directory.")
    sys.exit(1)
  best_algorithms = pd.DataFrame(recorded_values)
  pd.set_option('display.max_colwidth', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_rows', None)
  print("Max: \n", best_algorithms.sort_values(by='Max', ascending=True))
  print("Mean: \n", best_algorithms.sort_values(by='Mean', ascending=True))
  print("Max: \n", best_algorithms.nlargest(3, 'Max'))
  print("Mean: \n", best_algorithms.nlargest(3, 'Mean'))

if __name__ == "__main__":
  main()

