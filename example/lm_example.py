import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import plotly.graph_objects as go

from ml_models.linear_models.models.linear_model import LinearModel

savings = pd.read_csv("/Users/wr80340/WorkSpace/mlflow/data/savings.csv").set_index(
    "Unnamed: 0"
)
X = savings[["p15", "p75", "inc", "gro"]].to_numpy()
y = savings[["sav"]].to_numpy()
lm = LinearModel()
lm.fit(X, y)
# %%
lm.plot_residual(index_name=savings.index.to_list())
lm.plot_leverage(index_name=savings.index.to_list())
lm.plot_studentized_residual(index_name=savings.index.to_list())
lm.plot_cook_statistics(index_name=savings.index.to_list())
lm.plot_jackknife_residual(index_name=savings.index.to_list())
# %%
