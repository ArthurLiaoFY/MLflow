# %%

from configparser import ConfigParser

import numpy as np
import pandas as pd
import seaborn as sns

from ml_projects.wafer_overlay.load_data import load_data
from ml_projects.wafer_overlay.plot_fn import plot_boxplot, plot_scatter, plot_wafer

# %%

config = ConfigParser()
config.read("wafer_overlay.ini")
# %%
df = load_data(
    data_file_path=config["wafer_overlay"]["data_file_path"],
    drop_columns=[
        "A1_Space.Bot.CD",
        "A1_Line.Bot.CD",
        "B3_Tx05",
        "B3_Tx03",
        "C6_Space.Bot.CD",
        "D8_Line.Bot.CD",
    ],
)
# %%
df.columns
# %%
wafer_point_cnt = df.groupby("wafer_id")["ChuckId"].count()
print(np.unique(wafer_point_cnt.sort_values(), return_counts=True))
# %%
[
    "A2_EQUIP_ID",
    "A2_Chm",
    "A2_HR",
    "A2_ESC",
    "B4_EQUIP_ID",
    "B4_Chm",
    "B4_HR",
    "A5_EQUIP_ID",
    "A5_Chm",
    "A5_ESC",
    "D7_EQUIP_ID",
    "D7_PM",
    "D7_HR",
    "E9_EQUIP_ID",
    "E9_PM",
    "E9_HR",
]
# %%
plot_scatter(
    df=df,
    x="A2_HR",
    y="A2_ESC",
    group_factor=["A2_EQUIP_ID", "A2_Chm"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_scatter(
    df=df,
    x="WaferStartTime",
    y="A2_ESC",
    group_factor=["A2_EQUIP_ID", "A2_Chm"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_scatter(
    df=df,
    x="WaferStartTime",
    y="A2_HR",
    group_factor=["A2_EQUIP_ID", "A2_Chm"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_scatter(
    df=df,
    x="WaferStartTime",
    y="B4_HR",
    group_factor=["B4_EQUIP_ID", "B4_Chm"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_scatter(
    df=df,
    x="WaferStartTime",
    y="A5_ESC",
    group_factor=["A5_EQUIP_ID", "A5_Chm"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_scatter(
    df=df,
    x="WaferStartTime",
    y="D7_HR",
    group_factor=["D7_EQUIP_ID", "D7_PM"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_scatter(
    df=df,
    x="WaferStartTime",
    y="E9_HR",
    group_factor=["E9_EQUIP_ID", "E9_PM"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
# %%
plot_boxplot(
    df=df,
    x="A2_HR",
    group_factor=["A2_EQUIP_ID", "A2_Chm"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_boxplot(
    df=df,
    x="A2_ESC",
    group_factor=["A2_EQUIP_ID", "A2_Chm"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_boxplot(
    df=df,
    x="B4_HR",
    group_factor=["B4_EQUIP_ID", "B4_Chm"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_boxplot(
    df=df,
    x="A5_ESC",
    group_factor=["A5_EQUIP_ID", "A5_Chm"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_boxplot(
    df=df,
    x="D7_HR",
    group_factor=["D7_EQUIP_ID", "D7_PM"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
plot_boxplot(
    df=df,
    x="E9_HR",
    group_factor=["E9_EQUIP_ID", "E9_PM"],
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
# %%
df.drop_duplicates(
    subset=[
        "wafer_id",
        "A2_EQUIP_ID",
        "A2_Chm",
        # "B4_EQUIP_ID",
        # "B4_Chm",
        # "A5_EQUIP_ID",
        # "A5_Chm",
        # "D7_EQUIP_ID",
        # "D7_PM",
        # "E9_EQUIP_ID",
        # "E9_PM",
    ]
).groupby(
    [
        "A2_EQUIP_ID",
        "A2_Chm",
        # "B4_EQUIP_ID",
        # "B4_Chm",
        # "A5_EQUIP_ID",
        # "A5_Chm",
        # "D7_EQUIP_ID",
        # "D7_PM",
        # "E9_EQUIP_ID",
        # "E9_PM",
    ]
)[
    "wafer_id"
].count()

# %%
df.isna().sum()
# %%
na_cols = df.columns[df.isna().sum() > 0]
print(na_cols)
# %%

# %%
filled_df = df.groupby("wafer_id").bfill().ffill()
filled_df["wafer_id"] = df["wafer_id"]
# %%
for i in filled_df["wafer_id"].unique():
    if len(filled_df.loc[filled_df["wafer_id"] == i, na_cols].drop_duplicates()) > 1:
        print(i)
        print(filled_df.loc[filled_df["wafer_id"] == i, na_cols].drop_duplicates())
# %%
df.head()
# %%
df.tail()
# %%
df.columns

# %%
sub_df = df.loc[df["wafer_id"].isin(wafer_point_cnt[wafer_point_cnt == 587].index), :]
# %%
sns.heatmap(sub_df.isna())
# %%
df.drop_duplicates(subset=["wafer_id", "WaferStartTime"]).groupby("wafer_id")[
    "ChuckId"
].count().sort_values()

# %%
plot_wafer(
    df=df,
    wafer_id="W_77",
    plot_file_path=config["wafer_overlay"]["plot_file_path"],
)
# %%
df.loc[
    df["wafer_id"] == "W_77",
    ["wafer_id", "ChuckId", "WaferStartTime", "posx", "posy", "OVL_Y"],
].drop_duplicates()
# %%
df.loc[
    df["wafer_id"] == "W_77",
    [
        # A2
        "A2_EQUIP_ID",
        "A2_Chm",
        "A2_HR",
        "A2_ESC",
        # A5
        "A5_EQUIP_ID",
        "A5_Chm",
        "A5_ESC",
        # B4
        "B4_EQUIP_ID",
        "B4_Chm",
        "B4_HR",
        # D7
        "D7_EQUIP_ID",
        "D7_PM",
        "D7_HR",
        # E9
        "E9_EQUIP_ID",
        "E9_PM",
        "E9_HR",
    ],
].drop_duplicates()
# %%
sns.heatmap(
    df.loc[
        df["wafer_id"] == "W_3",
        [
            # A2
            "A2_EQUIP_ID",
            "A2_Chm",
            "A2_HR",
            "A2_ESC",
            # A5
            "A5_EQUIP_ID",
            "A5_Chm",
            "A5_ESC",
            # B4
            "B4_EQUIP_ID",
            "B4_Chm",
            "B4_HR",
            # D7
            "D7_EQUIP_ID",
            "D7_PM",
            "D7_HR",
            # E9
            "E9_EQUIP_ID",
            "E9_PM",
            "E9_HR",
        ],
    ].isna()
)
# %%
df.columns[["Space.Bot.CD" in col for col in df.columns]]
# %%
sns.heatmap(
    df[
        [
            "A1_Space.Bot.CD",
            "A1_Line.Bot.CD",
            "C6_Space.Bot.CD",
            "D8_Line.Bot.CD",
            "B3_Tx05",
            "B3_Tx03",
        ]
    ].isna()
)
# %%
sns.heatmap(
    df[
        [
            "A1_Space.Bot.CD",
            "A1_Line.Bot.CD",
            "C6_Space.Bot.CD",
            "D8_Line.Bot.CD",
            "B3_Tx05",
            "B3_Tx03",
        ]
    ]
    .sum(axis=1)
    .to_numpy()[np.newaxis, :]
)
# %%
df[
    [
        "E9_EQUIP_ID",
        "E9_PM",
    ]
]
# %%
df["E9_PM"].unique()
# %%
