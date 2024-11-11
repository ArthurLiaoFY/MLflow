import os

import pandas as pd

code_to_city_map = {
    "a": "臺北市",
    "b": "臺中市",
    "c": "基隆市",
    "d": "臺南市",
    "e": "高雄市",
    "f": "新北市",
    "g": "宜蘭縣",
    "h": "桃園市",
    "i": "嘉義市",
    "j": "新竹縣",
    "k": "苗栗縣",
    "m": "南投縣",
    "n": "彰化縣",
    "o": "新竹市",
    "p": "雲林縣",
    "q": "嘉義縣",
    "t": "屏東縣",
    "u": "花蓮縣",
    "v": "臺東縣",
    "w": "金門縣",
    "x": "澎湖縣",
}


def load_presale_data(data_file_path: str) -> pd.DataFrame:
    presale_data_list = []
    presale_key_list = []

    for path, root, files in os.walk(data_file_path):
        for f in files:
            if f.endswith("lvr_land_b.csv"):
                presale_key_list.append(
                    (
                        path.split("/")[-1][:3],
                        path.split("/")[-1][3:],
                        code_to_city_map.get(f.split("_")[0]),
                    )
                )
                presale_data_list.append(os.path.join(path, f))

    presale_df = (
        pd.concat(
            objs=(pd.read_csv(f, header=[0]).iloc[1:, :] for f in presale_data_list),
            axis=0,
            keys=presale_key_list,
        )
        .reset_index(level=[0, 1, 2])
        .rename(
            columns={"level_0": "記錄年份", "level_1": "記錄季度", "level_2": "城市"}
        )
    ).reset_index(drop=True)

    return presale_df.loc[
        (presale_df["都市土地使用分區"] == "住")
        & (presale_df["解約情形"].isna())
        & (
            ~presale_df["建物型態"].isin(
                ["其他", "工廠", "辦公商業大樓", "廠辦", "店面(店鋪)"]
            )
        ),
        [
            "記錄年份",
            "記錄季度",
            "城市",
            "鄉鎮市區",
            "交易筆棟數",
            "建物型態",
            "土地移轉總面積平方公尺",
            "建物移轉總面積平方公尺",
            "車位移轉總面積平方公尺",
            "建物現況格局-房",
            "建物現況格局-廳",
            "建物現況格局-衛",
            "建物現況格局-隔間",
            "總價元",
        ],
    ]
