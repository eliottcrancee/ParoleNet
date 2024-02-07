"""
Download examples:

Whole archive:
curl 'https://amubox.univ-amu.fr/s/gkfA7rZCWGQFqif/download' -H 'Connection: keep-alive' -H 'Accept-Encoding: gzip, deflate, br' --output archive.zip

Specific files with path:
curl 'https://amubox.univ-amu.fr/s/gkfA7rZCWGQFqif/download?path=%2Fvideo%2F&files=openface.zip' -H 'Connection: keep-alive' -H 'Accept-Encoding: gzip, deflate, br' --output video/openface.zip
"""

import pandas as pd
import numpy as np
from glob import glob

import sys
import os
from os.path import dirname as up
sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))


def load_all_ipus(folder_path: str = "transcr", load_words: bool = False):
    """Load all csv and concatenate"""
    file_list = glob(
        os.path.join(folder_path, f"*_merge{'_words' if load_words else ''}.csv")
    )
    # Load all csv files
    data = []
    for file in file_list:
        df = pd.read_csv(file, na_values=[""])  # one speaker name is 'NA'
        df["dyad"] = file.split("/")[-1].split("_")[0]
        data.append(df)

    data = pd.concat(data, axis=0).reset_index(drop=True)
    print(data.shape)
    plabels = [
        col
        for col in data.columns
        if not any(
            [
                col.startswith(c)
                for c in [
                    "dyad",
                    "ipu_id",
                    "speaker",
                    "start",
                    "stop",
                    "text",
                    "duration",
                ]
            ]
        )
    ]
    print(data[plabels].sum(axis=0) / data.shape[0])
    return data


def filter_after_jokes(df_ipu: pd.DataFrame):
    """First few ipus are useless / common to all conversations"""
    jokes_end = (
        df_ipu[
            df_ipu.text.apply(
                lambda x: (
                    False
                    if isinstance(x, float)
                    else (
                        ("il y avait un âne" in x) or ("qui parle ça c'est cool" in x)
                    )
                )
            )
        ]
        .groupby("dyad")
        .agg({"ipu_id": "max"})
        .to_dict()["ipu_id"]
    )
    return (
        df_ipu[df_ipu.apply(lambda x: x.ipu_id > jokes_end.get(x.dyad, 0), axis=1)],
        jokes_end,
    )


if __name__ == "__main__":

    dir_path = up(up(up(os.path.abspath(__file__))))
    print(dir_path)
    audio_path = dir_path + "/dataset/audio/2_channels/"
    raw_data = load_all_ipus(dir_path + "/dataset/transcr")
    display(raw_data)
