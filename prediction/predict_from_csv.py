"""
指定したcsvのファイルの声優を予想する
"""

# 標準ライブラリ
from pathlib import Path
import glob
import sys
import json
import re

# 外部ライブラリ
import sklearn.metrics
import pycaret.classification as pyc
import pandas as pd
import numpy as np

# 自作ライブラリ
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
import common.voice_processor as vp


def load_label_mapping(model_path):
    mapping_path = f"{model_path}_label_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as file:
        label_mapping = json.load(file)
    rev = {v: k for k, v in label_mapping.items()}
    return rev


def print_report(report_dict):
    """
    f1-scoreで降順ソートしたレポートをdf形式で出力する
    """
    accuracy = report_dict.pop("accuracy")
    # stats = {key: report_dict.pop(key) for key in ["accuracy"]}

    df = pd.DataFrame.from_dict(report_dict, orient="index")
    sorted_df = df.sort_values(by="f1-score", ascending=False)
    # NOTE：average行を除くため2引く
    speaker_count = len(df) - 2
    data_count = df.loc["weighted avg", "support"]
    print(f"|正解率|声優数|テストデータ件数|")
    print(f"|{round(accuracy, 4)}|{speaker_count}|{int(data_count)}|")
    print(sorted_df.to_string())


def generete_report(predictions, label_decoding_map):
    y_true = predictions["target"]
    y_pred = predictions["prediction_label"]

    tmp = []
    for s in y_pred:
        name = label_decoding_map[int(s)]
        tmp.append(name)
    y_pred = tmp
    report = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)
    print_report(report)


def main():
    # 使用するモデルとパラメータの指定
    # MODEL_NAME = "mfcc_PRODUCT_SE_OFF_by20sec_remove_silence_better_model"
    MODEL_NAME = "mfcc_PRODUCT_SE_OFF_model"
    FILE_NAME = "mfcc_17000_mte10.csv"

    MODEL_PATH = f"{ROOT_DIR}/data/{MODEL_NAME}"
    FILE_PATH = f"{ROOT_DIR}/data/{FILE_NAME}"

    # モデルの読み込み
    model = pyc.load_model(MODEL_PATH)
    label_decoding_map = load_label_mapping(MODEL_PATH)

    # ファイル読み込み
    mfcc_df = pd.read_csv(FILE_PATH)
    # モデルにない名前は除去する
    name_set = set(label_decoding_map.values())
    mfcc_df = mfcc_df[mfcc_df["target"].isin(name_set)]

    # 分類の実行
    predictions = pyc.predict_model(estimator=model, data=mfcc_df)
    # レポート出力
    generete_report(predictions, label_decoding_map)


if __name__ == "__main__":
    main()
