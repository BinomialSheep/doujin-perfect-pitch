"""
test_data配下のファイルの声優を予想する
"""

# 標準ライブラリ
from pathlib import Path
import glob
import sys
import json

# 外部ライブラリ
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


def predict_topN(model, data, label_decoding_map, topN=1):
    """
    生データを基に、予測上位N件までを取得するようにする
    TODO：auto_model_optimizerと類似なのでcommonで管理する？
    """
    predictions = pyc.predict_model(model, data=data, raw_score=True)

    prob_columns = [
        col for col in predictions.columns if col.startswith("prediction_score_")
    ]

    topN_classes = predictions[prob_columns].apply(
        lambda x: x.nlargest(topN).index.tolist(), axis=1
    )
    import re

    new_topN_classes = []
    for sublist in topN_classes:
        tmp = []
        for s in sublist:
            v = int(re.search(r"\d+", s).group())
            name = label_decoding_map[v]

            tmp.append(name)
        new_topN_classes.append(tmp)

    predictions["TopN_Classes"] = new_topN_classes

    predictions["TopN_Scores"] = predictions[prob_columns].apply(
        lambda x: x.nlargest(topN).values.tolist(), axis=1
    )
    return predictions[["TopN_Classes", "TopN_Scores"]]


def main():
    # 使用するモデルとパラメータの指定
    MODEL_NAME = "mfcc_17000_mte100_model"

    DIR_PATH = f"{ROOT_DIR}/prediction/test_data"
    files = glob.glob(DIR_PATH + "/*.wav") + glob.glob(DIR_PATH + "/*.mp3")
    MODEL_PATH = f"{ROOT_DIR}/data/{MODEL_NAME}"

    # モデルの読み込み
    model = pyc.load_model(MODEL_PATH)
    label_decoding_map = load_label_mapping(MODEL_PATH)

    # mfccに変換
    columns = [f"mfcc_{n}" for n in range(1, 13)]
    mfcc_df = pd.DataFrame(index=np.arange(len(files)), columns=columns)
    for i in range(len(files)):
        voice = vp.VoiceProcessor(files[i])
        mfcc = voice.calc_mfcc()
        mfcc_df.iloc[i] = mfcc

    # 予測
    predictions = predict_topN(model, mfcc_df, label_decoding_map, 5)

    # 出力
    for index, prediction in predictions.iterrows():
        print(files[index])
        print(prediction)


if __name__ == "__main__":
    main()
