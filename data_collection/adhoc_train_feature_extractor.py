"""
ローカルから特徴量抽出してcsvを作る（DBに入れない）
実験用で、基本的に雑に書き捨てる
"""

# 標準ライブラリ
from pathlib import Path
import os
import sys

# 外部ライブラリ
import pandas as pd
import numpy as np

# 自作ライブラリ
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
import common.voice_processor as vp


def fetch_audio_info(file_path):
    """yaml形式のファイルを開いてdictで返す
    NOTE：リストとか使いたくなったら外部ライブラリのyaml処理を使う
    """
    audio_info = dict()
    if not os.path.exists(file_path):
        audio_info["speaker"] = "unknown"
        audio_info["audio_info"] = "unknown"
        return audio_info
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                audio_info[key.strip()] = val.strip()
    return audio_info


def list_files(dir_path):
    """訓練データフォルダの音声ファイルをリストアップする。
    各フォルダのinfo.yamlファイルから情報を取得し、
    [speaker, audio_info, file_path]のリストを返す。
    """
    train_files = []

    for root, dirs, files in os.walk(dir_path):
        for dir_name in dirs:
            dir_full_path = os.path.join(root, dir_name)
            # info.yamlから情報取得
            audio_info = fetch_audio_info(f"{dir_full_path}/info.yaml")
            if (audio_info["speaker"]) == "unknown":
                continue
            for sub_root, sub_dirs, sub_files in os.walk(dir_full_path):
                for file_name in sub_files:
                    if not vp.VoiceProcessor.is_audio_file(file_name):
                        continue
                    file_full_path = os.path.join(sub_root, file_name)
                    train_files.append(
                        [
                            audio_info["speaker"],
                            audio_info["audio_info"],
                            file_full_path,
                        ]
                    )
    return train_files


def main():
    DIR_PATH = f"{ROOT_DIR}/data/train_data"

    train_files = list_files(DIR_PATH)

    # mfccを計算する場合（audio_infoごとに分ける）

    for speaker, audio_type, file_path in train_files:
        print(speaker, audio_type, file_path)
        voice = vp.VoiceProcessor(file_path)
        mfcc = voice.calc_mfcc()
        mfcc.append(speaker)

        csv_file = f"{ROOT_DIR}/data/mfcc_{audio_type}.csv"
        is_file_exist = os.path.exists(csv_file)
        with open(csv_file, "a", encoding="utf-8") as file:
            if not is_file_exist:
                mfcc_header = "mfcc_1,mfcc_2,mfcc_3,mfcc_4,mfcc_5,mfcc_6,mfcc_7,mfcc_8,mfcc_9,mfcc_10,mfcc_11,mfcc_12,target"
                file.write(mfcc_header + "\n")
            file.write(",".join(map(str, mfcc)) + "\n")
        # print(mfcc)


if __name__ == "__main__":
    main()
