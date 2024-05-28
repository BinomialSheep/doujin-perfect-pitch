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
import matplotlib.pyplot as plt
import librosa
import soundfile

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


def calc_mfccs_bysec(y: np.ndarray, sr: int, speaker: str, segment_length: int):
    # 音声の長さを秒単位で計算
    duration = librosa.get_duration(y=y, sr=sr)

    num_segments = int(duration // segment_length)

    # 各セグメントのMFCCを計算して保存
    mfccs_list = []

    for i in range(num_segments):
        start_sample = int(i * segment_length * sr)
        if i == num_segments - 1:
            # 最後のセグメントは残りのすべてのサンプルを含む
            segment = y[start_sample:]
        else:
            end_sample = int((i + 1) * segment_length * sr)
            segment = y[start_sample:end_sample]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr)
        mfcc = np.average(mfcc, axis=1).flatten().tolist()
        mfcc = mfcc[1:13]
        mfcc.append(speaker)
        mfccs_list.append(mfcc)

    return mfccs_list


def calc_one_mfcc(file_path, speaker):
    # mfcc_list = f(file_path, speaker)
    voice = vp.VoiceProcessor(file_path)
    mfcc = voice.calc_mfcc()
    mfcc.append(speaker)
    return mfcc


def remove_silence(
    voice: np.ndarray, sr: int, min_silence_duration: float = 1.5, top_db: int = 20
) -> np.ndarray:
    """min_silence_duration秒以上続く無音をカットする"""
    # 無音部分を検出する
    intervals = librosa.effects.split(voice, top_db=top_db)
    # 無音部分をカットして音の部分を結合する
    min_silence_samples = int(min_silence_duration * sr)

    parts = []
    silence = np.zeros(int(0.2 * sr))
    for (start1, end1), (start2, end2) in zip(intervals, intervals[1:]):
        if (start2 - end1) > min_silence_samples:
            # 空白が大きいなら空白はカット
            # 不自然にならないように0.1秒空ける
            parts.append(voice[start1:end1])
            parts.append(silence)
        else:
            # 大きくないならカットしない
            parts.append(voice[start1:start2])
    # 最後の要素がループされないので加える
    parts.append(voice[intervals[-1]])
    non_silent_voice = np.concatenate(parts)
    return non_silent_voice


def plot_waveforms(audio1: np.ndarray, audio2: np.ndarray, sr: int):
    """2種類のauidoを波形を比較"""
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio1, sr=sr)
    plt.title("Audio 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(audio2, sr=sr)
    plt.title("Audio 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def save_audio(audio: np.ndarray, sr: int, output_path="output.wav"):
    """加工後音声を保存して聴きたい場合とかに使う"""
    soundfile.write(output_path, audio, sr)


def main():
    DIR_PATH = f"{ROOT_DIR}/data/train_data"

    train_files = list_files(DIR_PATH)

    # mfccを計算する場合（audio_infoごとに分ける）
    segment_length = 20

    for speaker, audio_type, file_path in train_files:
        print(speaker, audio_type, file_path)
        voice, sr = librosa.load(file_path)
        non_silent_voice = remove_silence(voice, sr)
        print(
            librosa.get_duration(y=voice, sr=sr),
            librosa.get_duration(y=non_silent_voice, sr=sr),
        )
        # plot_waveforms(voice, non_silent_voice, sr)
        # save_audio(non_silent_voice, sr)
        # exit()

        mfcc_list = calc_mfccs_bysec(non_silent_voice, sr, speaker, segment_length)

        csv_file = f"{ROOT_DIR}/data/mfcc_{audio_type}_by{segment_length}sec_remove_silence_better.csv"
        is_file_exist = os.path.exists(csv_file)
        with open(csv_file, "a", encoding="utf-8") as file:
            if not is_file_exist:
                mfcc_header = "mfcc_1,mfcc_2,mfcc_3,mfcc_4,mfcc_5,mfcc_6,mfcc_7,mfcc_8,mfcc_9,mfcc_10,mfcc_11,mfcc_12,target"
                file.write(mfcc_header + "\n")
            for mfcc in mfcc_list:
                file.write(",".join(map(str, mfcc)) + "\n")
        # print(mfcc)


if __name__ == "__main__":
    main()
