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
import noisereduce

# 自作ライブラリ
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
import common.voice_processor as vp


def plot_one(audio: np.ndarray, sr: int, idx: int, title: str):
    # 生波形
    ax = plt.subplot(3, 3, idx)
    ax.set_ylim([-0.4, 0.4])
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"{title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    # スペクトログラム
    plt.subplot(3, 3, idx + 3)
    D_original = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D_original, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{title} Spectrogram")
    # MFCC
    plt.subplot(3, 3, idx + 6)
    mfcc_original = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfcc_original, sr=sr, x_axis="time")
    plt.colorbar()
    plt.title(f"{title} MFCC")


def plot_waveforms(audio1: np.ndarray, audio2: np.ndarray, audio3: np.ndarray, sr: int):
    """3種類のauidoを波形を比較"""
    plt.figure(figsize=(24, 12))

    plot_one(audio1, sr, 1, "SE OFF")
    plot_one(audio2, sr, 2, "SE ON")
    plot_one(audio3, sr, 3, "SE noisereduce")

    plt.tight_layout()
    plt.show()


def save_audio(audio: np.ndarray, sr: int, output_path="output.wav"):
    """加工後音声を保存して聴きたい場合とかに使う"""
    soundfile.write(output_path, audio, sr)


def main():
    DIR = f"{ROOT_DIR}/data/train_data/RJ01085787"
    # file_path_1 = f"{DIR}/SE_OFF/03.お風呂上がりに爪切りとハンドマッサージSEなし.wav"
    # file_path_2 = f"{DIR}/SE_ON/03.お風呂上がりに爪切りとハンドマッサージ.wav"
    file_path_1 = f"{ROOT_DIR}/data/train_data/RJ01085787/SE_OFF/03_20sec.wav"
    file_path_2 = f"{ROOT_DIR}/data/train_data/RJ01085787/SE_ON/03_20sec.wav"

    voice1, sr = librosa.load(file_path_1)
    voice2, _ = librosa.load(file_path_2)
    # voice2 = voice2[: len(voice1)]
    voice3 = noisereduce.reduce_noise(y=voice2, sr=sr)
    print("処理OK")
    plot_waveforms(voice1, voice2, voice3, sr)

    # 120~140秒を取り出す
    # voice1 = voice1[120 * sr : 140 * sr]
    # voice2 = voice2[120 * sr : 140 * sr]
    # voice3 = voice3[120 * sr : 140 * sr]
    # plot_waveforms(voice1, voice2, voice3, sr)

    # save_audio(voice1, sr, f"{ROOT_DIR}/data/train_data/RJ01085787/SE_OFF/03_20sec.wav")
    # save_audio(voice2, sr, f"{ROOT_DIR}/data/train_data/RJ01085787/SE_ON/03_20sec.wav")


if __name__ == "__main__":
    main()
