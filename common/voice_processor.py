import numpy as np
import librosa


class VoiceProcessor:

    def __init__(self, file_path):
        self.y, self.sr = librosa.load(file_path)  # 音声ファイルへのパス

    def calc_mfcc(self) -> list[float]:
        """12次までのメル周波数ケプストラム係数を求める"""
        X_data = []  # 特徴行列
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr)  # MFCC
        mfcc = np.average(mfcc, axis=1).flatten().tolist()  # MFCC
        # 低次の係数だけ取り出す（12次まで取り出すことが多い）
        return mfcc[1:13]

    def get_duration(self):
        """音声ファイルの長さを秒単位で返す"""
        return int(librosa.get_duration(y=self.y, sr=self.sr))

    @staticmethod
    def is_audio_file(file_name):
        """librosaで扱えるファイルか判別する"""
        # 拡張子チェック
        AUDIO_EXTENTIONS = ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]
        return file_name[-3:].lower() in AUDIO_EXTENTIONS
