# 標準ライブラリ
from pathlib import Path

# 外部ライブラリ
import pandas as pd
import pycaret.classification as pyc
import sklearn.preprocessing
import sklearn.metrics

# 自作ライブラリ


class AutoModelOptimizer:
    """
    CSVデータを読み込み、pycaretを使用して最適な機械学習モデルの探索・構築をクラス。

    Note:
        cumlをインストールしていないので大量のWARNINGがログ出力されるが現状未対応（Dockerに移したらするかも）
    """

    def __init__(self):
        pass

    def load_data(self, data_path, grouund_truth):
        self.data = pd.read_csv(data_path, encoding="utf-8")
        self.grouund_truth = grouund_truth

    def encode_label(self):
        """日本語の声優名を数字にエンコードする"""
        self.encoder = sklearn.preprocessing.LabelEncoder()
        encoded = self.encoder.fit_transform(self.data[self.grouund_truth])
        self.data[self.grouund_truth] = encoded

    def check_data(self):
        """デバッグ用"""
        print(self.data.head())
        print(self.data.dtypes)
        print(self.data.count())
        print(self.data.isnull().sum())

    def setup_model(self):
        pyc.setup(
            data=self.data,
            target=self.grouund_truth,  # 正解ラベル
            data_split_shuffle=True,
            use_gpu=True,
            fold=3,  # TODO：データが増えたら5にする
            verbose=False,
            n_jobs=-1,
            system_log="pycratlog.log",
        )

    def compare_models(self) -> list[any]:
        # models = pyc.compare_models()
        # 遅いモデルを避ける場合
        models = pyc.compare_models(exclude=["catboost", "xgboost", "gbc"])
        return models

    def create_model(self, best_model, save_path):
        self.tuned_model = pyc.create_model(best_model, verbose=False)
        pyc.save_model(self.tuned_model, save_path)

    def generete_report(self):
        predictions = pyc.predict_model(self.tuned_model)
        y_true = predictions[self.grouund_truth]
        y_pred = predictions["prediction_label"]
        if self.encoder:
            y_true = self.encoder.inverse_transform(y_true)
            y_pred = self.encoder.inverse_transform(y_pred)
        report = sklearn.metrics.classification_report(y_true, y_pred)
        return report
