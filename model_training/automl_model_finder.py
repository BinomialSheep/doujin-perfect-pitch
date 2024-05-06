# 標準ライブラリ
from pathlib import Path

# 外部ライブラリ
import pandas as pd
from pycaret.classification import setup, compare_models
from sklearn.preprocessing import LabelEncoder

# 自作ライブラリ


class AutoMLModelFinder:
    """
    CSVデータを読み込み、pycaretを使用して最適な機械学習モデルを探索するクラス。

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
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(self.data[self.grouund_truth])
        self.data[self.grouund_truth] = encoded

    def check_data(self):
        """デバッグ用"""
        print(self.data.head())
        print(self.data.dtypes)
        print(self.data.count())
        print(self.data.isnull().sum())

    def setup_model(self):
        setup(
            data=self.data,
            target=self.grouund_truth,  # 正解ラベル
            data_split_shuffle=True,
            use_gpu=True,
            fold=3,  # TODO：データが増えたら5にする
            verbose=False,
            n_jobs=-1,
            system_log="pycratlog.log",
        )

    def find_best_model(self) -> list[any]:
        best_model = compare_models()
        # 遅いモデルを避ける場合
        # best_model = compare_models(exclude=["catboost", "xgboost", "gbc", "rf"])
        return best_model


def main():
    # dataフォルダ下のcsvファイルと、正解ラベルのカラム名を指定する
    csv_file = "mfcc_1000_noisy.csv"
    grouund_truth = "target"

    # autoMLの実行
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CSV_DATA_PATH = f"{ROOT_DIR}/data/{csv_file}"
    model_finder = AutoMLModelFinder()
    model_finder.load_data(CSV_DATA_PATH, grouund_truth)
    model_finder.encode_label()

    # model_finder.check_data()
    # exit()
    model_finder.setup_model()
    best_model = model_finder.find_best_model()

    # 雑にいろいろ標準出力
    print(best_model)


if __name__ == "__main__":
    main()
