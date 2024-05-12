# 標準ライブラリ
import json

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
        self.grouund_truth = grouund_truth
        self.data = pd.read_csv(data_path, encoding="utf-8")

        self.data = self.data.applymap(
            lambda x: (
                x.encode("cp932", "ignore").decode("cp932") if isinstance(x, str) else x
            )
        )
        # 実験用
        # self.data[grouund_truth] = self.data[grouund_truth].replace(
        #     "乙倉ゅい（乙倉由依）", "乙倉ゅい(乙倉由依)"
        # )

    def encode_label(self):
        """日本語の声優名を数字にエンコードする"""
        self.encoder = sklearn.preprocessing.LabelEncoder()
        encoded = self.encoder.fit_transform(self.data[self.grouund_truth])
        self.data[self.grouund_truth] = encoded

    def jsonify_encoder(self, file_path):
        """予測時に日本語の声優名が使えるように外部出力しておく"""
        label_mapping = dict(
            zip(self.encoder.classes_, self.encoder.transform(self.encoder.classes_))
        )
        # json.dumpはnumpy.int32を扱えないので標準型に直している
        label_mapping = {str(k): int(v) for k, v in label_mapping.items()}

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(label_mapping, file, ensure_ascii=False, indent=4)

    def check_data(self):
        """デバッグ用"""
        print(self.data.head())
        print(self.data.dtypes)
        print(self.data.count())
        print(self.data.isnull().sum())

    def setup_model(self):
        self.clf1 = pyc.setup(
            data=self.data,
            target=self.grouund_truth,  # 正解ラベル
            train_size=0.7,
            data_split_shuffle=True,
            use_gpu=True,
            fold=10,  # 訓練データの分割数
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
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        self.tuned_model = pyc.create_model(best_model, verbose=True)
        pyc.save_model(self.tuned_model, save_path)

    def print_report(self, report_dict):
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

    def generete_report(self):
        predictions = pyc.predict_model(self.tuned_model)
        y_true = predictions[self.grouund_truth]
        y_pred = predictions["prediction_label"]
        if self.encoder:
            y_true = self.encoder.inverse_transform(y_true)
            y_pred = self.encoder.inverse_transform(y_pred)
        report = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)
        self.print_report(report)

    def predict_topN(self, topN=3):
        """
        生データを基に、予測上位N件までを取得するようにする
        """
        predictions = pyc.predict_model(self.tuned_model, raw_score=True)

        y_true = predictions[self.grouund_truth]
        if self.encoder:
            predictions[self.grouund_truth] = self.encoder.inverse_transform(y_true)

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
                name = self.encoder.inverse_transform([v])
                tmp.append(name[0])
            new_topN_classes.append(tmp)

        predictions["TopN_Classes"] = new_topN_classes

        predictions["TopN_Scores"] = predictions[prob_columns].apply(
            lambda x: x.nlargest(topN).values.tolist(), axis=1
        )

        return predictions[[self.grouund_truth, "TopN_Classes", "TopN_Scores"]]

    def generate_my_report(self, topN=3):
        """
        追加のレポートを出力する
        """
        # 予想上位N件まで出力する
        prediction_topN = self.predict_topN(topN)
        print(prediction_topN.to_string(index=False))
        # 予想上位N件までに含まれていた場合は正解とする場合のレポートを出す
        y_true = prediction_topN[self.grouund_truth]
        y_pred = prediction_topN.apply(
            lambda row: (
                row[self.grouund_truth]
                if row[self.grouund_truth] in row["TopN_Classes"]
                else row["TopN_Classes"][0]
            ),
            axis=1,
        )
        report = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)
        self.print_report(report)
        # 確信度と正解率を0.1刻みで集計して出す
        for i in range(11):
            threshold = i / 10
            filtered = prediction_topN[
                prediction_topN["TopN_Scores"].apply(lambda x: float(x[0]) >= threshold)
            ]
            y_true = filtered[self.grouund_truth]
            y_pred = filtered.apply(
                lambda row: row["TopN_Classes"][0],
                axis=1,
            )
            print(f"閾値：{threshold}, 件数：{len(y_true) - 2}")
            report = sklearn.metrics.classification_report(
                y_true, y_pred, output_dict=True
            )
            self.print_report(report)
