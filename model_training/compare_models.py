# 標準ライブラリ
from pathlib import Path

# 外部ライブラリ

# 自作ライブラリ
import auto_model_optimizer


def main():
    """autoMLでいろいろなモデルの正解率などを比較する
    （比較するだけで、ここではモデルの構築は行わない）
    """

    # dataフォルダ下のcsvファイルと、正解ラベルのカラム名を指定する
    csv_file = "mfcc_17000_mte100.csv"
    grouund_truth = "target"

    # autoMLの実行
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CSV_DATA_PATH = f"{ROOT_DIR}/data/{csv_file}"
    model_finder = auto_model_optimizer.AutoModelOptimizer()
    model_finder.load_data(CSV_DATA_PATH, grouund_truth)
    model_finder.encode_label()

    # model_finder.check_data()
    # exit()
    model_finder.setup_model()
    models = model_finder.compare_models()

    # 雑にいろいろ標準出力
    print(models)


if __name__ == "__main__":
    main()
