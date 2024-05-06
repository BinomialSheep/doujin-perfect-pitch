# 標準ライブラリ
from pathlib import Path

# 外部ライブラリ

# 自作ライブラリ
import auto_model_optimizer


def main():
    """モデルの構築と保存を行う"""
    # CSVファイル名
    CSV_FILE_NAME = "mfcc_1000_noisy.csv"
    # 正解ラベルのカラム名
    GROUND_TRUTH = "target"
    # 保存するモデルのファイル名
    SAVE_FILE_NAME = "mfcc_1000_noisy_model"
    # 使用するモデル（通常、compare_modelsで最善のモデル）
    MODEL = "et"

    # フルパス
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CSV_DATA_PATH = f"{ROOT_DIR}/data/{CSV_FILE_NAME}"
    SAVE_FILE_PATH = f"{ROOT_DIR}/data/{SAVE_FILE_NAME}"
    # モデル構築の実行
    model_finder = auto_model_optimizer.AutoModelOptimizer()
    model_finder.load_data(CSV_DATA_PATH, GROUND_TRUTH)
    model_finder.encode_label()
    model_finder.setup_model()
    model_finder.create_model(MODEL, SAVE_FILE_PATH)

    # 分類レポートの出力
    report = model_finder.generete_report()
    print(report)


if __name__ == "__main__":
    main()
