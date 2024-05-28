# 標準ライブラリ
from pathlib import Path

# 外部ライブラリ

# 自作ライブラリ
import auto_model_optimizer


def main():
    """モデルの構築と保存を行う"""
    # CSVファイル名
    CSV_FILE_NAME = "mfcc_PRODUCT_SE_OFF_by20sec_remove_silence_better.csv"
    # 正解ラベルのカラム名
    GROUND_TRUTH = "target"
    # 保存するモデルのファイル名
    SAVE_FILE_NAME = "mfcc_PRODUCT_SE_OFF_by20sec_remove_silence_better_model"
    # 使用するモデル（通常、compare_modelsで最善のモデル）
    MODEL = "et"

    # フルパス
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CSV_DATA_PATH = f"{ROOT_DIR}/data/{CSV_FILE_NAME}"
    SAVE_FILE_PATH = f"{ROOT_DIR}/data/{SAVE_FILE_NAME}"
    # モデル構築の実行
    model_optimizer = auto_model_optimizer.AutoModelOptimizer()
    model_optimizer.load_data(CSV_DATA_PATH, GROUND_TRUTH)
    model_optimizer.encode_label()
    model_optimizer.setup_model()
    model_optimizer.create_model(MODEL, SAVE_FILE_PATH)

    model_optimizer.jsonify_encoder(f"{SAVE_FILE_PATH}_label_mapping.json")

    # 分類レポートの出力
    model_optimizer.generete_report()
    model_optimizer.plot_mmodel()

    # topMなら正解とする場合
    # model_optimizer.generate_my_report(3)

if __name__ == "__main__":
    main()
