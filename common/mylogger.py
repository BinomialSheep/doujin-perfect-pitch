import logging


def setup_logger(name="default", logfile="error.log") -> logging.Logger:
    """ロガーを設定し、設定されたロガーを返す"""
    # ロガーを作成し、ログレベルをINFOに設定
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # ログファイルの出力設定
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(logging.INFO)  # ファイルに記録する最低レベル

    # ログメッセージのフォーマット設定
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # ハンドラをロガーに追加
    logger.addHandler(file_handler)

    return logger
