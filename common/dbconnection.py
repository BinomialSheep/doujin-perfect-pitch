import sqlite3
import os
from pathlib import Path


class DatabaseConnection:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATABASE_NAME = f"{ROOT_DIR}/data/doujin_perfect_pitch.db"

    def __init__(self):
        pass

    def __enter__(self):
        # データベースに接続（データベースが存在しない場合は新規作成）
        self.conn = sqlite3.connect(self.DATABASE_NAME)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
