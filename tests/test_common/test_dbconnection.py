import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from common.dbconnection import DatabaseConnection


def test_connection_open_close():
    """副作用として、doujin_perfect_pitch.dbファイルがない場合作成される"""
    with DatabaseConnection() as conn:
        assert conn
