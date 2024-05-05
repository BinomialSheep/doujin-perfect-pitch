import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from common.dbconnection import DatabaseConnection
import unittest


class TestDatabaseConnection(unittest.TestCase):
    def test_connection_open_close(self):
        """副作用として、doujin_perfect_pitch.dbファイルがない場合作成される"""
        with DatabaseConnection() as conn:
            self.assertIsNotNone(conn)


if __name__ == "__main__":
    unittest.main()
