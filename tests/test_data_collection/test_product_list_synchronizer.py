import sys
from pathlib import Path
import os

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from data_collection import product_list_synchronizer as pls


def test_fetch_product_list():
    result = pls.fetch_product_list(1)
    assert len(result) == 100


def test_write_product_list():
    product_list = ["PJxxxxxx", "PJyyyyyy"]
    registed_product_set = set()
    registed_product_set.add("PJyyyyyy")
    path = "dummy"
    res = pls.write_product_list(product_list, path, registed_product_set)
    os.remove("dummy")
    assert res is False
