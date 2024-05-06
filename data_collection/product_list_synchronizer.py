from bs4 import BeautifulSoup
from pathlib import Path

# import sys

# root_dir = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(root_dir))

from .web_content_fetcher import WebContentFetcher


"""
DLsiteの新着音声作品をチェックし、product_list.txtを最新化する。
途中で打ち切るので削除済み作品を削除することはしない（したければ一度product_list.txtを削除する）
product_list.txtの順序は、今回の更新差分が新着順で末尾に入るようになる（いずれより良い性質を保証しない）
"""


def fetch_product_list(page_num: int) -> list[str]:
    """作品のURLリストを新着順で最大100件取得する"""
    url = f"https://www.dlsite.com/maniax/works/type/=/language/jp/sex_category%5B0%5D/male/work_category%5B0%5D/doujin/order%5B0%5D/release_d/work_type_category%5B0%5D/audio/work_type_category_name%5B0%5D/%E3%83%9C%E3%82%A4%E3%82%B9%E3%83%BBASMR/per_page/100/page/{page_num}/show_type/3"
    fetcher = WebContentFetcher()
    html = fetcher.fetch_html(url)
    if html is None:
        return
    soup = BeautifulSoup(html, "html.parser")

    worklist_ul = soup.find("ul", class_="n_worklist")
    product_elements = worklist_ul.find_all(attrs={"data-product_id": True})
    product_list = [elem["data-product_id"] for elem in product_elements]
    return product_list


def write_product_list(
    product_list: list[str], file_path: str, registed_product_set: set[str]
) -> bool:
    """指定されたファイルに商品を追加する
    全商品を書き込んだらTrue, 途中で書き込みを打ち切ったらFalseを返す
    """
    # ファイルを追記モードで開く（ファイルが存在しない場合は新規作成される）
    with open(file_path, "a", encoding="utf-8") as file:
        for product_id in product_list:
            if product_id in registed_product_set:
                return False
            # 各商品IDを新しい行に書き出す
            file.write(product_id + "\n")
    return True


def main():
    FILE_PATH = "../product_list.txt"
    # 登録済み作品
    registed_product_set = set()
    with open(FILE_PATH, "r") as file:
        for line in file:
            registed_product_set.add(line.strip())

    for i in range(1, 500):
        product_list = fetch_product_list(i)
        res = write_product_list(product_list, FILE_PATH, registed_product_set)
        if res is False:
            break


if __name__ == "__main__":
    main()
