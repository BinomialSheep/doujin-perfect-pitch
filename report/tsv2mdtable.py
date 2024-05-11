"""
タブ区切りのデータをマークダウンで扱える表に変換する
"""

import re

"""

"""

input_data = """
   speaker        threshold   precision    recall  f1-score  support
こやまはる    0.0      0.500000  0.205479  0.291262     73.0
こやまはる    0.1      0.500000  0.206349  0.292135     63.0
こやまはる    0.2      0.750000  0.142857  0.240000     21.0
こやまはる    0.3      0.000000  0.000000  0.000000      6

"""


def normalize_text(str):
    """キレイなtsv形式にする"""
    # 前後に空白があることがある
    str = str.strip()
    # タブでなく適当な文字数の半角で区切られているらしい
    str = re.sub(r" {2,}", "\t", str)
    return str


def main():
    lines = input_data.strip().split("\n")
    lines = [normalize_text(line) for line in lines]

    header = "| " + " | ".join(lines[0].split("\t")) + " |"
    separator = "|" + "---------|" * len(lines[0].split("\t"))

    # ヘッダーとセパレーターを出力
    print(header)
    print(separator)

    # 各データ行に対して整形して出力
    for line in lines[1:]:
        formatted_line = "| " + " | ".join(line.split("\t")) + " |"
        print(formatted_line)


if __name__ == "__main__":
    main()
