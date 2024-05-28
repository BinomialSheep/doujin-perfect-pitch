"""
タブ区切りのデータをマークダウンで扱える表に変換する
"""

import re

input_data = """
    speaker                precision    recall  f1-score  support
みもりあいの         0.426559  0.441667  0.433982    480.0
伊ヶ崎綾香          0.345924  0.559486  0.427518    311.0
涼花みなせ          0.348285  0.514954  0.415530    769.0
架月らみゅ          0.350000  0.385321  0.366812    109.0
weighted avg   0.396943  0.332803  0.331056   3771.0
沢野ぽぷら          0.290323  0.303371  0.296703    178.0
陽向葵ゅか          0.527027  0.198545  0.288431   1375.0
macro avg      0.284701  0.309572  0.281391   3771.0
こやまはる          0.320652  0.240816  0.275058    245.0
浅見ゆい           0.123288  0.206897  0.154506     87.0
一之瀬りと          0.075893  0.140496  0.098551    121.0
かの仔            0.039062  0.104167  0.056818     96.0
"""


def normalize_tsv(str):
    """キレイなtsv形式にする"""
    # 前後に空白があることがある
    str = str.strip()
    # タブでなく適当な文字数の半角空白で区切られているらしい
    str = re.sub(r" {2,}", "\t", str)
    return str


def main():
    lines = input_data.strip().split("\n")
    lines = [normalize_tsv(line) for line in lines]

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
