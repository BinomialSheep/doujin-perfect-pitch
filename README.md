# 同人ダメ絶対音感AI（doujin-perfect-pitch）

## これは？

音声ファイルを入力すると、どのDLsite声優の声か識別できるシステム。
以下の3つの役割がある。

### データ収集
Webスクレイピングによるデータ収集と特徴量抽出を頑張る

### モデリング
機械学習によるモデルの訓練を頑張る

### 予測
予測を頑張る

## レポート

[20240511_MFCC×pycaret.md](#./report/20240511_MFCC×pycaret.md)


## 使用方法
### Python環境構築
コードを読んで必要なpip installをする（そんな……）（Docker化予定）

### Python以外の環境構築
データ収集ではsubprocessでffmpegと7zipを呼び出しているので、インストールしてパスを通す必要がある。

- ffmpeg：体験版が動画ファイルの場合にmp3でダウンロードして使用する際に使用
- 7zip：体験版zipファイルがDeflate64で圧縮されていた場合に使用

### データ収集
todo（自分以外の使用は想定していない）

### モデリング
```.\model_training\compare_models.py```

モデル比較を行うスクリプト。
compare_models.py内のCSV_FILE_NAMEで学習に使うファイルを設定する。
2024年5月13日現在は"mfcc_17000_mte100.csv"が標準。

訓練データ比率や分割数の変更は想定していないが、auto_model_poptimizer.py内のsetup_model関数の引数で変更する。
一部の遅くて今回性能が出ないモデルはデフォルトでは除外しているが、この設定についてはauto_model_poptimizer.pyのcompare_models関数を参照。

```.\model_training\build_model.py```

モデル構築を行うスクリプト。
build_model.py内のCSV_FILE_NAME, SAVE_FILE_NAME, MODELを必要に応じて変える。
2024年5月13日現在、実験の結果性能が一番いいet（Extra Trees Classifier）をデフォルトにしている。
構築したモデルはdataフォルダに出力される。


### 予測
```python ./prediction/predict.py```
1. 声優を予測したい音声ファイルを./prediction/test_dataに格納する
2. 必要であれば./prediction/predict.py内の使用するモデルを変更する


## フォルダ構成
```
> tree
C:.
├─common            ：DB接続、ログ出力など共通で使うプログラム
├─data              ：db, csv, 作品リスト（product_list.txt）、学習済モデル（.pkl）などです
├─data_collection   ：データ収集用プログラム
├─model_training    ：モデリング用
├─prediction        ：構築済みモデルで実際に予測を行うプログラム
│  └─test_data      ：予測したいファイルを入れるフォルダ
├─report            ：システムの概要や評価をまとめたレポートたち
└─tests             ：pytest用
```