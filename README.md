# 同人ダメ絶対音感（doujin-perfect-pitch）

## これは？

音声ファイルを入力すると、どのDLsite声優の声か識別できるシステム。
以下の3つの役割がある。

### データ収集
Webスクレイピングによるデータ収集と特徴量抽出を頑張る

### モデリング
機械学習によるモデルの訓練を頑張る

### 予測
予測を頑張る

## 使用方法
###


### Python以外の環境構築
subprocessでffmpegと7zipを呼び出しているので、インストールしてパスを通す必要がある。

- ffmpeg：体験版が動画ファイルの場合にmp3でダウンロードして使用する際に使用

- 7zip：体験版zipファイルがDeflate64で圧縮されていた場合に使用


## フォルダ構成
### 直下
README.md, .gitignore, error.log

### data
dbファイル、csvファイル、作品リストファイル（product_list.txt）、学習済みモデル（.pkl）が入る

### commmon
DB接続、ログ出力など共通で使うプログラム

### data_collection
データ収集用

### model_training
モデリング用

### prediction
予測用

### tests
pytest用