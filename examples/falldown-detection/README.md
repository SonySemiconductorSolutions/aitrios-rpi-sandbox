# Falldown detection based on HigherHRNet

## システム概要

- **目標**: 人体の姿勢をリアルタイムで監視し、転倒行為の有無を検出する。
- **用いた技術**:
  - HigherHRNet モデルを使用して人体のキーポイントを検出する。
  - 幾何学的計算を応用して人体の姿勢の特徴（例えば、体の角度）を分析する。
  - Raspberry Pi + RaspberryPi AI Camera を組み合わせて、リアルタイムの推論と処理を行う。
 
## ハードウェア環境
- **処理ユニット**: Raspberry Pi（Raspberry Pi 4 Model B またはそれ以上の機種を推奨）
- **カメラモジュール**: Raspberry Pi AI Camera
- **ネットワーク接続**: モデルと依存関係をダウンロードするためにインターネットに接続する必要がある

---

## ソフトウェア環境
- Python 3.11 (Raspberry Piに搭載済み)
- Python 3- OpenCV 4.6（APT を通じてインストール）
- Python 3- Munkres

---

## インストールと使用方法

1. **依存関係のインストール**:
   ```bash
   apt install python3-opencv imx500-all python3-munkres
   ```

2. **HigherHRNet モデルのダウンロード**：
   *もしすでに APT を通じて imx500-all をインストールしている場合は、このステップをスキップできます*
   [RaspberryPi Repo](https://github.com/raspberrypi/imx500-models/raw/refs/heads/main/imx500_network_higherhrnet_coco.rpk) から必要なモデルファイルをダウンロードしてください。

3. **システムの実行**:
   ```bash
   python main.py --model <path to model> 
   ```

4. **主なパラメータの説明**:
   - `--model`: RPK モデルの位置。apt を通じて imx500-all をインストールした場合は変更する必要がない。
   - `--detection-threshold`: 人体検出器の閾値。 `[0,1]` の範囲で通常は変更する必要がない。
   - `--fall-threshold`: 転倒判定閾値。 `[0,100]` の範囲で 45 が推奨。通常は変更する必要がないが、カメラの角度などに応じて必要に応じて変更する。  
