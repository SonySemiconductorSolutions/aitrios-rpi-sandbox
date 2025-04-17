# Home Camera App 
## はじめに
こちらのサンプルアプリケーションについては[Qiita記事](https://qiita.com/SSS-AtsushiNishimura/private/49058923518bf91729ce)でも公開しておりますので、必要に応じてご参照ください。

## セットアップ
submoduleのセットアップについては[aitrios-rpi-application-module-library](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library)を参考にしてください。  
別途、pip installにて下記のパッケージをインストールしてください。

    pip install pygame slack-sdk opencv-python requests lap aiohttp

## サンプルアプリケーション実行
HomeCameraApp.pyを開き、下記の4つを修正してください。

    ALERT_SOUND_PATH = "/your/file/path/hoge.wav"
    SCREENSHOT_FILENAME = "ScreenShot.png"
    OAUTH_TOKEN = 'SlackのOAuth Token'
    CHANNEL_ID = 'SlackのチャンネルID'

修正後、HomeCameraApp.pyを保存してからコマンドを実行します。

    python3 HomeCameraApp.py


## 注意事項
「Slack」は他社が提供するサービスであり、これらの利用によって損害が発生した場合でも責任を負いかねます。  
お客様側でサービス内容をご確認のうえでご利用ください。