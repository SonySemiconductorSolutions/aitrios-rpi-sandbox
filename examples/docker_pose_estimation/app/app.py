# BSD 2-Clause License
# 
# Copyright (c) 2021, Raspberry Pi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import sys
import time
import numpy as np
import cv2
from flask import Flask, render_template, Response

from picamera2 import Picamera2, MappedArray, CompletedRequest
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_highernet import \
    postprocess_higherhrnet

# Flask アプリの初期化
app = Flask(__name__)

# グローバル変数
args = None
imx500 = None
intrinsics = None
picam2 = None
drawer = None

# 撮影画像サイズ（高さ, 幅） ※サンプルに合わせています
WINDOW_SIZE_H_W = (480, 640)

# Pose estimation の結果保持用（必要に応じてグローバル変数として利用）
last_boxes = None
last_scores = None
last_keypoints = None

def ai_output_tensor_parse(metadata: dict):
    """
    カメラのメタデータから出力テンソルを取得し、pose estimation 用に解析する。
    戻り値: boxes, scores, keypoints
    """
    global last_boxes, last_scores, last_keypoints
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=np_outputs,
            img_size=WINDOW_SIZE_H_W,
            img_w_pad=(0, 0),
            img_h_pad=(0, 0),
            detection_threshold=args.detection_threshold,
            network_postprocess=True
        )
        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes = [np.array(b) for b in boxes]
            last_scores = np.array(scores)
    return last_boxes, last_scores, last_keypoints

def get_drawer():
    """
    COCODrawer を生成（labels は intrinsics.labels から取得）。
    ※ 本来は pose 用の描画ができるように設定されています。
    """
    categories = intrinsics.labels
    categories = [c for c in categories if c and c != "-"]
    return COCODrawer(categories, imx500, needs_rescale_coords=False)

def generate_video_frames():
    """
    通常のカメラ映像を JPEG でストリーミングするジェネレーター。
    左側用の映像として利用します。
    """
    while True:
        frame = picam2.capture_array()
        # Picamera2 は RGB を返すため BGR に変換
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def generate_pose_frames():
    """
    Pose estimation の結果（骨格情報）を白い背景に描画した映像を JPEG でストリーミングするジェネレーター。
    右側用の映像として利用します。
    """
    while True:
        # キャプチャ直前のメタデータから推論結果を取得
        metadata = picam2.capture_metadata()
        boxes, scores, keypoints = ai_output_tensor_parse(metadata)
        # キャプチャ画像からサイズを取得（実際の映像は利用しないので、白画像作成用）
        frame = picam2.capture_array()
        h, w, _ = frame.shape
        # 白背景（BGR で 255,255,255）
        white_bg = np.full((h, w, 3), 255, dtype=np.uint8)
        # Pose 推論結果がある場合、drawer を用いて白背景上に骨格を描画する
        if boxes is not None and len(boxes) > 0 and scores is not None and len(scores) > 0:
            # 第3引数はクラス情報ですが、pose estimation では不要なため np.zeros(scores.shape) を指定
            drawer.annotate_image(white_bg, boxes, scores, np.zeros(scores.shape), keypoints,
                                  args.detection_threshold, args.detection_threshold,
                                  metadata, picam2, "pose")
        ret, jpeg = cv2.imencode('.jpg', white_bg, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Flask エンドポイント
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_feed')
def pose_feed():
    return Response(generate_pose_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk",
                        help="Path to the pose estimation model")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--detection-threshold", type=float, default=0.3,
                        help="Detection threshold for pose estimation")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print network intrinsics then exit")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # IMX500 のインスタンス生成（pose estimation 用モデル）
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        print("Network is not a pose estimation task", file=sys.stderr)
        sys.exit(1)
    # コマンドライン引数で intrinsics を上書き
    for key, value in vars(args).items():
        if key == "labels" and value is not None:
            with open(value, "r") as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()
    if args.print_intrinsics:
        print(intrinsics)
        sys.exit(0)
    
    # Pose 用描画器の生成
    drawer = get_drawer()
    
    # Picamera2 の初期化
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={'FrameRate': intrinsics.inference_rate},
        buffer_count=12
    )
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)
    imx500.set_auto_aspect_ratio()
    
    # Flask アプリ起動
    app.run(host='0.0.0.0', port=5000, debug=False)