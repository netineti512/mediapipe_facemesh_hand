今回は以下のような画像のシーンを想定して、MediaPipeを使い顔と手の座標を同時に出力するプログラムを作成した。

処理をする画像です。
![a.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2702119/3bc64725-7a34-3db5-fb76-9b85ffcf52e4.jpeg)


また、顔の座標点の画像です。
![S__7790595.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2702119/b215fa9f-6a2c-dabf-bfea-3546f82d8939.jpeg)

全体のソースコードです。
以下のサイトを参考にしました。
[【MediaPipe】FaceMeshで取得した468点の座標から3次元顔を再現できるか？](https://qiita.com/Esp-v2/items/ba48ea3b2491f3d6bbb4)
[MediaPipeを使って手から取得した骨格座標の情報をCSVに保存する](https://qiita.com/h-ueno2/items/b8dd54b396add5c3b12a)
```MediaPipe_face_hand.py
import encodings
import cv2
import glob
import os
import csv

import mediapipe as mp

import numpy as np
import pandas as pd
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def fields_name():
    # CSVのヘッダを準備
    fields = []
    fields.append('file_name')
    for i in range(21):
        fields.append(str(i)+'_x')
        fields.append(str(i)+'_y')
        fields.append(str(i)+'_z')
    return fields

if __name__ == '__main__':
    # 元の画像ファイルの保存先を準備
    resource_dir = r'./image'
    # 対象画像の一覧を取得
    file_list = glob.glob(os.path.join(resource_dir, "a.jpg"))

    # 保存先の用意
    save_csv_dir = './result'
    os.makedirs(save_csv_dir, exist_ok=True)
    save_csv_name = 'landmark_hand.csv'
    save_image_dir = './image_landmark'
    os.makedirs(save_image_dir, exist_ok=True)

    with mp_hands.Hands(static_image_mode=True,
            max_num_hands=1, # 検出する手の数（最大2まで）
            min_detection_confidence=0.5) as hands, \
        open(os.path.join(save_csv_dir, save_csv_name), 
            'w', encoding='utf-8', newline="") as f:

        # csv writer の用意
        writer = csv.DictWriter(f, fieldnames=fields_name())
        writer.writeheader()

        for file_path in file_list:
            # 画像の読み込み
            image = cv2.imread(file_path)

            # 鏡写しの状態で処理を行うため反転
            image = cv2.flip(image, 1)

            # OpenCVとMediaPipeでRGBの並びが違うため、
            # 処理前に変換しておく。
            # CV2:BGR → MediaPipe:RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 推論処理
            results = hands.process(image)

            # 前処理の変換を戻しておく。
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not results.multi_hand_landmarks:
                # 検出できなかった場合はcontinue
                continue

            # ランドマークの座標情報
            landmarks = results.multi_hand_landmarks[0]

            # CSVに書き込み
            record = {}
            record["file_name"] = os.path.basename(file_path)
            for i, landmark in enumerate(landmarks.landmark):
                record[str(i) + '_x'] = landmark.x
                record[str(i) + '_y'] = landmark.y
                record[str(i) + '_z'] = landmark.z
            writer.writerow(record)

            # 元画像上にランドマークを描画
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            # 画像を保存
            cv2.imwrite(
                os.path.join(save_image_dir, os.path.basename(file_path)),
                cv2.flip(image, 1))




# 初期設定
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def main():
    image = cv2.imread('./image/a.jpg')
    results = holistic.process(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # ランドマークの座標dataframeとarray_imageを取得
    df_xyz, landmark_image = landmark(image)

    # ランドマーク記載画像を整形、出力
    landmark_image = cv2.cvtColor(
        landmark_image, cv2.COLOR_BGR2RGB)  # BGRtoRGB
    landmark_image = Image.fromarray(landmark_image.astype(np.uint8))
    # landmark_image.show()

    height, width, channels = image.shape[:3]
    # ランドマークの色情報を取得
    df_rgb = color(image, df_xyz, height, width)

    # xyzとrgb結合
    df_xyz_rgb = pd.concat([df_xyz, df_rgb], axis=1)
    df_xyz_rgb.to_csv('./result/landmark_face.csv', header=False, index=False)


# ランドマークの画素値を取得する
def color(image, xyz, height, width):
    label = ['r', 'g', 'b']
    data = []
    for _ in range(len(xyz)):
        x = int(xyz.iloc[_, 0]*width)
        y = int(xyz.iloc[_, 1]*height)

        b = int(image[y, x, 0])
        g = int(image[y, x, 1])
        r = int(image[y, x, 2])

        data.append([r, g, b])

    df = pd.DataFrame(data, columns=label)
    return df


# ランドマークの座標を取得する
def face(results, annotated_image):
    label = ["x", "y", "z"]
    data = []
    if results.face_landmarks:
        # ランドマークを描画する
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        for landmark in results.face_landmarks.landmark:
            data.append([landmark.x, landmark.y, landmark.z])

    else:  # 検出されなかったら欠損値nanを登録する
        data.append([np.nan, np.nan, np.nan])

    df = pd.DataFrame(data, columns=label)
    return df


# imageに対してmediapipeでランドマークを表示、出力する
def landmark(image):
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()

    # ランドマーク取得
    df_xyz = face(results, annotated_image)
    return df_xyz, annotated_image


if __name__ == "__main__":
    main()
```
指の座標を取得したときに出力された画像です。
顔の座標は表示できませんでした。。。
![aa.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2702119/93033e6d-c516-3a3e-edee-0d154bccd9d7.jpeg)

手の座標データです。
```landmark_hand.csv
file_name,0_x,0_y,0_z,1_x,1_y,1_z,2_x,2_y,2_z,3_x,3_y,3_z,4_x,4_y,4_z,5_x,5_y,5_z,6_x,6_y,6_z,7_x,7_y,7_z,8_x,8_y,8_z,9_x,9_y,9_z,10_x,10_y,10_z,11_x,11_y,11_z,12_x,12_y,12_z,13_x,13_y,13_z,14_x,14_y,14_z,15_x,15_y,15_z,16_x,16_y,16_z,17_x,17_y,17_z,18_x,18_y,18_z,19_x,19_y,19_z,20_x,20_y,20_z
a.jpg,0.5040732026100159,0.8144383430480957,3.908506229777231e-08,0.45767349004745483,0.7689361572265625,0.0003120772307738662,0.4395635426044464,0.7210878133773804,-0.008808918297290802,0.42513376474380493,0.6876957416534424,-0.020445434376597404,0.40934836864471436,0.6594039797782898,-0.03366963565349579,0.48915770649909973,0.6488195657730103,-0.018444018438458443,0.4920259416103363,0.5807587504386902,-0.03804915025830269,0.49744001030921936,0.534264326095581,-0.05050565302371979,0.503707230091095,0.49469539523124695,-0.058645058423280716,0.5037156343460083,0.6636590957641602,-0.02817574143409729,0.4438828229904175,0.6240291595458984,-0.0544872023165226,0.4077708423137665,0.6481381058692932,-0.06259068101644516,0.390238493680954,0.6776097416877747,-0.06425030529499054,0.5060617923736572,0.6912791132926941,-0.0376296266913414,0.4384792149066925,0.6759335994720459,-0.06244102865457535,0.4158512353897095,0.7100601196289062,-0.061497706919908524,0.411849707365036,0.7378430366516113,-0.055186718702316284,0.5004495978355408,0.7217206358909607,-0.04663043096661568,0.4457991421222687,0.7190925478935242,-0.06642518192529678,0.4307228624820709,0.7418685555458069,-0.06727462261915207,0.43071770668029785,0.7591906785964966,-0.06307884305715561
```

顔の座標データです。
468行あります。
```landmark_face.csv
0.5134931206703186,0.48759475350379944,-0.0160954762250185,205,116,102
0.5119926929473877,0.45384481549263,-0.044657569378614426,221,182,153
0.5115312337875366,0.4635353684425354,-0.021530626341700554,208,165,131
.........
.........
0.565422534942627,0.3468426465988159,0.005376789253205061,69,42,35
0.5691334009170532,0.3415432274341583,0.005679022986441851,32,25,15
```

>他の記事
・[MediaPipeで顔と手の座標データを出力する方法[CSV]](https://qiita.com/netineti512/items/cd642e130887a4636ec1)
・[[MediaPipe]手の座標データ(21個)と特定の顔の座標データ(4個)を結合させる。](https://qiita.com/netineti512/items/64bf47d5a61b474a55b5)
・[【MediaPipe】顔と手の座標データを正規化する(CSVファイル)](https://qiita.com/netineti512/items/34fb10a172588a71fe01)
