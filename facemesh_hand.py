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
