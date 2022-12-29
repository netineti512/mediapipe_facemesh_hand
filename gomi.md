---
title: [MediaPipe]手の座標データ(21個)と特定の顔の座標データ(4個)を結合させる。
tags: MediaPipe CSV
author: netineti512
slide: false
---
以下の記事で取得した顔座標から必要な点を取り出す。
[MediaPipeで顔と手の座標データを出力する方法[CSV]]()

処理をする画像です。
![a.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2702119/3bc64725-7a34-3db5-fb76-9b85ffcf52e4.jpeg)

また、顔の座標点の画像です。
![aa.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2702119/e3837124-7bb3-061d-9558-6bfca06f3f80.jpeg)

以下の4点を取得します。
![a.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2702119/669720d0-7ee0-abf3-f8bf-7caac4400433.jpeg)

顔の座標データです。
468行あります。
[landmark_face.csv, landmark_hand.csv](https://drive.google.com/drive/folders/10yytVY-0LR-fA7255X6oI5C1wO-Z05BV?usp=sharing)
```landmark_face.csv
#x軸, y軸, z軸, R, G, B
0.5134931206703186,0.48759475350379944,-0.0160954762250185,205,116,102
0.5119926929473877,0.45384481549263,-0.044657569378614426,221,182,153
0.5115312337875366,0.4635353684425354,-0.021530626341700554,208,165,131
.........
.........
0.565422534942627,0.3468426465988159,0.005376789253205061,69,42,35
0.5691334009170532,0.3415432274341583,0.005679022986441851,32,25,15
```

顔の座標データから点6, 10. 27, 257を取り出し、landmark_hand.csvと合わせて1つのデータにしたいと思います。

顔から4点を取り出し、手の座標データに追加するソースコードです。
```select_four_landmarks.py
import csv
import pandas as pd 
import numpy as np

#6と10と27と257
#467箇所
#行列を入れ替えた顔座標データを一次元で読み込む
with open("./result/landmark_face.csv") as file_name1:
    array1 = np.loadtxt(file_name1, delimiter=",").ravel()

#x軸, y軸, z軸でそれぞれ分ける
face_x = array1[::6]
face_y = array1[1::6]
face_z = array1[2::6]

#点6,10,27,257取り出して格納
face_x1 = []
face_y1 = []
face_z1 = []
list = [6, 10, 27, 257]
for i in list:
    face_x1.append(face_x[i])
    face_y1.append(face_y[i])
    face_z1.append(face_z[i])

#手の座標データを開く
with open("./result/landmark_hand.csv") as file_name2:
    array2 = np.loadtxt(file_name2, delimiter=",").ravel()
    #print(array)
hand_x = array2[::3]
hand_y = array2[1::3]
hand_z = array2[2::3]

#手の座標データに顔の座標データを結合させる
x = np.append(hand_x, face_x1)
y = np.append(hand_y, face_y1)
z = np.append(hand_z, face_z1)

#x, y, zをまとめる
xyz = []
list = [x, y, z]
for i in list:
    xyz.extend(i)
np.savetxt('./result/landmark_hand21_face4.csv', xyz, delimiter=',')
```

出力された顔の、4点の座標データです。(×3軸)
```landmark_face_4.csv
0.50649184    #6_x
0.500466168   #10_x
0.454794437   #27_x
0.550741434   #257_x
0.369761795   #6_y
0.237196505   #10_y
0.337439477   #27_y
0.331445545   #257_y
-0.029559772  #6_z
-0.029976776  #10_z
-0.012285766  #27_z
-0.006326044  #257_z
```
手の座標データです。
21点あります。(×3軸)
```landmark_hand.csv
0.504073203,0.814438343,3.91E-08,0.45767349,0.768936157,0.000312077,0.439563543,0.721087813,-0.008808918,0.425133765,0.687695742,-0.020445434,0.409348369,0.65940398,-0.033669636,0.489157706,0.648819566,-0.018444018,0.492025942,0.58075875,-0.03804915,0.49744001,0.534264326,-0.050505653,0.50370723,0.494695395,-0.058645058,0.503715634,0.663659096,-0.028175741,0.443882823,0.62402916,-0.054487202,0.407770842,0.648138106,-0.062590681,0.390238494,0.677609742,-0.064250305,0.506061792,0.691279113,-0.037629627,0.438479215,0.675933599,-0.062441029,0.415851235,0.71006012,-0.061497707,0.411849707,0.737843037,-0.055186719,0.500449598,0.721720636,-0.046630431,0.445799142,0.719092548,-0.066425182,0.430722862,0.741868556,-0.067274623,0.430717707,0.759190679,-0.063078843
```

上のlandmark_face_4.csvとlandmark_hand.csvを組み合わせたCSVファイルです。
75行あります。
```landmark_hand21_face4.csv
0.504073203
0.45767349
...........
...........
-0.012285766
-0.006326044
```
終わり
>他の記事
・[MediaPipeで顔と手の座標データを出力する方法[CSV]](https://qiita.com/netineti512/items/cd642e130887a4636ec1)
・[[MediaPipe]手の座標データ(21個)と特定の顔の座標データ(4個)を結合させる。](https://qiita.com/netineti512/items/64bf47d5a61b474a55b5)
・[【MediaPipe】顔と手の座標データを正規化する(CSVファイル)](https://qiita.com/netineti512/items/34fb10a172588a71fe01)
