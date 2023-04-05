import cv2
import numpy as np
from PIL import Image
import os
import dlib
import math

# # 눈에 대한 기울기 찾기
# def eye_grad(eye_list):
#     eye1_x, eye1_y, eye1_w, eye1_h = eye_list[0]
#     eye2_x, eye2_y, eye2_w, eye2_h = eye_list[1]
    
#     eye1_xmean, eye1_ymean = (eye1_x*2 + eye1_w)/2 , (eye1_y*2 + eye1_h)/2
#     eye2_xmean, eye2_ymean = (eye2_x*2 + eye2_w)/2 , (eye2_y*2 + eye2_h)/2

#     return (int(eye1_xmean), int(eye1_ymean), int(eye2_xmean), int(eye2_ymean))


def eye_degree(left_eye, right_eye):
    x1, y1 = left_eye[0], left_eye[1]
    x2, y2 = right_eye[0], right_eye[1]
    
    radian = math.atan2(abs(y2-y1),x2-x1)
    degree = (radian*180)/math.pi
    
    return degree


# image preprocessing
"""
    os.remove() os.unlink() : 파일 삭제
    
    - 파일 개수 확인
    folder_path = "path/to/folder"
    file_list = os.listdir(folder_path)
    file_count = len(file_list)
    print(file_count)
    
    folder_path = "path/to/folder"
    file_count = sum([len(files) for r, d, files in os.walk(folder_path)])
    print(file_count)
    
"""

# cc_loc = "/Users/choisihyun/Downloads/sanhak/opencv/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml"
# cc = cv2.CascadeClassifier(cc_loc)


# eye_cc = cv2.CascadeClassifier("/Users/choisihyun/Downloads/sanhak/opencv/data/haarcascades_cuda/haarcascade_eye.xml")


#img_loc = "/Users/choisihyun/Downloads/g9bJuVaboD4Rvdh4WGLmyeXzlU77UrtaHEraElSZitHR4aI7YCzrRIDSUxJke4LeZSSNY3FuXycUZ8Ou1vB1nw.webp"
img_loc = "/Users/choisihyun/Downloads/test4.jpeg"
image = cv2.imread(img_loc)

# 얼굴 검출기 및 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/choisihyun/dlproject/AnimalFaceClassifier/dlib_face_detect/shape_predictor_68_face_landmarks.dat')

# 이미지에서 얼굴 검출
faces = detector(image)

print(faces)

# 검출된 얼굴에 대한 랜드마크 예측
for face in faces:
    landmarks = predictor(image, face)
    print(landmarks)
    # landmarks 변수는 예측된 랜드마크 좌표를 포함합니다.

# 검출된 얼굴에 대해 랜드마크 예측
for face in faces:
    landmarks = predictor(image, face)
    
    # 예측된 랜드마크 출력
    for n in range(0, 68):
        if n==39:
            x1 = landmarks.part(n).x
            y1 = landmarks.part(n).y
        if n==42:
            x2 = landmarks.part(n).x
            y2 = landmarks.part(n).y
            cv2.line(image, (x1,y1),(x2,y2),(255,0,120),2)
            print(eye_degree((x1,y1),(x2,y2)))
            dg = eye_degree((x1,y1),(x2,y2))
                
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 0, 255), 2)


#얼굴

dets = detector(image, 1)
print(dets)
## 이제부터 인식된 얼굴 개수만큼 반복하여 얼굴 윤곽을 표시한다.
# k: 얼굴 인덱스, d: 얼굴 좌표

for k, d in enumerate(dets): 
    shape = predictor(image, d) #shape: 얼굴 랜드마크 추출 
    print(shape.num_parts) #추출된 점은 68개.



(h, w) = image.shape[:2]
M = cv2.getRotationMatrix2D((h//2, w//2), 360-dg, 1.0)
image = cv2.warpAffine(image, M, (w, h))

cv2.imshow('test_img', image)
cv2.waitKey()
cv2.destroyAllWindows()






# face = cc.detectMultiScale(image)
# eye = eye_cc.detectMultiScale(image,minNeighbors=7)
# #만약 찾은게 2개가 딱 되면 눈 기울기 찾아서 이미지 회전시키기 (얼굴 탐지 사각형도)
# x1,y1,x2,y2 = eye_grad(eye)
# x,y,w,h = face[0]




# (h, w) = image.shape[:2]
# M = cv2.getRotationMatrix2D((h//2, w//2), 45, 1.0)
# rotated_45 = cv2.warpAffine(image, M, (w, h))

# cv2.rectangle(image, (x, y, w, h), (255, 0, 255), 2)
# cv2.line(image, (x1,y1),(x2,y2),(255,0,255),10)
# cv2.imshow('src', rotated_45)
# cv2.waitKey()
# cv2.destroyAllWindows()


# for (x, y, w, h) in eye:
#     cv2.rectangle(image, (x, y, w, h), (255, 0, 255), 2)
#     cv2.imshow('src', image)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
# print(face)


# print(eye)
# print(face)

# if face.shape == (1,4):
#     print(4)
#     (x, y, w, h) = face[0]
#     print(x,y,w,h)
# image = image[y:y+h,x:x+w]
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#조건문 : 아무것도 찾지 못함

#조건문 : 많이 찾음 -> 분류기 이웃개수 재설정

# face_img = Image.fromarray(image)
# face_img = face_img.resize((256, 256))
# face_img.save('/Users/choisihyun/Downloads/save_test6','PNG') # save PIL image

# cv2.imshow('src', image[y:y+h,x:x+w])
# cv2.waitKey()
# cv2.destroyAllWindows()





# 이미지를 정규화하는 이유 
# https://goodtogreate.tistory.com/entry/Neural-Network-적용-전에-Input-data를-Normalize-해야-하는-이유

"""
크기 조정(Resizing): 모델에 입력되는 이미지는 일반적으로 동일한 크기여야 합니다. 따라서 훈련 이미지의 크기를 모두 동일하게 조정하는 것이 일반적입니다.
정규화(Normalization): 모든 이미지 픽셀 값은 0과 255 사이에 있습니다. 정규화는 모든 픽셀 값을 0과 1 사이로 조정합니다. 이는 모델이 이미지를 더 잘 처리할 수 있도록 하며, 모델의 수렴 속도를 높일 수 있습니다.
데이터 증강(Data Augmentation): 데이터 증강은 이미지를 회전, 이동, 대칭 등으로 변형하여 데이터를 늘리는 것입니다. 이는 모델이 더 일반화되게 학습할 수 있도록 도와줍니다. 또한, 과적합(overfitting)을 방지할 수 있습니다.
컬러 채널 변환(Color Channel Conversion): RGB 이미지는 세 개의 채널(R, G, B)로 이루어져 있습니다. 그러나 일부 모델은 다른 채널 수를 요구합니다. 예를 들어, 흑백 이미지는 단일 채널을 사용합니다. 이 경우, 이미지를 흑백으로 변환해야 합니다.

-> 카스케이드 분류기에서 

따라서 dlib은 보다 정확한 얼굴 인식을 제공할 가능성이 높습니다. 
그러나 학습된 모델이 더 복잡하므로 처리 속도는 느릴 수 있습니다. 
Cascade는 상대적으로 간단하며 빠르게 얼굴을 인식할 수 있지만 정확도는 덜할 수 있습니다.

"""



"""
라이브러리

import cv2
import numpy as np
from PIL import Image
import os


함수 
    눈 기울기 찾기 -> 각도 찾기 ->  이미지 회전
    사진 전체 기울여주는 함수 
    
def eye_grad(eye_list):
    eye1_x, eye1_y, eye1_w, eye1_h = eye_list[0]
    eye2_x, eye2_y, eye2_w, eye2_h = eye_list[1]
    
    eye1_xmean, eye1_ymean = (eye1_x*2 + eye1_w)/2 , (eye1_y*2 + eye1_h)/2
    eye2_xmean, eye2_ymean = (eye2_x*2 + eye2_w)/2 , (eye2_y*2 + eye2_h)/2

    return (int(eye1_xmean), int(eye1_ymean), int(eye2_xmean), int(eye2_ymean))


def rotate() :
    ~~~



분류기 초기 설정

face_cc_loc = ""
eye_cc_loc = ""

face_cc = cv2.CascadeClassifier(face_cc_loc)
eye_cc = cv2.CascadeClassifier(eye_cc_loc)


celeb_dir = {
    동물 : [연예인]
}


파일 찾기 -> os 나 keras 통해서 불러오기
반복문 : 폴더 설정 

    folder_loc = "폴더위치 설정"

    반복문 : 사진 하나하나에 대한 전처리 단계 
        
        이미지 불러오기
        image = cv2.imread(img_loc)

        
        반복문
            얼굴 반복문 -> 무조건 하나 (아예 없으면 규제 감소, 너무 많으면 규제 빡새게)
                얼굴 탐지 (파라미터 조정) -> 리스트 하나 남을떄 까지만
            
            눈 반복문 -> 아님말고
                눈 탐지 (파라미터 조정) -> 리스트 두개 남을 때 까지만
                
        // 얼굴 위치랑 눈 기울기 찾았음
        1. 얼굴 위치만 -> 바로 추출하여 저장

        2. 얼굴 위치 + 눈 기울기 -> 사진 자체를 회전 +(얼굴 각도 맞게 추출할 수 있게 각도 파악)

        // 얼굴 추출 완료

        얼굴 정규화 및 전처리 -> 사진 저장 // 끝








"""