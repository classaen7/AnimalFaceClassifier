import cv2
import numpy as np
from PIL import Image
import os
import dlib
import math

# 눈 랜드마크로 기울기 찾기
def eye_degree(left_eye, right_eye):
    x1, y1 = left_eye.x, left_eye.y
    x2, y2 = right_eye.x, right_eye.y
    
    radian = math.atan2(abs(y2-y1),x2-x1)
    degree = (radian*180)/math.pi
    
    return degree

# 이미지 회전, 얼굴 추출하는 함수
# def find_face(image):
    
    
#     image = image[:]
#     return image


dir = {
    "강아지" : ["박보영","아이유","강다니엘","송중기"],
    "고양이" : ["강동원", "이준기","한예슬","뉴진스 해린"],
    "토끼" : ["정국", "박지훈","트와이스 나연","장원영" ],
    "공룡" : ["김우빈", "공유","송지효","신민아"],
    "곰" : ["안재홍", "김대명","레드벨벳 슬기","엔믹스 오해원"]     
}

default_path = defualt_path = 'C:/Users/CHOI/OneDrive/바탕 화면/산학프로젝트 인공지능/데이터 크롤링/'
predictor_path = '/Users/choisihyun/dlproject/AnimalFaceClassifier/dlib_face_detect/shape_predictor_68_face_landmarks.dat'

# 얼굴 검출기 생성
detector = dlib.get_frontal_face_detector()
# 랜드마크 검출기 생성
predictor = dlib.shape_predictor(predictor_path)

for animal in dir:
    for celeb in animal:
        # animal : 동물, celeb : 연예인 이름

        folder_path = default_path + animal + '/' + celeb
        file_list = os.listdir(folder_path)
        
        for img_name in range(0,len(file_list)):
            #이미지 불러오기
            img_loc = folder_path + '/' + file_list[img_name]
            image = cv2.imread(img_loc)
            
            #1. 얼굴 검출
            faces = detector(image)
            if len(faces) == 1:
                landmarks = predictor(image, faces[0])
                
                if landmarks.num_parts == 68:
                    #눈의 각도로 이미지 회전
                    degree = eye_degree(landmarks.part(39), landmarks.part(42))
                    
                    if degree != 0: #이미지를 회전 시켜야 하면
                        if  landmarks.part(39).y > landmarks.part(42).y:
                            rot_degree = 360 - degree 
                        else:
                            rot_degree = degree 
                     
                        #이미지 회전하고 다시 얼굴 탐지
                        (h, w) = image.shape[:2]
                        M = cv2.getRotationMatrix2D((h//2, w//2), rot_degree, 1.0)
                        image = cv2.warpAffine(image, M, (w, h))
                    
                    
                    #회전을 하거나 안하거나 얼굴만 다시 인식하여 이미지 추출
                    faces = detector(image)
                    
                    #얼굴 하나 찾으면 이미지 그걸로 변환
                    if len(faces) == 1:
                        face = faces[0]
                        image = image[face.top(),face.bottom(),face.left():face.right()]
                        
                        #변환된 이미지 출력
                        cv2.imshow('test_img', image)
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                        
                        
                        #이미지 파일 저장                        
                        face_img = Image.fromarray(image)
                        #크기조정
                        face_img = face_img.resize((256, 256))
                        
                        #얼굴 정규화같은 추가 전처리 과정 필요할 수도
                        
                        
                        face_img.save('/Users/choisihyun/Downloads/save_test6','PNG') # save PIL image
                        

                else: 
                    #랜드마크가 정확히 (인덱스를 할 수 있게) 관측하지 못한 경우 -> 그냥 얼굴만 추출
                    face = faces[0]
                    image = image[face.top(),face.bottom(),face.left():face.right()]
                    
                    
                    #이미지 파일 저장                        
                    face_img = Image.fromarray(image)
                    #크기조정
                    face_img = face_img.resize((256, 256))
                    
                    #얼굴 정규화같은 추가 전처리 과정 필요할 수도
                    face_img.save('/Users/choisihyun/Downloads/save_test6','PNG') # save PIL image
                    
                    continue
                    
            
            else:
                #얼굴이 여러개나 0개 추출이면 그냥 패스
                continue
            
            
            #2. 검출된 얼굴의 랜드마크 찾기










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

"""
# 얼굴 검출기 생성
detector = dlib.get_frontal_face_detector()
# 랜드마크 검출기 생성
predictor = dlib.shape_predictor(predictor_path)


img_loc = "/Users/choisihyun/Downloads/202105061428801176_1.jpg"
image = cv2.imread(img_loc)
            
#1. 얼굴 검출
faces = detector(image)
if len(faces) == 1:
    landmarks = predictor(image, faces[0])
    
    if landmarks.num_parts == 68:
        #눈의 각도로 이미지 회전
        degree = eye_degree(landmarks.part(39), landmarks.part(42))
        
        if degree != 0: #이미지를 회전 시켜야 하면
            if  landmarks.part(39).y > landmarks.part(42).y:
                rot_degree = 360 - degree 
            else:
                rot_degree = degree 
            
            #이미지 회전하고 다시 얼굴 탐지
            (h, w) = image.shape[:2]
            M = cv2.getRotationMatrix2D((h//2, w//2), rot_degree, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        
        #회전을 하거나 안하거나 얼굴만 다시 인식하여 이미지 추출
        faces = detector(image)
        
        #얼굴 하나 찾으면 이미지 그걸로 변환
        if len(faces) == 1:
            face = faces[0]
            image = image[face.top():face.bottom(),face.left():face.right()]
            
            #변환된 이미지 출력
            cv2.imshow('test_img', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
            
"""
