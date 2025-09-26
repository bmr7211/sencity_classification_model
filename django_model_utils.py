# Django에서 분류모델 사용 방법

import tensorflow as tf
import numpy as np
from tensorflow import keras
import json
import os
from PIL import Image
import io

class AnimalClassifier:
    """
    Django 서버에서 사용할 동물 분류기 클래스
    """
    
    def __init__(self, model_path, class_info_path):
        """
        분류기를 초기화함
        
        Args:
            model_path: 저장된 모델 파일 경로
            class_info_path: 클래스 정보 JSON 파일 경로
        """
        self.model = None
        self.class_names = []
        self.img_size = 224
        self.num_classes = 10
        
        # 모델과 클래스 정보 로드
        self.load_model(model_path)
        self.load_class_info(class_info_path)
        
    def load_model(self, model_path):
        """
        저장된 모델을 로드함
        """
        try:
            self.model = keras.models.load_model(model_path)
            print(f"모델이 성공적으로 로드됨: {model_path}")
        except Exception as e:
            raise Exception(f"모델 로드 실패: {e}")
    
    def load_class_info(self, class_info_path):
        """
        클래스 정보를 JSON 파일에서 로드함
        """
        try:
            with open(class_info_path, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
                self.class_names = class_info['class_names']
                self.num_classes = class_info['num_classes']
                self.img_size = class_info['input_size']
            print(f"클래스 정보가 성공적으로 로드됨: {len(self.class_names)}개 클래스")
        except Exception as e:
            raise Exception(f"클래스 정보 로드 실패: {e}")
    
    def preprocess_image(self, image_data, is_file_path=False):
        """
        이미지를 모델 입력 형태로 전처리함
        
        Args:
            image_data: 이미지 파일 경로 또는 바이트 데이터
            is_file_path: image_data가 파일 경로인지 여부
        
        Returns:
            전처리된 이미지 배열
        """
        try:
            if is_file_path:
                # 파일 경로에서 이미지 로드
                img = Image.open(image_data)
            else:
                # 바이트 데이터에서 이미지 로드 (Django 업로드 파일)
                img = Image.open(io.BytesIO(image_data))
            
            # RGB로 변환 (RGBA나 Grayscale인 경우)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 크기 조정
            img = img.resize((self.img_size, self.img_size))
            
            # 넘파이 배열로 변환
            img_array = np.array(img)
            
            # 배치 차원 추가 및 정규화
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype(np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            raise Exception(f"이미지 전처리 실패: {e}")
    
    def predict(self, image_data, is_file_path=False, top_k=3):
        """
        이미지에 대해 동물 종을 예측함
        
        Args:
            image_data: 이미지 파일 경로 또는 바이트 데이터
            is_file_path: image_data가 파일 경로인지 여부
            top_k: 상위 몇 개 예측 결과를 반환할지
        
        Returns:
            예측 결과 딕셔너리
        """
        try:
            # 이미지 전처리
            img_array = self.preprocess_image(image_data, is_file_path)
            
            # 예측 수행
            predictions = self.model.predict(img_array, verbose=0)
            predicted_probs = predictions[0]
            
            # 상위 k개 예측 결과 생성
            top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
            
            top_predictions = []
            for idx in top_indices:
                top_predictions.append({
                    'class_name': self.class_names[idx],
                    'confidence': float(predicted_probs[idx]),
                    'confidence_percent': round(float(predicted_probs[idx]) * 100, 2)
                })
            
            # 모든 클래스별 확률
            all_predictions = {}
            for i, class_name in enumerate(self.class_names):
                all_predictions[class_name] = {
                    'confidence': float(predicted_probs[i]),
                    'confidence_percent': round(float(predicted_probs[i]) * 100, 2)
                }
            
            return {
                'success': True,
                'predicted_class': top_predictions[0]['class_name'],
                'confidence': top_predictions[0]['confidence'],
                'confidence_percent': top_predictions[0]['confidence_percent'],
                'top_predictions': top_predictions,
                'all_predictions': all_predictions
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self):
        """
        모델 정보를 반환함
        """
        return {
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'input_size': self.img_size,
            'model_loaded': self.model is not None
        }

# Django views.py에서 사용할 수 있는 함수들

def initialize_classifier(model_dir):
    """
    분류기를 초기화하고 전역 변수로 저장함
    Django의 settings.py나 앱 초기화 시에 호출함
    """
    model_path = os.path.join(model_dir, 'animal_classifier.h5')
    class_info_path = os.path.join(model_dir, 'class_info.json')
    
    global animal_classifier
    animal_classifier = AnimalClassifier(model_path, class_info_path)
    return animal_classifier

def predict_uploaded_image(uploaded_file, top_k=3):
    """
    Django에서 업로드된 파일에 대해 예측을 수행함
    
    Args:
        uploaded_file: Django의 UploadedFile 객체
        top_k: 상위 몇 개 예측 결과를 반환할지
    
    Returns:
        예측 결과 딕셔너리
    """
    try:
        # 파일 내용을 바이트로 읽음
        image_data = uploaded_file.read()
        
        # 예측 수행
        result = animal_classifier.predict(image_data, is_file_path=False, top_k=top_k)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'예측 중 오류 발생: {str(e)}'
        }

def predict_image_path(image_path, top_k=3):
    """
    파일 경로로부터 이미지 예측을 수행함
    
    Args:
        image_path: 이미지 파일 경로
        top_k: 상위 몇 개 예측 결과를 반환할지
    
    Returns:
        예측 결과 딕셔너리
    """
    try:
        result = animal_classifier.predict(image_path, is_file_path=True, top_k=top_k)
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'예측 중 오류 발생: {str(e)}'
        }

# Django에서 사용할 예제 view 코드
"""
# views.py 예제

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .utils.django_model_utils import predict_uploaded_image, animal_classifier
import json

@csrf_exempt
@require_POST
def predict_animal_api(request):
    '''
    동물 이미지 분류 API 뷰
    POST 요청으로 이미지 파일을 받아 동물 종을 예측함
    '''
    try:
        # 업로드된 파일 확인
        if 'image' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': '이미지 파일이 필요함'
            }, status=400)
        
        uploaded_file = request.FILES['image']
        
        # 파일 형식 확인
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        if uploaded_file.content_type not in allowed_types:
            return JsonResponse({
                'success': False,
                'error': '지원하지 않는 파일 형식임 (JPEG, PNG, WebP만 지원)'
            }, status=400)
        
        # 예측 수행
        result = predict_uploaded_image(uploaded_file, top_k=3)
        
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'서버 오류: {str(e)}'
        }, status=500)

def predict_animal_page(request):
    '''
    동물 분류 웹 페이지 뷰
    '''
    context = {
        'model_info': animal_classifier.get_model_info()
    }
    return render(request, 'animal_classifier.html', context)
"""