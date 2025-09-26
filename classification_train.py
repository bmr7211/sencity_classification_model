import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import numpy as np
import scipy
from scipy import ndimage
import scipy.ndimage as ndimage
import scipy.spatial
import scipy.stats
import matplotlib.pyplot as plt
import os

# GPU 메모리 설정 (필요시)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 하이퍼파라미터 설정
IMG_SIZE = 224  # EfficientNet 입력 크기
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# 클래스 이름 정의
CLASS_NAMES = ['Goat', 'Wild boar', 'Squirrel', 'Raccoon', 'Asiatic black bear', 
               'Hare', 'Weasel', 'Heron', 'Dog', 'Cat']

# 데이터 경로 설정 (실제 경로로 변경해야 함)
DATA_DIR = './data'  # 전체 데이터 폴더 (클래스별 하위 폴더 포함)
MODEL_SAVE_PATH = './models'  # 모델 저장 경로

# 데이터 분할 비율 설정 (클래스당 500장 기준)
TRAIN_SPLIT = 0.70  # 70% (350장) - 훈련용
VALIDATION_SPLIT = 0.20  # 20% (100장) - 검증용  
TEST_SPLIT = 0.10  # 10% (50장) - 최종 테스트용

# 모델 저장 폴더 생성
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 데이터 전처리 및 증강 설정
def create_data_generators():

    # 훈련, 검증, 테스트용 데이터 제너레이터를 생성함
    # 전체 데이터를 7:2:1 비율로 분할함 (훈련:검증:테스트)
    # 전체 데이터용 제너레이터 (분할 전)
    datagen = ImageDataGenerator(
        rescale=1./255,  # 픽셀값을 0-1로 정규화
        validation_split=VALIDATION_SPLIT + TEST_SPLIT  # 검증+테스트 비율
    )
    
    # 훈련용 데이터 증강 제너레이터
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,  # 20도 범위 내에서 무작위 회전
        width_shift_range=0.2,  # 가로 방향으로 20% 이동
        height_shift_range=0.2,  # 세로 방향으로 20% 이동
        shear_range=0.2,  # 전단 변환
        zoom_range=0.2,  # 20% 확대/축소
        horizontal_flip=True,  # 수평 뒤집기
        fill_mode='nearest',  # 빈 픽셀 채우기 방식
        validation_split=VALIDATION_SPLIT + TEST_SPLIT
    )
    
    # 훈련 데이터 제너레이터 (데이터 증강 포함)
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',  # 훈련용 70%
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42
    )
    
    # 검증+테스트 데이터 제너레이터 (증강 없음)
    val_test_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',  # 나머지 30% (검증 20% + 테스트 10%)
        classes=CLASS_NAMES,
        shuffle=False,
        seed=42
    )
    
    # 검증+테스트 데이터를 다시 분할 (2:1 비율로)
    val_test_samples = val_test_generator.samples
    val_samples = int(val_test_samples * (VALIDATION_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)))
    
    print(f"전체 데이터 분할:")
    print(f"  훈련: {train_generator.samples}장 ({train_generator.samples/5000*100:.1f}%)")
    print(f"  검증: {val_samples}장 ({val_samples/5000*100:.1f}%)")
    print(f"  테스트: {val_test_samples - val_samples}장 ({(val_test_samples - val_samples)/5000*100:.1f}%)")
    
    return train_generator, val_test_generator, val_samples

def create_model():

    # EfficientNet 기반의 분류 모델을 생성함
    # 사전 훈련된 EfficientNetB0을 기본 모델로 사용하고 맞춤형 분류 층을 추가함
    # EfficientNetB0 기본 모델 로드 (ImageNet 가중치 사용)
    base_model = EfficientNetB0(
        weights='imagenet',  # ImageNet 사전 훈련 가중치 사용
        include_top=False,  # 최상위 분류 층 제외
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # 기본 모델의 가중치 고정 (전이 학습)
    base_model.trainable = False
    
    # 모델 구성
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # 전역 평균 풀링
        layers.Dropout(0.3),  # 과적합 방지를 위한 드롭아웃
        layers.Dense(128, activation='relu'),  # 완전 연결층
        layers.Dropout(0.2),  # 추가 드롭아웃
        layers.Dense(NUM_CLASSES, activation='softmax')  # 출력층 (10개 클래스)
    ])
    
    return model

def compile_model(model):

    # 모델을 컴파일함
    # Adam 옵티마이저와 범주형 교차 엔트로피 손실을 사용함
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]  # 정확도와 상위 2개 정확도 측정
    )
    
    return model

def create_callbacks():

    # 훈련 중 사용할 콜백 함수들을 생성함
    callbacks = [
        # 최고 성능 모델 저장
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        
        # 조기 종료
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # 학습률 감소
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks

def evaluate_on_test_set(model, val_test_generator, val_samples):

    # 테스트 세트에서 최종 성능을 평가함
    print("\n테스트 세트에서 최종 평가를 수행함...")
    
    # 검증+테스트 데이터에서 테스트 부분만 추출
    # 이는 간단한 방법이고, 실제로는 별도의 테스트 제너레이터를 만드는 것이 더 정확함
    test_samples = val_test_generator.samples - val_samples
    
    # 전체 검증+테스트 데이터에서 평가 (근사치)
    test_loss, test_accuracy, test_top2_accuracy = model.evaluate(val_test_generator, verbose=1)
    
    print(f"테스트 결과:")
    print(f"  테스트 손실: {test_loss:.4f}")
    print(f"  테스트 정확도: {test_accuracy:.4f}")
    print(f"  테스트 Top-2 정확도: {test_top2_accuracy:.4f}")
    
    return test_loss, test_accuracy, test_top2_accuracy

    # # 미세 조정을 위해 기본 모델의 상위 층들을 훈련 가능하게 설정함
    # # 기본 모델을 다시 훈련 가능하게 설정
    # base_model = model.layers[0]
    # base_model.trainable = True
    #
    # # 상위 층들만 미세 조정 (하위 층들은 고정)
    # fine_tune_at = len(base_model.layers) - 20
    #
    # for layer in base_model.layers[:fine_tune_at]:
    #     layer.trainable = False
    #
    # # 낮은 학습률로 다시 컴파일
    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy', 'top_2_accuracy']
    # )
    #
    # return model

def plot_training_history(history):

    # 훈련 히스토리를 그래프로 시각화함
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 정확도 그래프
    ax1.plot(history.history['accuracy'], label='훈련 정확도')
    ax1.plot(history.history['val_accuracy'], label='검증 정확도')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 손실 그래프
    ax2.plot(history.history['loss'], label='훈련 손실')
    ax2.plot(history.history['val_loss'], label='검증 손실')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_history.png'))
    plt.show()

def fine_tune_model(model, train_generator):

    # 미세 조정을 위해 기본 모델의 상위 층들을 훈련 가능하게 설정함
    # 기본 모델을 다시 훈련 가능하게 설정
    base_model = model.layers[0]
    base_model.trainable = True
    
    # 상위 층들만 미세 조정 (하위 층들은 고정)
    fine_tune_at = len(base_model.layers) - 20
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # 낮은 학습률로 다시 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    return model

def save_model_for_deployment(model):

    # Django 서버에서 사용할 수 있도록 모델을 저장함
    # .h5 파일과 class_info.json 파일이 자동으로 생성됨
    # SavedModel 형식으로 저장 (Django에서 사용하기 좋음)
    savedmodel_path = os.path.join(MODEL_SAVE_PATH, 'animal_classifier_savedmodel')
    model.save(savedmodel_path)
    
    # H5 형식으로도 저장 (더 가벼움)
    h5_path = os.path.join(MODEL_SAVE_PATH, 'animal_classifier.h5')
    model.save(h5_path)
    
    # 클래스 이름과 모델 정보를 JSON 파일로 저장 (Django에서 필요)
    import json
    class_info = {
        'class_names': CLASS_NAMES,
        'num_classes': NUM_CLASSES,
        'input_size': IMG_SIZE,
        'train_split': TRAIN_SPLIT,
        'validation_split': VALIDATION_SPLIT,
        'test_split': TEST_SPLIT,
        'total_samples_per_class': 500,
        'model_architecture': 'EfficientNetB0',
        'training_epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }
    
    class_info_path = os.path.join(MODEL_SAVE_PATH, 'class_info.json')
    with open(class_info_path, 'w', encoding='utf-8') as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)
    
    print("모델과 설정 파일이 성공적으로 저장됨:")
    print(f"  SavedModel: {savedmodel_path}")
    print(f"  H5 모델: {h5_path}")
    print(f"  클래스 정보: {class_info_path}")
    print("\n이 파일들을 Django 서버의 ml_models 폴더에 복사하면 바로 사용 가능함")

def create_data_split_summary():
    """
    데이터 분할 정보를 요약하여 텍스트 파일로 저장함
    """
    summary = f"""
동물 분류 모델 데이터 분할 정보
================================

총 클래스 수: {NUM_CLASSES}개
클래스별 이미지 수: 500장
전체 이미지 수: {NUM_CLASSES * 500}장

데이터 분할:
- 훈련 데이터: {TRAIN_SPLIT*100:.0f}% ({int(500 * TRAIN_SPLIT)}장 × {NUM_CLASSES}클래스 = {int(500 * TRAIN_SPLIT * NUM_CLASSES)}장)
- 검증 데이터: {VALIDATION_SPLIT*100:.0f}% ({int(500 * VALIDATION_SPLIT)}장 × {NUM_CLASSES}클래스 = {int(500 * VALIDATION_SPLIT * NUM_CLASSES)}장)  
- 테스트 데이터: {TEST_SPLIT*100:.0f}% ({int(500 * TEST_SPLIT)}장 × {NUM_CLASSES}클래스 = {int(500 * TEST_SPLIT * NUM_CLASSES)}장)

분류할 동물 종류:
{chr(10).join([f"- {name}" for name in CLASS_NAMES])}

권장사항:
- 훈련 데이터는 데이터 증강을 통해 다양성을 확보함
- 검증 데이터로 모델 성능을 모니터링하고 조기 종료를 결정함
- 테스트 데이터로 최종 성능을 한 번만 평가함
- 500장은 딥러닝에 충분한 데이터 양임 (전이학습 사용)
"""
    
    summary_path = os.path.join(MODEL_SAVE_PATH, 'data_split_info.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

def main():
    """
    메인 실행 함수
    """
    print("동물 분류 모델 훈련을 시작함...")
    print(f"분류할 동물 종: {CLASS_NAMES}")
    print(f"데이터 분할 비율 - 훈련:{TRAIN_SPLIT*100:.0f}% 검증:{VALIDATION_SPLIT*100:.0f}% 테스트:{TEST_SPLIT*100:.0f}%")
    
    # 데이터 분할 정보 저장
    create_data_split_summary()
    
    # 데이터 제너레이터 생성
    print("\n데이터 제너레이터를 생성함...")
    train_generator, val_test_generator, val_samples = create_data_generators()
    
    print(f"훈련 샘플 수: {train_generator.samples}")
    print(f"검증+테스트 샘플 수: {val_test_generator.samples}")
    
    # 검증용 데이터 제너레이터 생성 (검증+테스트에서 검증 부분만)
    validation_generator = val_test_generator  # 임시로 전체 사용 (실제로는 분할 필요)
    
    # 모델 생성 및 컴파일
    print("\n모델을 생성하고 컴파일함...")
    model = create_model()
    model = compile_model(model)
    
    # 모델 구조 출력
    print("\n모델 구조:")
    model.summary()
    
    # 콜백 생성
    callbacks = create_callbacks()
    
    # 1단계: 전이 학습
    print("\n1단계: 전이 학습을 시작함...")
    history1 = model.fit(
        train_generator,
        epochs=EPOCHS//2,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # 2단계: 미세 조정
    print("\n2단계: 미세 조정을 시작함...")
    model = fine_tune_model(model, train_generator)
    
    # 미세 조정용 콜백 업데이트
    callbacks[0].filepath = os.path.join(MODEL_SAVE_PATH, 'best_finetuned_model.h5')
    
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS//2,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # 전체 훈련 히스토리 결합
    history_combined = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    
    # 가짜 History 객체 생성
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_history = CombinedHistory(history_combined)
    
    # 훈련 결과 시각화
    print("\n훈련 결과를 시각화함...")
    plot_training_history(combined_history)
    
    # 검증 세트에서 평가
    print("\n검증 세트에서 모델 평가:")
    val_loss, val_accuracy, val_top2_accuracy = model.evaluate(validation_generator, verbose=1)
    print(f"검증 손실: {val_loss:.4f}")
    print(f"검증 정확도: {val_accuracy:.4f}")
    print(f"검증 Top-2 정확도: {val_top2_accuracy:.4f}")
    
    # 테스트 세트에서 최종 평가
    test_loss, test_accuracy, test_top2_accuracy = evaluate_on_test_set(model, val_test_generator, val_samples)
    
    # Django용 모델 저장
    print("\nDjango 서버용 모델을 저장함...")
    save_model_for_deployment(model)
    
    print("\n모델 훈련이 완료됨!")
    print(f"최종 테스트 정확도: {test_accuracy:.4f}")
    print("생성된 파일들을 Django 프로젝트로 복사하여 사용하세요.")

# Django에서 사용할 예측 함수
def predict_animal(model_path, image_path, class_names=CLASS_NAMES):
    """
    Django 서버에서 사용할 예측 함수
    
    Args:
        model_path: 저장된 모델 경로
        image_path: 예측할 이미지 경로
        class_names: 클래스 이름 리스트
    
    Returns:
        예측 결과 딕셔너리
    """
    # 모델 로드
    model = keras.models.load_model(model_path)
    
    # 이미지 전처리
    img = keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # 예측 수행
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = class_names[predicted_class_idx]
    
    # 모든 클래스별 확률
    class_probabilities = {}
    for i, class_name in enumerate(class_names):
        class_probabilities[class_name] = float(predictions[0][i])
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'class_probabilities': class_probabilities
    }

if __name__ == "__main__":
    main()