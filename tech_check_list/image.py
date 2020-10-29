# TensorFlow 2.x를 활용하여 심층신경망 및 합성곱 신경망(CNN)으로 이미지 인식 및 객체 탐지 모델을
# 빌드하는 방법을 이해해야 합니다. 다음을 할 줄 알아야 합니다.

# Conv2D 및 풀링 레이어로 합성곱 신경망(CNN) 정의.
import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax') # 26 alphabets/hand-signs so 26 classes!
])
# 실제 이미지 데이터세트를 처리하기 위한 모델 빌드 및 훈련.
# 합성곱 활용 방법을 이해하여 신경망 개선.
# 다양한 형태와 크기의 실제 이미지 활용.
# 이미지 확장을 이용하여 과적합 예방.
# ImageDataGenerator 활용.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_images = np.expand_dims(training_images, axis=-1)
testing_images = np.expand_dims(testing_images, axis=-1)
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale = 1./255)
history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=32),
                              steps_per_epoch=len(training_images) / 32,
                              epochs=15,
                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
                              validation_steps=len(testing_images) / 32)
    
# 디렉터리 구조를 기반으로 ImageDataGenerator에서 이미지의 라벨을 지정하는 방법 이해.
train_datagen = ImageDataGenerator(
rescale=1/255,
rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150,150)
)     
validation_generator =  test_datagen.flow_from_directory( 
    validation_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150,150)
)