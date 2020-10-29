# TensorFlow 2.x를 활용하여 머신러닝(ML) 모델 빌드, 컴파일 및 훈련.

# 모델에서 활용할 수 있도록 데이터 전처리.
import shutil
import zipfile, os
from os import getcwd

path_cats_and_dogs = f"{getcwd()}/../tmp2/cats-and-dogs.zip"
shutil.rmtree('/tmp')

local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp') #압축 해제
zip_ref.close()

try:
    os.mkdir('/tmp/cats-v-dogs')
    os.mkdir('/tmp/cats-v-dogs/training')
    os.mkdir('/tmp/cats-v-dogs/testing')
    os.mkdir('/tmp/cats-v-dogs/training/cats')
    os.mkdir('/tmp/cats-v-dogs/training/dogs')
    os.mkdir('/tmp/cats-v-dogs/testing/cats')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dataset = []
    
    for unit_data in os.listdir(SOURCE):
        data = SOURCE + unit_data
        if (os.path.getsize(data) > 0):
            dataset.append(unit_data)
        else:
            print('Skipped ' + unit_data)
            print('Invalid file size! i.e Zero length.')
    
    import random
    train_data_length = int(len(dataset) * SPLIT_SIZE)
    test_data_length = int(len(dataset) - train_data_length)
    shuffled_set = random.sample(dataset, len(dataset))
    train_set = shuffled_set[0:train_data_length]
    test_set = shuffled_set[-test_data_length:]
    
    from shutil import copyfile

    for unit_data in train_set:
        temp_train_data = SOURCE + unit_data
        final_train_data = TRAINING + unit_data
        copyfile(temp_train_data, final_train_data)
    
    for unit_data in test_set:
        temp_test_data = SOURCE + unit_data
        final_test_data = TESTING + unit_data
        copyfile(temp_train_data, final_test_data)


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# 모델을 활용하여 결과 예측.
# 다양한 레이어로 순차적인 모델 빌드.
import tensorflow as tf
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])
# 바이너리 분류에 대한 모델 빌드 및 훈련.
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

# 멀티클래스 분류에 대한 모델 빌드 및 훈련.
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(127, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(training_images, training_labels, epochs=20)

# 훈련된 모델의 플롯 손실 및 정확도.
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')

# 확장 및 드롭아웃을 비롯한 전략 파악하여 과적합 예방.
# 훈련된 모델 활용(전이 학습).
path_inception = f"{getcwd()}/../tmp2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
from tensorflow.keras.applications.inception_v3 import InceptionV3
local_weights_file = path_inception
pre_trained_model =InceptionV3(
    input_shape=(150,150,3),
    include_top=False,
    weights=None
)
pre_trained_model.load_weights(local_weights_file)
last_layer = pre_trained_model.get_layer("mixed7")
last_output = last_layer.output
from tensorflow.keras import layers, Model
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(1, activation="sigmoid")(x)           
model = Model(pre_trained_model.input, x) 
model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = "binary_crossentropy",
              metrics = ["acc"])
# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable=False
# 사전 훈련된 모델에서 특성 추출.
pre_trained_model.summary()
# 모델에 대한 입력값이 올바른 형태인지 확인.
# 테스트 데이터와 신경망의 입력 형태를 일치시킬 수 있어야 함.
# 신경망의 출력 데이터를 테스트 데이터의 지정된 입력 형태와 일치시킬 수 있어야 함.
# 대용량 데이터 로드 이해.
# 콜백을 사용하여 훈련 주기 마지막을 트리거.
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, callback, logs={}):
        if (logs.get('acc') > 0.998):
            print("\nReached {}/% accuracy, stopping training".format(0.998*100))
            self.model.stop_training = True
# 여러 다른 소스의 데이터세트 활용.
# JSON 및 CSV을 포함한 다양한 형식의 데이터세트 활용.
import numpy as np
def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        first_line = True
        temp_labels= []
        temp_images = []

        for row in csv_reader:
            if first_line: 
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_array = np.array_split(image_data, 28) # Make 28 x 28
                temp_images.append(image_array)

        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
    return images, labels
path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)

# tf.data.datasets의 데이터세트 활용.
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
training_images=training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images / 255.0