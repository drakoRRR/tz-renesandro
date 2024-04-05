import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping


current_file_path = os.path.abspath(__file__)

project_root = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(os.path.dirname(project_root) + '\\' + 'application')

training_path = os.path.join(project_root, "images_for_model", "train")
validation_path = os.path.join(project_root, "images_for_model", "validation")
test_path = os.path.join(project_root, "images_for_model", "testing")

# print(cv2.imread(image_path).shape)
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(
    training_path,
    target_size=(700, 700),
    batch_size=3,
    class_mode='binary'
)

validation_dataset = train.flow_from_directory(
    validation_path,
    target_size=(700, 700),
    batch_size=3,
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(700, 700, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    #
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(700, 700, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    #
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(700, 700, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    #
    tf.keras.layers.Flatten(),
    ##
    tf.keras.layers.Dense(512, activation='relu'),
    ##
    tf.keras.layers.Dense(1, activation='sigmoid')
])

early_stopping_callback = EarlyStopping(
    monitor='val_acc',
    min_delta=0.001,
    patience=10,
    verbose=1,
    restore_best_weights=True,
    mode='max'
)

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
model_fit = model.fit(
    train_dataset,
    steps_per_epoch=3,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=[early_stopping_callback]
)


# For test
# for i in os.listdir(test_path):
#     img = image.load_img(test_path + '//' + i, target_size=(700, 700))
#     print(test_path + '//' + i)
#     plt.imshow(img)
#     plt.show()
#
#     X = image.img_to_array(img)
#     X = np.expand_dims(X, axis=0)
#     images = np.vstack([X])
#
#     val = model.predict(images)
#     if val == 0:
#         print('bad picture')
#     else:
#         print('good picture')
