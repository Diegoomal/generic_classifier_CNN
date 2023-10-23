import json
import numpy as np
from PIL import Image
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, MaxPooling2D)

from configs import WIDTH, HEIGHT, CHANNELS, SIZE
from data_augmentation import execute_data_augmentation_operation_on_image_by_path
from util import (read_image, reset_seeds,
                  get_all_files_in_path_filtered_by_extesion,
                  get_object_from_metadata)


class GenericClassifierCNN:


  def __init__(self, path_dataset='../dataset'):
    self.path_dataset = path_dataset


  def verify_gpu(self):
    print('1) Verify physical GPU')
    print(tf.config.list_physical_devices('GPU'))


  def test_dataset(self):
    img = Image.open(self.path_dataset + '/0/0.jpg')
    print(f'''ebeer_dataset/0/0.jpg --> 
          width: {img.size[0]} x height {img.size[1]} (pixels)''')


  def create_metadata_file(self):

    paths_metadata = get_all_files_in_path_filtered_by_extesion(
      self.path_dataset, ['json'])

    json_content_arr = []
    for path_metadata in paths_metadata:
      with open(path_metadata, 'r') as file:
        file_content = json.load(file)
        json_content_arr.append(file_content)

    json_content_arr = sorted(list(json_content_arr), key=lambda x: x['path'])

    with open('general_metadata.json', 'w') as file_writer:
      json.dump(json_content_arr, file_writer)


  def run_data_augmentation(self):

    paths_dataset_images = get_all_files_in_path_filtered_by_extesion(self.path_dataset, [".jpg", ".png"])

    with tf.device('/gpu:0'):
      for path_file in paths_dataset_images:
        execute_data_augmentation_operation_on_image_by_path(path_file)

    print("execute_data_augmentation -> len(imgs):", len(paths_dataset_images))


  def create_train_data(self):

    json_obj = get_object_from_metadata()

    n_neurons_out = len(json_obj)

    list_train_data_X = []
    list_train_data_y = []

    for i in range(n_neurons_out):

      paths_dataset_images = get_all_files_in_path_filtered_by_extesion(
        f"{self.path_dataset}/{i}", [".jpg", ".png"])

      train_data = [
        np.array(image).astype('float32')/255 for image in
        [
          read_image('', '', image_name).resize(SIZE)
          for image_name in paths_dataset_images
        ]
      ]

      list_train_data_X.append(train_data)
      list_train_data_y.append(np.ones(len(train_data)) * i)
    
    return n_neurons_out, list_train_data_X, list_train_data_y


  def create_model(self, n_neurons_out):
    
    reset_seeds()

    model = Sequential()

    # Extração de caracteristicas
    model.add(Conv2D(256, (3, 3), activation='relu',
                     input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Achatamento
    model.add(Flatten())

    # classificadores
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_neurons_out, activation='softmax'))

    learning_rate = 1e-4

    model.compile(
      optimizer=Adam(learning_rate=learning_rate),
      loss='binary_crossentropy',
      metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
      ]
    )

    return model


  def train_model(self, model, list_train_data_X, list_train_data_y):
    
    X_train, X_test, y_train, y_test = train_test_split(
      np.concatenate(list_train_data_X),
      to_categorical(np.concatenate(list_train_data_y)),
      test_size=0.3,
      random_state=42
    )
    
    reset_seeds()

    with tf.device('/gpu:0'):
      hist = model.fit(X_train, y_train, epochs=25, validation_split=0.2)

    model.evaluate(X_test, y_test)

    return hist


  def run_train(self):

    self.create_metadata_file()
    self.run_data_augmentation()
    n_neurons_out, list_train_data_X, list_train_data_y = self.create_train_data()
    model = self.create_model(n_neurons_out)
    hist = self.train_model(model, list_train_data_X, list_train_data_y)
    model.save('trained_model.h5')
    return hist, model


  def predict(self, model, path_test, filename):

    image_original = tf.keras.utils.load_img(
      self.path_dataset + path_test + filename,
      grayscale=False,
      color_mode='rgb',
      target_size=None,
      interpolation='nearest'
    )

    image_prepared = np.expand_dims(image_original.resize(SIZE), axis=0)

    predicted = model.predict(image_prepared)

    n_pos = predicted.argmax(axis=-1)[0]

    json_obj = get_object_from_metadata()

    dict_labels_index = {i: json_obj[i] for i in range(len(json_obj))}

    print("--->:", dict_labels_index[n_pos]['name'])
