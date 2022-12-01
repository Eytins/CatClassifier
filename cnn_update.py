import argparse
import os

import cv2
import numpy as np
from keras import regularizers
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', help='Where to look for the training image dataset', type=str,
                        default='./output_dir/train_set/')
    parser.add_argument('--validate-dataset', help='Where to look for the validation image dataset', type=str,
                        default='./output_dir/validation_set/')
    parser.add_argument('--batch-size', help='How many images in training image dataset', type=int, default=128)
    parser.add_argument('--epochs', help='How many training epochs to run', type=int, default=5)
    parser.add_argument('--input-shape', help='The size of the datasets', type=tuple, default=(235, 235, 3))
    parser.add_argument('--number-class', help='The number of the class', type=int, default=18)
    parser.add_argument('--model-depth', help='The depth of the model', type=int, default=5)
    parser.add_argument('--model-size', help='The size of the model', type=int, default=2)
    parser.add_argument('--width', help='Width of captcha image', type=int, default=235)
    parser.add_argument('--height', help='Height of captcha image', type=int, default=235)
    args = parser.parse_args()

    train_dataset = args.train_dataset
    validate_dataset = args.validate_dataset
    batch_size = args.batch_size
    epochs = args.epochs
    input_shape = args.input_shape
    number_class = args.number_class
    model_depth = args.model_depth
    model_size = args.model_size
    height = args.height
    width = args.width

    train_list = os.listdir(train_dataset)

    x_train_list = []
    y_train_list = []
    for image in train_list:
        try:
            image_data = cv2.imread(os.path.join(train_dataset, image))
            rgb_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            x_train_list.append(rgb_data)
            y_train_list.append(image.split('_')[0])
            print()
        except Exception as e:
            print(e)
    x_train = np.array(x_train_list).astype("float32") / 255
    y_train = np.array(y_train_list)

    validation_list = os.listdir(validate_dataset)

    x_test_list = []
    y_test_list = []
    for image in validation_list:
        try:
            image_data = cv2.imread(os.path.join(validate_dataset, image))
            rgb_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            x_test_list.append(rgb_data)
            y_test_list.append(image.split('_')[0])
            print()
        except Exception as e:
            print(e)
    x_test = np.array(x_test_list).astype("float32") / 255
    y_test = np.array(y_test_list)

    y_train = keras.utils.to_categorical(y_train, number_class)
    y_test = keras.utils.to_categorical(y_test, number_class)

    use_saved_model = False
    model = ''
    if use_saved_model:
        print()
    else:
        model = keras.Sequential()
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(number_class, activation='softmax', kernel_regularizer=regularizers.l1(1)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.summary()
        start_time = time.time()
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        end_time = time.time()
        print(end_time - start_time, "s")
        model.save("cifar.model")

        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    preds = model.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1, y_pred))

    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1, y_pred))


if __name__ == '__main__':
    main()
