import argparse

import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, InputLayer
from keras.layers import Dense, Dropout, Flatten
from matplotlib import pyplot as plt
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', help='Where to look for the training image dataset', type=str,
                        default='./aug_output_dir')
    parser.add_argument('--batch-size', help='How many images in training image dataset', type=int, default=128)
    parser.add_argument('--epochs', help='How many training epochs to run', type=int, default=10)
    parser.add_argument('--input-shape', help='The size of the datasets', type=int, default=100)
    parser.add_argument('--number-class', help='The number of the class', type=int, default=7)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    epochs = args.epochs
    input_shape = args.input_shape
    number_class = args.number_class

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.1,
    )

    train_data = generator.flow_from_directory(dataset_dir,
                                               target_size=(input_shape, input_shape),
                                               subset='training',
                                               class_mode='categorical',
                                               batch_size=batch_size,
                                               shuffle=True,
                                               seed=1)
    validate_data = generator.flow_from_directory(dataset_dir,
                                                  target_size=(input_shape, input_shape),
                                                  subset='validation',
                                                  class_mode='categorical',
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  seed=1)

    # Train
    model = keras.models.Sequential()
    model.add(InputLayer(input_shape=(input_shape, input_shape, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(number_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../cat_classifier.h5',
                                                    save_best_only=True,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    verbose=1)

    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=validate_data,
                        callbacks=[checkpoint],
                        verbose=1
                        )

    # plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
