import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
image_size = 256
batch_size = 32

idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=20, # You can uncomment these parameters to make you generator rotate & flip the images to put the train model in stricter conditions.
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1
)
train_gen = idg.flow_from_directory('./input/group_project/final_aug/',
                                    target_size=(image_size, image_size),
                                    subset='training',
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    shuffle=True,
                                    seed=1
                                    )
val_gen = idg.flow_from_directory('./input/group_project/final_aug/',
                                  target_size=(image_size, image_size),
                                  subset='validation',
                                  class_mode='categorical',
                                  batch_size=batch_size,
                                  shuffle=True,
                                  seed=1
                                  )
unique, counts = np.unique(train_gen.classes, return_counts=True)
dict1 = dict(zip(train_gen.class_indices, counts))

keys = dict1.keys()
values = dict1.values()

plt.xticks(rotation='vertical')
bar = plt.bar(keys, values)
# plt.show()

x, y = next(train_gen)


def show_grid(image_list, nrows, ncols, label_list=None, show_labels=False, figsize=(10, 10)):
    fig = plt.figure(None, figsize, frameon=False)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.2,
                     share_all=True,
                     )
    for i in range(nrows * ncols):
        ax = grid[i]
        ax.imshow(image_list[i], cmap='Greys_r')
        ax.axis('off')
    # plt.show()


show_grid(x, 2, 4, show_labels=True, figsize=(10, 10))

# plt.show()

# model building
# 我们使用的模型是来自 Keras 的 Sequential。
#
# 然后输入层从
#
# 输入层
#
# Conv2D（64、128 个过滤器）
#
# MaxPool2D
#
# GlobalMaxPool2D
#
# 批量归一化
#
# 展平
#
# 退出
#
# 稠密
#
# 我们将输入大小设置为 256 x 256 x 3（大小 x 颜色）。
#
# 在 MaxPool2D 层中，我们将池大小设置为 2 x 2（大小 x 颜色）。
#
# 在 Conv2D 层中，我们将内核大小设置为 3 x 3。

model = tf.keras.models.Sequential()

# Input layer
# Can be omitted, you can specify the input_shape in other layers
model.add(tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3,)))

# Here we add a 2D Convolution layer
# Check https://keras.io/api/layers/convolution_layers/convolution2d/ for more info
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Max Pool layer
# It downsmaples the input representetion within the pool_size size
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# Normalization layer
# The layer normalizes its output using the mean and standard deviation of the current batch of inputs.
model.add(tf.keras.layers.BatchNormalization())

# 2D Convolution layer
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

# Max Pool layer
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# Normalization layer
model.add(tf.keras.layers.BatchNormalization())

# 2D Convolution layer
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

# Max Pool layer
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# Normalization layer
model.add(tf.keras.layers.BatchNormalization())

# 2D Convolution layer
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

# Max Pool layer
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# Global Max Pool layer
model.add(tf.keras.layers.GlobalMaxPool2D())

# Dense Layers after flattening the data
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))

# Dropout
# is used to nullify the outputs that are very close to zero and thus can cause overfitting.
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))

# Normalization layer
model.add(tf.keras.layers.BatchNormalization())

# Add Output Layer ####################################################################################################################
model.add(tf.keras.layers.Dense(34, activation='softmax'))  # = 12 predicted classes
# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# You can save the best model to the checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint('plant_classifier.h5',  # where to save the model
                                                save_best_only=True,
                                                monitor='val_accuracy',
                                                mode='max',
                                                verbose=1)
# 接下来，我们将拟合模型。我们将 epochs 设置为 20，以便训练模型不会花费太长时间。
#
# 将每 1 个 epoch 的步长设置为 3803，这是我们找到的训练图像的数量。
#
# 将验证步骤设置为 947，等于我们找到的验证图像的数量。
#
# 然后回调回检查点。
############################################################################################################################################
history = model.fit(train_gen,
                    epochs=20,  # Increase number of epochs if you have sufficient hardware
                    steps_per_epoch=310133 // batch_size,  # Number of train images // batch_size
                    validation_data=val_gen,
                    validation_steps=34446 // batch_size,  # Number of val images // batch_size
                    callbacks=[checkpoint],
                    verbose=1
                    )
# 接下来，让我们看一下 Learning curves 和 Epoch graph 之间的 accuracy 图，
# 其中 x 轴是 Epoch，y 轴是 Accuracy。
print(history.history)
plt.subplot()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(list(range(1, 21)))
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.show()

# Visualizing
# 使用数据集中的图像检查模型的预测结果
maize = cv2.imread('./input/group_project/final_aug/Achillea maritima/001.jpg')
plt.imshow(maize)
plt.show()
# 预处理图像：resize + expand_dims。
maize = cv2.resize(maize, (256, 256))
maize_batch = np.expand_dims(maize, axis=0)  # 将二维数组的轴切换到三维
conv_maize = model.predict(maize_batch)
conv_maize.shape


def visualize(maize_batch):
    maize = np.squeeze(maize_batch, axis=0)
    print(maize.shape)
    plt.imshow(maize)


plt.imshow(conv_maize)
plt.show()

# 图像预测概率的可视化示例 图像预测概率的可视化示例
# #
# # 创建一个简单的模型以查看神经网络的工作原理。

simple_model = tf.keras.models.Sequential()
simple_model.add(tf.keras.layers.Conv2D(1, 3, 3, input_shape=maize.shape))  # 3x3 kernel


# Function to show the mask of the image (aka how the model sees the image)
def visualize_color(simple_model, maize):
    maize_batch = np.expand_dims(maize, axis=0)
    conv_maize2 = simple_model.predict(maize_batch)
    conv_maize2 = np.squeeze(conv_maize2, axis=0)

    print(conv_maize2.shape)
    conv_maize2 = conv_maize2.reshape(conv_maize2.shape[:2])
    print(conv_maize2.shape)
    plt.imshow(conv_maize2)


visualize_color(simple_model, maize)

plt.show()
