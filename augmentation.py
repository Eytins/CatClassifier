from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array, load_img
import os
import argparse
from PIL import Image

# Code reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

data_generator = ImageDataGenerator(
    rotation_range=90,  # Int. Degree range for random rotations.
    fill_mode='nearest',  # used with rotation, fill the points outside the boundaries of the input image
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=[0.8, 1.2],
    # width_shift_range=[-0.1, 0.1],
    # height_shift_range=[-0.1, 0.1],
    shear_range=0.2
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', help='source image directory', default='./rename_dir', type=str)
    parser.add_argument('--out-dir', help='where to save the augmented images', default='./aug_output_dir', type=str)
    parser.add_argument('--augment-num', help='number of images for one picture augmentation', default=10, type=int)
    args = parser.parse_args()

    src_dir = args.src_dir
    out_dir = args.out_dir

    # create output directory if not exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    catalog_list = os.listdir(src_dir)
    for catalog in catalog_list:
        # if the output directory not exist, create one
        if not os.path.exists(os.path.join(out_dir, catalog)):
            os.mkdir(os.path.join(out_dir, catalog))

    for catalog in catalog_list:
        catalog_path = os.path.join(src_dir, catalog + '/')
        if os.path.exists(os.path.join(catalog_path, '.DS_Store')):
            os.remove(os.path.join(catalog_path, '.DS_Store'))

        for img in os.listdir(catalog_path):
            img_path = catalog_path + img
            out_img_path = os.path.join(out_dir, catalog)

            original = Image.open(img_path)
            try:
                # first save the original image in the output path
                original.save(out_dir + '/' + catalog + '/' + img)
            except OSError as oserr:
                print(oserr)
                if len(oserr.args) >= 1 and oserr.args[0] == 'cannot write mode RGBA as JPEG':
                    # if it's rgba, convert to rgb first then save
                    original = original.convert('RGB')
                    original.save(out_dir + '/' + catalog + '/' + img)
            # augment this image and save them in the same path
            aug_img = load_img(img_path)
            img_arr = img_to_array(aug_img)
            img_arr = img_arr.reshape((1,) + img_arr.shape)

            num = 0
            for _ in data_generator.flow(img_arr, batch_size=1, save_to_dir=out_img_path,
                                         save_prefix=os.path.basename(img_path).split('.')[0], save_format='jpg'):
                num += 1
                # if the number of augmented images exceeds args.augment_num, break
                if num >= args.augment_num:
                    break


if __name__ == '__main__':
    main()
