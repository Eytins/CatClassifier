import os, argparse
import random
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', help='source image directory', default='./aug_output_dir', type=str)
    parser.add_argument('--output', help='where to save the processed images', default='./output_dir', type=str)
    parser.add_argument('--split-ratio', help='how many images should be assigned to train and validation set',
                        default=0.9,
                        type=int)
    args = parser.parse_args()

    src_dir = args.src_dir
    out_dir = args.output

    # if os.path.exists(os.path.join(src_dir, '.DS_Store')):
    #     os.remove(os.path.join(src_dir, '.DS_Store'))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'train_set')):
        os.mkdir(os.path.join(out_dir, 'train_set'))
    if not os.path.exists(os.path.join(out_dir, 'validation_set')):
        os.mkdir(os.path.join(out_dir, 'validation_set'))

    img_amt = 0
    for catalog in os.listdir(src_dir):
        if os.path.exists(os.path.join(src_dir, catalog + '/' + '.DS_Store')):
            os.remove(os.path.join(src_dir, catalog + '/' + '.DS_Store'))
        img_amt += len(os.listdir(os.path.join(src_dir, catalog)))

    for catalog in os.listdir(src_dir):
        catalog_path = os.path.join(src_dir, catalog + '/')
        for image in os.listdir(catalog_path):
            img = Image.open(os.path.join(catalog_path, image))
            if random.randint(0, img_amt) >= img_amt * args.split_ratio:
                img.save(os.path.join(os.path.join(out_dir, 'validation_set'), os.path.basename(image)))
            else:
                img.save(os.path.join(os.path.join(out_dir, 'train_set'), os.path.basename(image)))


if __name__ == '__main__':
    main()
