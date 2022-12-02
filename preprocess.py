import os
import argparse
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', help='source image directory', default='./img_dir', type=str)
    parser.add_argument('--output', help='where to save the processed images', default='./rename_dir', type=str)
    parser.add_argument('--img-size', help='output size of the image (width, height)', default=(150, 150), type=tuple)
    parser.add_argument('--split-ratio', help='how many images should be assigned to train and validation set',
                        default=0.9,
                        type=int)
    parser.add_argument('--kind', help='how many breeds of cats in the target image files', default=7, type=int)
    args = parser.parse_args()

    # check output directory exists or not
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    kind_list = [str(x) for x in range(1, args.kind + 1)]
    kind = 1

    src_dir = args.src_dir
    if os.path.exists(os.path.join(src_dir, '.DS_Store')):
        os.remove(os.path.join(src_dir, '.DS_Store'))

    catalog_list = os.listdir(src_dir)
    # get amount of images to split train and validation set
    img_amt = 0
    for i in range(len(catalog_list)):
        img_amt += len(os.listdir(os.path.join(src_dir, catalog_list[i])))

    for catalog in catalog_list:
        catalog_path = os.path.join(src_dir, catalog + '/')
        if os.path.exists(os.path.join(catalog_path, '.DS_Store')):
            os.remove(os.path.join(catalog_path, '.DS_Store'))

        images = os.listdir(catalog_path)
        for image in images:
            if image.split('_')[0] in kind_list:
                continue
            else:
                # rename image name to kind_origin.jpg eg. 1_naver_0435.jpg
                os.rename(catalog_path + image, catalog_path + str(kind) + '_' + image.split('.')[0] + '.jpg')

        kind += 1

        for image in os.listdir(catalog_path):
            out_catalog = os.path.join(out_dir, catalog)
            if not os.path.exists(out_catalog):
                os.mkdir(out_catalog)
            img = Image.open(os.path.join(catalog_path, image))
            try:
                new_img = img.resize((args.img_size[0], args.img_size[1]), Image.BILINEAR)
                new_img.save(os.path.join(out_catalog + '/', os.path.basename(image)))
            except OSError as oserr:
                print(oserr)
                rgb_img = img.convert('RGB').resize((args.img_size[0], args.img_size[1]), Image.BILINEAR)
                rgb_img.save(os.path.join(out_catalog + '/', os.path.basename(image)))
            except Exception as e:
                print(e)

        # resize_and_split(args, catalog_path, img_amt, out_dir)


if __name__ == '__main__':
    main()
