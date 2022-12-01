import os
import random

img_amt = 0
dict = {}
for catalog in os.listdir('./aug_output_dir'):
    img_amt += len(os.listdir(os.path.join('./aug_output_dir', catalog)))
    dict[catalog] = len(os.listdir(os.path.join('./aug_output_dir', catalog)))

    catalog_path = os.path.join('./aug_output_dir', catalog)
    file_list = os.listdir(catalog_path)
    num_file = len(file_list)
    if num_file < 3100:
        catalog_path = os.path.join('./aug_output_dir', catalog)
        continue
    del_file_num = num_file - 3100
    for i in range(del_file_num):
        del_file_idx = random.randint(0, num_file - 1)
        os.remove(os.path.join(catalog_path, file_list[del_file_idx]))
        file_list = os.listdir(catalog_path)
        num_file -= 1

    # dict[catalog] = len(os.listdir(os.path.join('./aug_output_dir', catalog)))

print(dict)
print(img_amt)
