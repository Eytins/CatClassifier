import os
import random

img_amt = 0
dict = {}
for catalog in os.listdir('./aug_output_dir'):
    dict[catalog] = len(os.listdir(os.path.join('./aug_output_dir', catalog)))
    img_amt += len(os.listdir(os.path.join('./aug_output_dir', catalog)))
    # catalog_path = os.path.join('./aug_output_dir', catalog)
    # file_list = os.listdir(catalog_path)
    # num_file = len(file_list)
    # if num_file < 3100:
    #     catalog_path = os.path.join('./aug_output_dir', catalog)
    #     continue
    # del_file_num = num_file - 3100
    # del_list = []
    # for i in range(del_file_num):
    #     del_file_idx = random.randint(0, num_file)
    #     if del_file_idx in del_list:
    #         for j in range(del_file_idx + 1, num_file):
    #             if j not in del_list:
    #                 del_file_idx = j
    #                 break
    #     del_list.append(del_file_idx)
    #     os.remove(os.path.join(catalog_path, file_list[del_file_idx]))
    #     num_file -= 1

print(dict)
print(img_amt)
