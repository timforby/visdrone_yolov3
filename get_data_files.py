import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", "-n", type=str, help="name of dataset")
parser.add_argument("--dataset_type", type=str, default="train", help="type of dataset train/test/val")
parser.add_argument("--image_folder_path", type=str, default="../rs_visdrone/visdrone_data", help="path to image folder")
opt = parser.parse_args()
print(opt)

print(opt.image_folder_path)
image_names = []
for (dir_path, dir_names, file_names) in os.walk(opt.image_folder_path):
    break

os.makedirs("data/"+opt.dataset_name, exist_ok=True)
f = open("data/"+opt.dataset_name+"/"+opt.dataset_type+".txt", "w")
for fn in file_names:
    f.write('/'.join([opt.image_folder_path,fn])+'\n')
f.close()