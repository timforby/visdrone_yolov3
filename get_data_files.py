import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", "-n", type=str, help="name of dataset")
parser.add_argument("--dataset_type", "-t", type=str, help="type of dataset train/target/ground_truth/test/val")
parser.add_argument("--image_folder_path", "-p", type=str, help="path to image folder")
parser.add_argument("--add_dimensions", "-d", action='store_true', help="whether or not to add dimensions")
opt = parser.parse_args()
print(opt)


print(opt.image_folder_path)

for (dir_path, dir_names, file_names) in os.walk(opt.image_folder_path):
    break
image_names = ['/'.join([dir_path,fn]) for fn in file_names]
for d in dir_names:
    for (d_p, d_n, f_n) in os.walk(dir_path+"/"+d):
        break
    image_names += ['/'.join([d_p,fn]) for fn in f_n]

os.makedirs("../visdrone_synthetic/data/"+opt.dataset_name, exist_ok=True)
f = open("../visdrone_synthetic/data/"+opt.dataset_name+"/"+opt.dataset_type+".txt", "w")
for path in image_names:
    if '.png' in path or '.jpg' in path:
        if opt.add_dimensions:
            img = cv2.imread(path)
            f.write(path+','+str(img.shape[0])+','+str(img.shape[1])+','+str(img.shape[2])+'\n')
        else:
            f.write(path+'\n')
f.close()