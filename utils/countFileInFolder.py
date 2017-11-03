import os, os.path

path, dirs, files = os.walk("/home/default/.keras/datasets/weedspic/lettuce/image_aug").next()
file_count = len(files)
print(file_count)
