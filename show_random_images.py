import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", required=False,
                help="path to train path of dataset (i.e., directory of images)")

args = vars(ap.parse_args())

def class_to_index_mapping(path):
    ix_to_class = {}
    classes =[]
    with open(path) as txt:
        classes = [l.strip() for l in txt.readlines()]
        ix_to_class = dict(zip(range(len(classes)), classes))
        return ix_to_class


data_dir = "D:\Metwalli\master\\reasearches proposals\Computer vision\Materials\VIREO\\ready_chinese_food" # args["data_dir"]
food_list_path = os.path.join("D:\Metwalli\master\\reasearches proposals\Computer vision\Materials\VIREO\\SplitAndIngreLabel", "FoodList.txt")
rows = 29
cols = 6
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(15, 64))
fig.suptitle('Random Image from Each Food Class', fontsize=20)
sorted_food_dirs = os.listdir(data_dir)
food_list = class_to_index_mapping(food_list_path)
for i in range(rows):
    for j in range(cols):
        try:
            food_dir = sorted_food_dirs[i*cols + j]
        except:
            break
        all_files = os.listdir(os.path.join(data_dir, food_dir))
        rand_img = np.random.choice(all_files)
        img = plt.imread(os.path.join(data_dir, food_dir, rand_img))
        ax[i][j].imshow(img)
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        ax[i][j].text(0, -20, food_list[int(food_dir)-1], size=10, rotation=0,
                ha="left", va="top",
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
