import glob
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import os
import multiprocessing as mp

# 8 cored are used for multi-thread
NUM_JOBS = 8
# output size
IMG_SIZE = 256
INPUT_DATA_DIR = '/path_to_imgs/'
OUTPUT_DATA_DIR = '/output_folder/'
LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600

def resize_img(img):
    nan_mask = np.isnan(img) # Remove NaN
    img[nan_mask] = LOW_THRESHOLD
    img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1,1])

    valid_plane_i = np.mean(img, (1,2)) != -1 # Remove blank axial planes
    img = img[valid_plane_i,:,:]

    img = resize(img, (IMG_SIZE, IMG_SIZE, IMG_SIZE), mode='constant', cval=-1)
    return img

def main():
    img_list = list(glob.glob(INPUT_DATA_DIR+"*.nii.gz"))

    processes = []
    for i in range(NUM_JOBS):
        processes.append(mp.Process(target=batch_resize, args=(i, img_list)))
    for p in processes:
        p.start()

def batch_resize(batch_idx, img_list):
    for idx in range(len(img_list)):
        if idx % NUM_JOBS != batch_idx:
            continue
        imgname = img_list[idx].split('/')[-1]
        if os.path.exists(OUTPUT_DATA_DIR+imgname.split('.')[0]+".npy"):
            # skip already finished pre-processing
            continue
        try:
            img = sitk.ReadImage(INPUT_DATA_DIR + img_list[idx])
        except Exception as e: 
            # skip corrupted images
            print(e)
            print("Image loading error:", imgname)
            continue 
        img = sitk.GetArrayFromImage(img)
        try:
            img = resize_img(img)
        except Exception as e: # Some images are corrupted
            print(e)
            print("Image resize error:", imgname)
            continue 
        np.save(OUTPUT_DATA_DIR+imgname.split('.')[0]+".npy", img)

if __name__ == '__main__':
    main()
