import SimpleITK as sitk
import pandas as pd
import numpy as np
from skimage.transform import resize
import os
import multiprocessing as mp

NUM_JOBS = 16
IMG_SIZE = 128
DATA_DIR = '/pghbio/dbmi/batmanlab/Data/COPDGene/'
LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600 # -250

def resize_img(img):
    nan_mask = np.isnan(img) # Remove NaN
    img[nan_mask] = LOW_THRESHOLD
    img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1,1])

    valid_plane_i = np.mean(img, (1,2)) != -1 # Remove blank planes
    valid_plane_j = np.mean(img, (0,2)) != -1
    valid_plane_k = np.mean(img, (0,1)) != -1
    img = img[valid_plane_i,:,:]
    img = img[:,valid_plane_j,:]
    img = img[:,:,valid_plane_k]

    #img = np.interp(img, [-1,1], [0,255])
    img = resize(img, (IMG_SIZE, IMG_SIZE, IMG_SIZE), mode='constant', cval=-1)
    return img

def main():
    csv_file = os.path.join(DATA_DIR, 'Database', 'Final_Status_Phase-1_18_11_2019_13_12_04.csv')
    df = pd.read_csv(csv_file, sep=',', usecols=['INSP_STD_RAW'])
    df = df[~df['INSP_STD_RAW'].isnull()]
    img_list = df['INSP_STD_RAW'].tolist()

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
        if os.path.exists("./img_size_"+str(IMG_SIZE)+"_threshold_"+str(HIGH_THRESHOLD)+"/"+imgname.split('.')[0]+".npy"):
            continue
        try:
            img = sitk.ReadImage(DATA_DIR + img_list[idx])
        except Exception as e: # Some images are corrupted
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
        np.save("./img_size_"+str(IMG_SIZE)+"_threshold_"+str(HIGH_THRESHOLD)+"/"+imgname.split('.')[0]+".npy", img)
        #img = sitk.GetImageFromArray(img)
        #img = sitk.Cast(img, sitk.sitkInt16)
        #sitk.WriteImage(img, "./img_size_"+str(IMG_SIZE)+"/"+imgname)

if __name__ == '__main__':
    main()
