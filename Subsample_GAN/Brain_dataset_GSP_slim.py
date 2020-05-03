from skimage.transform import resize
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import glob
import os

ROOT_DIR = "/pghbio/dbmi/batmanlab/Data/BrainSegmentationData/"
DATASETS = ["GSP/"] # "GSP/"

class Brain_dataset(Dataset):

    def __init__(self, img_size=256, stage="train"):
        self.sid_list = []
        self.root_dir = ROOT_DIR
        self.img_size = img_size

        for dataset in DATASETS:
            for item in glob.glob(self.root_dir+dataset+"*/norm.mgz"):
                img_path = dataset+item.split('/')[-2]
                self.sid_list.append(img_path)

        self.sid_list.sort()
        self.sid_list = np.asarray(self.sid_list)
        permutation_idx = np.random.RandomState(seed=0).permutation(self.sid_list.shape[0])
        if stage=="train":
            self.sid_list = self.sid_list[permutation_idx[500:]]
        else:
            self.sid_list = self.sid_list[permutation_idx[:500]]
        print("Dataset size:", len(self))

    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, idx):
        img = nib.load(self.root_dir+self.sid_list[idx]+"/norm.mgz")
        img = img.get_fdata().transpose((1,2,0))[30:-50,20:-20,20:-20]
        img = resize(img, (self.img_size, self.img_size, self.img_size), mode='constant', cval=0.)
        img = (img / 199.)*2.-1. # Rescale to [-1,1]
        #img = np.swapaxes(img,1,2)
        #img = np.flip(img,1)
        #img = np.flip(img,2)
        return img[None,:,:,:]
