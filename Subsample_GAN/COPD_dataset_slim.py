from torch.utils.data import Dataset
import numpy as np
import glob

ROOT_DIR = "/home/lisun/work/copd/3dgan/data/"

class COPD_dataset(Dataset):

    def __init__(self, img_size=64, stage="train", threshold=-250):
        self.sid_list = []
        self.root_dir = ROOT_DIR + "img_size_" + str(img_size)
        if threshold != -250:
            self.root_dir = self.root_dir + "_threshold_" + str(threshold) + "/"
        else:
            self.root_dir = self.root_dir + "/"
        for item in glob.glob(self.root_dir+"*.npy"):
            self.sid_list.append(item.split('/')[-1])

        self.sid_list.sort()
        self.sid_list = np.asarray(self.sid_list)
        permutation_idx = np.random.RandomState(seed=0).permutation(self.sid_list.shape[0])
        if stage=="train":
            self.sid_list = self.sid_list[permutation_idx[1000:]]
        else:
            self.sid_list = self.sid_list[permutation_idx[:1000]]
        print("Dataset size:", len(self))

    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, idx):
        img = np.load(self.root_dir+self.sid_list[idx])
        #img = np.swapaxes(img,1,2)
        #img = np.flip(img,1)
        #img = np.flip(img,2)
        return img[None,:,:,:]
