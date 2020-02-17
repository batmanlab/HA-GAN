from torch.utils.data import Dataset
import numpy as np
import glob

ROOT_DIR = "/home/lisun/work/copd/3dgan/data/"

class BRATS_dataset(Dataset):

    def __init__(self, img_size=240, stage="train"):
        self.root_dir = ROOT_DIR + "brats_" + str(img_size) + "/"
        self.stage = stage

        seg_list = []
        sid_list = []
        for item in glob.glob(self.root_dir+"*_seg.npy"):
            seg_list.append(item.split('/')[-1][:-8])
        for item in glob.glob(self.root_dir+"*_t1.npy"):
            if item.split('/')[-1][:-7] not in seg_list:
                sid_list.append(item.split('/')[-1][:-7])
        sid_list.sort()
        seg_list.sort()
        seg_list = np.asarray(seg_list)
        sid_list = np.asarray(sid_list)

        permutation_idx = np.random.RandomState(seed=0).permutation(seg_list.shape[0])
        if stage=="train":
            self.sid_list = np.concatenate([sid_list, seg_list[permutation_idx[50:]]])
            permutation_idx = np.random.RandomState(seed=0).permutation(self.sid_list.shape[0])
            self.sid_list = self.sid_list[permutation_idx]
        else:
            self.sid_list = seg_list[permutation_idx[:50]]
        print("Dataset size:", len(self))

    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, idx):
        img_t1 = np.load(self.root_dir+self.sid_list[idx]+"_t1.npy")
        img_flair = np.load(self.root_dir+self.sid_list[idx]+"_flair.npy")
        if self.stage != "train":
            img_seg = np.load(self.root_dir+self.sid_list[idx]+"_seg.npy")
            return img_t1[None,:,:,:], img_flair[None,:,:,:], img_seg[None,:,:,:]
        return img_t1[None,:,:,:], img_flair[None,:,:,:]
