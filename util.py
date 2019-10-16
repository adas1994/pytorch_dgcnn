import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os, h5py


class Jet_PointCloudDataSet(Dataset):
    def __init__(self, file_list, grp_list, dir_name=None):
        if dir_name==None:
            dir_name = os.getcwd()
        print(dir_name)
        num_files = len(file_list)
        assert len(file_list)==len(grp_list)
        en_H, en_W, en_Z, en_jet, en_rest, en_t, jet_dataset = [], [], [], [], [], [], []
        #['en_H', 'en_W', 'en_Z', 'en_jet', 'en_rest', 'en_t', 'jet_label', 'pt_H', 'pt_W', 'pt_Z', 'pt_jet', 'pt_rest', 'pt_t'] 
        y = []
        for i in range(num_files):
            file_name = file_list[i] + '.hdf5'
            f_path = os.path.join(dir_name,file_name)
            assert os.path.exists(f_path)
            f = h5py.File(f_path,'r')
            grp = f[grp_list[i]]
            stuff = list(grp.values())
            #print(list(grp.keys()))
            dset1,  dset2 = stuff[0][()], stuff[1][()]
            dset3,  dset4 = stuff[2][()], stuff[3][()]
            dset5,  dset6 = stuff[4][()], stuff[5][()]
            dset7,  dset8 = stuff[6][()], stuff[7][()]
            dset7  = dset7.reshape(len(dset7),1)
            if len(en_H)<1:
                en_H.append(dset1)
                en_W.append(dset2)
                en_Z.append(dset3)
                en_jet.append(dset4)
                en_rest.append(dset5)
                en_t.append(dset6)
                y.append(dset7)
                jet_dataset.append(dset8)
                en_H        = np.squeeze(np.asarray(en_H))
                en_W        = np.squeeze(np.asarray(en_W))
                en_Z        = np.squeeze(np.asarray(en_Z))
                en_jet      = np.squeeze(np.asarray(en_jet))
                en_rest     = np.squeeze(np.asarray(en_rest))
                en_t        = np.squeeze(np.asarray(en_t))
                jet_dataset = np.squeeze(np.asarray(jet_dataset))
                y           = np.squeeze(np.asarray(y))
                y           = y.reshape(len(y),1)

            else:
                en_H    = np.vstack((en_H,dset1))
                en_W    = np.vstack((en_W,dset2))
                en_Z    = np.vstack((en_Z,dset3))
                en_jet  = np.vstack((en_jet,dset4))
                en_rest = np.vstack((en_rest,dset5))
                en_t    = np.vstack((en_t,dset6))
                jet_dataset = np.vstack((jet_dataset, dset8))                                                                                                                                         
                y = np.vstack((y,dset7))

        self.en_H = torch.from_numpy(en_H)
        self.en_W = torch.from_numpy(en_W)
        self.en_Z = torch.from_numpy(en_Z)
        self.en_jet = torch.from_numpy(en_jet)
        self.en_rest = torch.from_numpy(en_rest)
        self.en_t = torch.from_numpy(en_t)
        self.X = torch.cat((self.en_H,self.en_t,self.en_W, self.en_H, self.en_jet, self.en_rest),dim=1)
        self.Y = torch.from_numpy(y)
        self.data_len = len(y)
    
    def __len__(self):
        return self.data_len

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx]
        y = self.Y[idx]
        sample_data = (x,y)
        return sample_data
        
    


def dataset_producer(file_list, grp_list,dir_name=None):# copied from DGCNN/temp_train.py
    if dir_name==None:
        dir_name = os.getcwd()
    num_files = len(file_list)
    assert len(file_list)==len(grp_list)
    en_H, en_W, en_Z, en_jet, en_rest, en_t = [], [], [], [], [], []
    pt_H, pt_W, pt_Z, pt_jet, pt_rest, pt_t = [], [], [], [], [], []
    #['en_H', 'en_W', 'en_Z', 'en_jet', 'en_rest', 'en_t', 'jet_label', 'pt_H', 'pt_W', 'pt_Z', 'pt_jet', 'pt_rest', 'pt_t']
    y = []
    for i in range(num_files):
        file_name = file_list[i] + '.hdf5'
        f_path = os.path.join(dir_name,file_name)
        assert os.path.exists(f_path)
        f = h5py.File(f_path,'r')
        grp = f[grp_list[i]]
        stuff = list(grp.values())
        #print(list(grp.keys()))                                                                                                                          
        dset1,  dset2 = stuff[0][()], stuff[1][()]
        dset3,  dset4 = stuff[2][()], stuff[3][()]
        dset5,  dset6 = stuff[4][()], stuff[5][()]
        dset7,  dset8 = stuff[6][()], stuff[7][()]
        dset9,  dset10 = stuff[8][()], stuff[9][()]
        dset11, dset12 = stuff[10][()], stuff[11][()]
        dset13 = stuff[12][()]
        dset7  = dset7.reshape(len(dset7),1)
        #print(dset2.shape, dset1.shape)
        if len(en_H)<1:
            en_H.append(dset1)
            en_W.append(dset2)
            en_Z.append(dset3)
            en_jet.append(dset4)
            en_rest.append(dset5)
            en_t.append(dset6)
            y.append(dset7)

            en_H    = np.squeeze(np.asarray(en_H))
            en_W    = np.squeeze(np.asarray(en_W))
            en_Z    = np.squeeze(np.asarray(en_Z))
            en_jet  = np.squeeze(np.asarray(en_jet))
            en_rest = np.squeeze(np.asarray(en_rest))
            en_t    = np.squeeze(np.asarray(en_t))
            y       = np.squeeze(np.asarray(y))
            y       = y.reshape(len(y),1)
        else:
            en_H    = np.vstack((en_H,dset1))
            en_W    = np.vstack((en_W,dset2))
            en_Z    = np.vstack((en_Z,dset3))
            en_jet  = np.vstack((en_jet,dset4))
            en_rest = np.vstack((en_rest,dset5))
            en_t    = np.vstack((en_t,dset6))
            #print(y.shape,dset1.shape)                                                                                                                   
            y = np.vstack((y,dset7))
            
    return (en_H, en_W, en_Z, en_jet, en_rest, en_t, y)


def make_onehot(target,num_classes=5,smoothing=False):
    target = target.contiguous().view(-1)
    if num_classes is None:
        target_array = target.cpu().numpy()
        t = np.unique(target_array)
        num_classes = len(t)
    batch_size = target.shape[0]
    onehot_array = np.zeros((batch_size,num_classes))
    one_hot = torch.from_numpy(onehot_array)
    one_hot = one_hot.scatter(1,target.view(-1,1),1)
    return one_hot
    
def cal_loss(pred, target, smoothing=True):
    target = target.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        target = target.long()
        #print(target)
        #print(pred.shape,target.shape,target.view(-1,1).shape)
        one_hot = torch.zeros_like(pred).scatter(1,target.view(-1,1),1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()

    else:
        loss = F.cross_entropy(pred, target, reduction='mean')

    return loss


class IOStream():
    def __init__(self,path):
        self.f = open(path, 'a')
        
    def cprint(self,text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

        

class PointCloudDataSet(Dataset):
    def __init__(self, h5pyFile, dirname, grp):
        file_path = os.path.join(dirname,h5pyFile)
        print(file_path)
        assert os.path.exists(file_path)
        f = h5py.File(file_path,"r")
        data_grp = f[grp]
        self.data_grp = data_grp
        X, Y = list(data_grp.values())                                                                                     
        X, Y = X[()], Y[()]
        self.data_len = len(Y)
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.data_len


    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx]
        y = self.Y[idx]
        sample_data = (x,y)
        return sample_data
