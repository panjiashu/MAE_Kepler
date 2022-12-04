from glob import glob

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as savgol
from astropy.io import fits

import torch

from .preprocessing import excel_read, preprocess

class WeightedDataset(Dataset):
  def __init__(self, excel_root, lc_root, augment=True, segment_len=4000):
    self.augment = augment # whether there is data augmentation
    self.segment_len = segment_len # the length of light curve segments
    self.data = [] # model input
    kids, loggs, radiis, es = excel_read(40, excel_root) # KIC, logg, radius and error of logg
    paths1 = sorted(glob(lc_root+'*.fits.ps'))
    paths2 = sorted(glob(lc_root+'*.fits'))
    paths = [x for x in paths2 if x+'.ps' in paths1] # use light curves with power spectra in The SWAN
    
    flag = 0
    for i in range(int(len(paths))): # 如果该dataset类出错，减小读取长度以缩短调试时间
      path = paths[i]
      fp = fits.open(path)
      kid = fp[0].header['KEPLERID']
      if kid in kids: # use light curves with predictions by The SWAN
        radius = radiis[flag]
        flag += 1
        fp1data = fp[1].data
        lc = preprocess(fp1data)
        if lc is None:
          continue
        self.data.append(lc)
    
    self.data = [torch.FloatTensor(x) for x in self.data]
    std = [torch.std(x) for x in self.data]
    self.std = std
    self.maxstd = max(std)
    print(self.maxstd)
    self.data = [(self.data[i]-torch.mean(self.data[i]))/std[i]/(torch.log(self.maxstd/std[i])+1) for i in range(len(self.data))]
    self.downsample = [x[::1] for x in self.data] # downsampled model input
    self.label = torch.FloatTensor(loggs) # model target output
    self.kids = kids # Kepler 
    self.es = torch.FloatTensor(es)
    weights = torch.exp(-self.es/0.02)
    self.weights = weights

  def explore(self):
    pass
    
  def __getitem__(self, idx):
    lc = self.downsample[idx]
    segment_len = self.segment_len
    if self.augment:
      if len(lc) > segment_len:
        start = np.random.randint(0, len(lc) - segment_len)
        lc = lc[start:start+segment_len]
      # if np.random.random() < 0.5:
      #   lc = torch.flip(lc,[0])
      # if np.random.random() < 0.5:
      #   lc = -lc
    else:
      lc = lc[:segment_len]
    return lc, self.label[idx], self.weights[idx], self.kids[idx]

  def __len__(self):
    return len(self.data)