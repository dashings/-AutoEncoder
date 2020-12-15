#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %load_ext autoreload
# %autoreload 2
import SparseAutoEncoder
import torch as t
from SparseAutoEncoder import  SparseAutoEncoder as sae
t.manual_seed(1)
model = sae(BETA=5.0, ROU=0.01, hiddenshape=100,USE_P=True)
model.trainNN(lr=0.001, weight_decay=0, epochs=15)
from torchvision.utils import make_grid,save_image

hidden=model.cal_hidden()
res=make_grid(hidden,normalize=True, scale_each=False)
model.showim(res)
save_image(hidden,normalize=True,fp='./hidden.png')
