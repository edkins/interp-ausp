import torch
from easy_transformer import EasyTransformer
from sklearn.decomposition import PCA
import numpy as np
import json

device = 'cpu'
torch.set_grad_enabled(False)

model = EasyTransformer.from_pretrained('gpt2', device=device)
unembed = model.unembed.W_U.data

with open('matrix.bin', 'wb') as f:
    unembed.T.numpy().tofile(f)

pca = PCA().fit(unembed.T)
obj = [[float(x) for x in xs] for xs in pca.components_]
var = [float(v) for v in pca.explained_variance_]
with open('pca.json', 'w') as f:
    json.dump({'d_vocab':model.cfg.d_vocab, 'd_model':model.cfg.d_model, 'variance':var, 'pca':obj}, f, indent=4)
