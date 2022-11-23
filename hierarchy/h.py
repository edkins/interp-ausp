import torch
from easy_transformer import EasyTransformer
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA
import numpy as np
import json

n_clusters=500

class RecoveredNode:
    def __init__(self, depth, left, right, min_label, max_label):
        self.depth = depth
        self.left = left
        self.right = right
        self.min_label = min_label
        self.max_label = max_label

    @classmethod
    def leaf(cls, depth, label):
        return cls(depth, None, None, label, label)

    @classmethod
    def branch(cls, left, right):
        if left.max_label + 1 != right.min_label:
            raise Exception("Label mismatch on branch")
        if left.depth != right.depth:
            raise Exception("Depth mismatch on branch")
        return cls(left.depth - 1, left, right, left.min_label, right.max_label)

    @classmethod
    def from_tree(cls, depth, tree):
        if tree.left == None:
            return cls.leaf(depth, tree.label)
        else:
            return cls.branch(cls.from_tree(depth + 1, tree.left), cls.from_tree(depth + 1, tree.right))

    def select(self, data, cluster_labels):
        indices, = np.nonzero((cluster_labels >= self.min_label) & (cluster_labels <= self.max_label))
        return data[indices,:]

    def select_pca(self, data, cluster_labels, n_dims=3):
        print(f"Performing PCA, labels {self.min_label}-{self.max_label}   ({self.depth})")
        selected = self.select(data, cluster_labels)
        result = np.zeros((selected.shape[0], 3))
        n_dims = min(n_dims, selected.shape[0])
        if n_dims >= 2:
            result[:, :n_dims] = PCA(n_dims).fit_transform(selected)
        return result

    def _subtrees(self, result):
        result.append(self)
        if self.left != None:
            self.left._subtrees(result)
        if self.right != None:
            self.right._subtrees(result)

    def subtrees(self):
        result = []
        self._subtrees(result)
        return result

    def jsonable(self, node_list, payload):
        left_index = None
        if self.left != None:
            left_index = node_list.index(self.left)
        right_index = None
        if self.right != None:
            right_index = node_list.index(self.right)

        return {
            'depth': self.depth,
            'left': left_index,
            'right': right_index,
            'min_label': self.min_label,
            'max_label': self.max_label,
            'x': list(x for x in payload[:,0]),
            'y': list(y for y in payload[:,1]),
            'z': list(z for z in payload[:,2]),
        }

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f"Using {device} device")
torch.set_grad_enabled(False)

model = EasyTransformer.from_pretrained('gpt2', device=device)

# Convenience function for decoding token
decode = model.tokenizer.decode

# Convenience function for encoding token
def encode(t):
    global model
    result = model.tokenizer.encode(t)
    if len(result) != 1:
        raise Exception(f"Not a single token: {t}")
    return result[0]

unembed = model.unembed.W_U.data
embed = model.embed.W_E.data
d_M = model.cfg.d_model
d_V = model.cfg.d_vocab

unembed_norm = torch.nn.functional.normalize(unembed, dim=0)

print("Performing clustering")
clustering = BisectingKMeans(n_clusters=n_clusters, random_state=12345)
cluster_labels = clustering.fit_predict(unembed.T)

print("Getting list of subtree nodes")
nodes = RecoveredNode.from_tree(0, clustering._bisecting_tree).subtrees()

pcas = [n.select_pca(unembed.T, cluster_labels, 3) for n in nodes]

print("Producing json output")
output_obj = {
        "tokens": [decode(t) for t in range(d_V)],
        "cluster_labels": [int(label) for label in cluster_labels],
        "nodes": [n.jsonable(nodes,p) for n,p in zip(nodes,pcas)]
}

with open('nodes.json','w') as f:
    json.dump(output_obj, f, indent=4)
