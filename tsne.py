import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
from cuml.manifold import TSNE
import pickle

def load_mnist_train(path):
    """Load MNIST data from path"""
    labels_path = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, 'train-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels


images, labels = load_mnist_train("data/fashion")

tsne = TSNE(n_components = 2, method = 'barnes_hut', random_state=23)
embedding = tsne.fit_transform(images)

print(embedding[:10], embedding.shape, type(embedding))

outdir = '/opt/dkube/output/'
if not os.path.exists(outdir + 'model'):
        os.makedirs(outdir + 'model')
                
with open(outdir + 'model/embeddings.pickle', 'wb') as handle:
    pickle.dump(embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)