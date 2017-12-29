import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

##############################################################
# prepare data
lines = joblib.load('proc_dir/wbow_lines.pickle')
lines = [lines[idx] for idx in np.random.choice(len(lines), len(lines)/10, replace=False)]

X = np.concatenate([l[2][None,:] for l in lines], axis=0)
Xnorm = np.sqrt((X ** 2).sum(axis=1))[:,None]
X /= Xnorm

words = [l[1] for l in lines]
ids = [l[0] for l in lines]
del lines
import gc; gc.collect()


##############################################################
# tsne
import sys
sys.path.append('/home/ycao/third_party_src/bhtsne')
from bhtsne import bh_tsne

num_dims = 2 
pca_dims = 50 
perplexity = 50 
theta = .5

tsne_out = list(bh_tsne(X, num_dims, pca_dims, perplexity, theta, verbose=True))

joblib.dump({'tsne':tsne_out, 'words':words, 'ids':ids}, 'proc_dir/tsne_out.pickle', compress=9)
import pdb; pdb.set_trace()
