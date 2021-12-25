import random
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt

N=1000
K=20
Category=4

class Gauss():
    def __init__(self,tag,dim):
        self.tag=tag
        self.dim=dim
        self.mu=[np.random.random() for i in range(self.dim)]
        self.sigma=[np.random.random() for i in range(self.dim)]

    def sample(self):
        x=np.random.normal(self.mu,self.sigma)
        return x

def sampleBatch(n_sample,dim):
    X,Y=[],[]
    gaussList=[Gauss(i,dim) for i in range(Category)]
    for i in range(n_sample):
        gauss=random.sample(gaussList,1)
        X.append(gauss[0].sample().tolist())
        Y.append(gauss[0].tag)

    return np.array(X),np.array(Y)

def normalize(X):
    maxcol=X.max(axis=0)
    mincol=X.min(axis=0)

    tmp=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        tmp[:,i]=(X[:,i]-mincol[i])/(maxcol[i]-mincol[i])
    return tmp

if __name__ == '__main__':
    X,Y=sampleBatch(N,K)

    projector=manifold.TSNE(n_components=2)
    X_tsne=projector.fit_transform(X)

    X_tsne=normalize(X_tsne).T

    plt.scatter(X_tsne[0],X_tsne[1],s=5,c=Y,cmap='coolwarm')
    plt.show()

