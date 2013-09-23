import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import *
from scipy import ndimage
from scipy import stats
from sklearn import svm
import pylab as pl

import infonet_fin as repo

##propscoring #################
def propscore(loaded,iters):
    X,Y,pX,pY,pXX,pYY,w,f,adj=loaded
    size=int(sqrt(len(pX)))
    nb=repo.allNeighbors(len(pX))
    propscores=zeros(len(Y))

    for img in range(len(Y)):
        print 'image being processed: '+repr(img)
        propsc=[None for _ in range(len(pX))]

        fromNode=random.randint(0,len(pX)) #start somewhere random
        prevNode=nb[fromNode][random.randint(0,len(nb[fromNode]))]
        
        propsc[prevNode]=repo.localInd(prevNode,pX,pY,Y[img],w)
        propsc[fromNode]=repo.localFunc(pX,pY,pXX,pYY,w,fromNode,prevNode,\
                                        propsc[prevNode],Y[img])
        ind=0              
        while(ind<iters):
            temp=random.randint(0,len(nb[fromNode]))
            thisNode=nb[fromNode][temp] 
            if (thisNode==prevNode):thisNode=nb[fromNode][(temp+1)%len(nb[fromNode])]
            propsc[thisNode]=repo.localFunc(pX,pY,pXX,pYY,w,thisNode,
                fromNode,propsc[fromNode],Y[img])
            prevNode=fromNode
            fromNode=thisNode
            ind+=1
        propscores[img]=sum([e for e in propsc if e is not None])
    return propscores

############################  template stuff ###########################


############################ svm testing ################################
def svmplot(X,Y):
    h = .02  
    C = 1.0 
    svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    poly_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)#svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
    lin_svc = svm.LinearSVC(C=C).fit(X, Y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    print 'done'
    titles = ['SVC with linear kernel',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel',
              'LinearSVC (linear kernel)']

    print 'got here'
    for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        pl.subplot(2, 2, i + 1)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
        pl.axis('off')

        # Plot also the training points
        pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

        pl.title(titles[i])

    pl.show()
