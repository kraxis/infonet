from numpy import *
from operator import eq
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shelve
from scipy.linalg import *
from scipy import ndimage
from scipy import stats

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
