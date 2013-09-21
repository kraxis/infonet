#@author Anand Srinivasan

#include in directory: infonet_fin and infonet_save
#defaults:
#   data input:
#       -each class has its own .txt CSV file in the default dir
#       -each .txt file is structured as: row-example,column-feature
#       -user specifies a list of filenames to infonet_save.getData()
#       -getData() returns a single (totExamples x featureLen) numpy array
#           which contains data from all specified files in concatenated form
#   computation:
#       -infonet_fin.pairDist() is non-redundant (transpose dist[j,i] if [i,j]
#           not in dist)
#       -score adjustment 'adj' from corScore() is accessed [centerNode,
#           neighborIndex],[y1,y2]
#       -infonet_fin.lostBits() uses Hamming distance
#       -both mutInfo and lostBits compute using empirical average
#   loading & saving:
#       -infonet_save.saveData() computes empiricals and scorefunc over 1000
#           examples of each class and stores a 500-size test fold
#       -infonet_save.loadData() takes class names and num examples


import infonet_fin as repo
import pgm as pgm
import infonet_save as saver
from numpy import *
import math

nums=['8','9']
lendata=100
res=4
numClasses=9 #classes over which empiricals were computed/saved

load=['testX','testY','pX','pY','pXX','pYY','w','f','adj']
loaded=repo.loadData('infonet_vars.db',nums,lendata,numClasses,load)
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
                                    
    while(propsc.count(None)>5):
        temp=random.randint(0,len(nb[fromNode]))
        thisNode=nb[fromNode][temp] 
        if (thisNode==prevNode):thisNode=nb[fromNode][(temp+1)%len(nb[fromNode])]
        propsc[thisNode]=repo.localFunc(pX,pY,pXX,pYY,w,thisNode,
            fromNode,propsc[fromNode],Y[img])
        prevNode=fromNode
        fromNode=thisNode
    propscores[img]=sum([e for e in propsc if e is not None])
        
print 'propscored'
sc=[repo.indScore(pX,pY,Y,w,res),repo.corScore(Y,f,adj),
    repo.largeScore(Y,f,adj),propscores]
    
print 'scored imgs'

repo.showplot(sc,nums,showpics=True,X=X,Y=Y)
