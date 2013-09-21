from numpy import *
from operator import eq
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shelve
from scipy.linalg import *
from scipy import ndimage
from scipy import stats

############################ data functions #########################
def readCsv(filepath):
    f=open(filepath,'r')
    content=f.readlines()
    for i in range(len(content)):
        content[i]=[float(x) for x in content[i].strip().split(',')]    
    f.close();
    return content

def writeCsv(data,filename):
    f=open(filename,'w')
    line=''
    for i in range(len(data)):
        for j in range(len(data[i])):
            if j < (len(data[i])-1):
                line += repr(data[i][j]) + ','
            else:
                line += repr(data[i][j]) + '\n'
            f.write(line)
            line=''
    f.close()

def getData(l):
    #l is a list of strings
    data=[]
    for i in range(len(l)):
        data.extend(readCsv('{0}.txt'.format(l[i])))
    return array(data)

def loadData(filename,nums,lendata):
    s=shelve.open(filename)
    pX=s['pX']; pXX=s['pXX']
    pY=s['pY']; pYY=s['pYY']
    tX=s['X'];   tY=s['Y']
    w=s['w']
    exlen=int(len(tX)/9)
    X=zeros([len(nums)*lendata,len(tX[0])]);
    Y=zeros(shape(X))
    ind=[int(e) for e in nums]
    for i in range(len(nums)):
        X[i*lendata:(i+1)*lendata,:]=tX[(ind[i]-1)*exlen:(ind[i]-1)*exlen+lendata,:]
        Y[i*lendata:(i+1)*lendata,:]=tY[(ind[i]-1)*exlen:(ind[i]-1)*exlen+lendata,:]
    print 'data loaded'
    return X,Y,pX,pY,pXX,pYY,w

############################ channels #################################
def getChan(res,centerVar=0.0,distVar=0.0,sym=False):
    #sym = symmetric?
    w=eye(res)*(1-centerVar)+distVar*abs(random.randn(res,res))
    for i in range(len(w[0])): #column-normalized
        w[:,i]/=sum(w[:,i])
    w=(w+w.T)/2 if sym else w
    return w

def centeredChannel(data,w):
    res=len(w)
    Y=zeros(data.shape)
    for i in range(len(data)):
        chan=[stats.rv_discrete(values=(arange(res),w[:,pix])) for pix in data[i]]
        Y[i,:]=[c.rvs() for c in chan]
    return Y

############################ distributions ############################

def pairDist(data,res):
    #access:[fromNode,toNode][px1_val][px2_val]
    size=int(math.sqrt(len(data[0])))
    dist={}

    for i in range(len(data)):
        for j in range(len(data[0])):
            nb=getNeighbors(j,size)
            for n in nb:
                if(i==0 and ((n,j) not in dist)):
                    dist[j,n]=zeros([res,res])
                    
                for aR in range(res):
                    for aC in range(res):
                        
                        if(n,j) not in dist:
                            dist[j,n][aR][aC]+=[data[i][j],data[i][n]]==[aR,aC]
                        if(i==len(data)-1 and ((n,j) not in dist)):
                            dist[j,n][aR][aC]=float(dist[j,n][aR][aC])/len(data)
    return dist

def getNeighbors(pos,size):
    #pos is within [0,size^2-1]
    #return format [n,s,e,w]
    nb=[];
    if (pos >= size):
        n=pos-size
        nb.append(n)
    if (pos < pow(size,2)-size):
        s=pos+size
        nb.append(s)
    if (pos%size != size-1):
        e=pos+1
        nb.append(e)
    if (pos%size != 0):
        w=pos-1
        nb.append(w)
        
    return nb

def pixDist(data,res):
    dist=zeros([res,len(data[0])])
    for p in range(res):
        for r in range(len(data)):
            dist[p]+=array(data[r])==p
    dist/=len(data)
    return dist

############################# scoring ###############################
def pixScore(pX,pY,Y,w,res):
    sc=zeros(shape(Y))
    JY=zeros([res,res])

    f=zeros(shape(pY))
    for x1 in range(len(Y[0])):
        ymat=eye(res)*(1/array([max(e,0.001) for e in sqrt(pY[x1])]))
        xmat=eye(res)*sqrt(pX[x1])
        rmv=dot(transpose(sqrt(array([pY[x1]]))),sqrt(array([pX[x1]])))
        
        B=dot(ymat,dot(w,xmat)) - rmv
        u,s,v=svd(B)
        for i in range(res):
            for chi in range(res):
                JY[i,chi]=u[chi,i]
        f[x1,:]=JY[0,:]/array([max(e,0.001) for e in sqrt(pY[x1,:])])
    for img in range(len(Y)):
        for x1 in range(len(Y[0])):
            sc[img,x1]=f[x1,Y[img,x1]]
    sc=sum(sc,1)
    return sc

def corScore(pX,pY,pXX,pYY,w):
    res=len(pX[0])
    size=int(math.sqrt(len(pX)))
    f=zeros(shape(pX))
    adj=[None for _ in range(len(pX))]
    for x1 in range(len(pX)):
        nb=getNeighbors(x1,size)

        ymat1=eye(res)*(1/array([max(e,0.001) for e in sqrt(pY[x1])]))
        xmat1=eye(res)*sqrt(pX[x1])
        rmv=dot(transpose(sqrt(array([pY[x1]]))),sqrt(array([pX[x1]])))
        B1=dot(ymat1,dot(w,xmat1)) - rmv
        u1,s1,v1=svd(B1)

        JX1=zeros([res,res])
        JY1=zeros(shape(JX1))
        for i in range(res):
            for chi in range(res):
                JX1[i,chi]=sqrt(pX[x1,chi])*v1[chi,i] 
                JY1[i,chi]=sqrt(pY[x1,chi])*u1[chi,i]

        rX1=zeros([res,res])
        rY1=zeros(shape(rX1))
        for i in range(res):
            for j in range(res):
                for chi in range(res):
                    rX1[i,j]+=JX1[0,chi]*JX1[i,chi]*JX1[j,chi]/max(square(pX[x1,chi]),0.001)
                    rY1[i,j]+=JY1[0,chi]*JY1[i,chi]*JY1[j,chi]/max(square(pY[x1,chi]),0.001)
              
        adj[x1]=[None for _ in range(len(nb))]
        for n in range(len(nb)):
            x2=nb[n]
                
            JX2=zeros([res,res])
            JY2=zeros(shape(JX2))
            rX2=zeros([res,res])
            rY2=zeros(shape(rX2))
            a=zeros([res,res])
                
            B2=dot(eye(res)*(1/array([max(e,0.001) for e in sqrt(pY[x2])])),dot(w,eye(res)*\
                sqrt(pX[x2]))) - dot(transpose(sqrt(array([pY[x2]]))),sqrt(array([pX[x2]])))
            u2,s2,v2=svd(B2)

            for i in range(res):
                for chi in range(res):
                    JX2[i,chi]=sqrt(pX[x2,chi])*v2[chi,i] 
                    JY2[i,chi]=sqrt(pY[x2,chi])*u2[chi,i]
            for i in range(res):
                for j in range(res):
                    for chi in range(res):
                        rX2[i,j]+=JX2[0,chi]*JX2[i,chi]*JX2[j,chi]/max(square(pX[x2,chi]),0.001)
                        rY2[i,j]+=JY2[0,chi]*JY2[i,chi]*JY2[j,chi]/max(square(pY[x2,chi]),0.001)
                        for chi2 in range(res):
                            pxx=pXX[x1,x2] if (x1,x2) in pXX else transpose(pXX[x2,x1])
                            a[i,j]+=(pxx[chi,chi2]/max((pX[x1,chi]*pX[x2,chi2]),0.001))* \
                                        JX1[i,chi]*JX2[j,chi2]

            adj[x1][n]=zeros([res,res])
            for chi1 in range(res):
                for chi2 in range(res):
                    for i in range(res):
                        for j in range(res):
                            term1=0
                            for k in range(res):
                                if k==0:
                                    gam=-1*s1[i]*s2[j]*rY1[i,0]
                                else:
                                    gam=(s1[0]*s1[k]*s2[j]*rX1[i,k] - square(s1[0])*\
                                         s1[i]*s2[j]*rY1[i,k])/(square(s1[0])-square(s1[k]))
                                term1+=gam*JY1[k,chi1]/max(pY[x1,chi1],0.001)
                            adj[x1][n][chi1,chi2]+=a[i,j]*term1*JY2[j,chi2]/max(pY[x2,chi2],0.001)
                            
        f[x1,:] = JY1[0,:]/[max(e,0.001) for e in pY[x1,:]]
    return f,adj

########################## saving & updating ##########################
def saveData(nums=None,trainFold=1000,testFold=500,chanparams=[0.3,0.5],res=4):
    if nums==None:
        nums=['1','2','3','4','5','6','7','8','9']
    X=getData(nums)
    print 'data loaded'
    exlen=int(len(X)/9)
    
    trainX=zeros([len(nums)*trainFold,len(X[0])]); 
    testX=zeros([len(nums)*testFold,len(X[0])]); 
    
    for i in range(len(nums)):
        trainX[i*trainFold:(i+1)*trainFold,:]=X[i*exlen:i*exlen+trainFold,:]
        testX[i*testFold:(i+1)*testFold,:]=X[i*exlen+trainFold:i*exlen+trainFold+testFold,:]
    print 'trainers loaded'
    
    w=getChan(res=res,centerVar=chanparams[0],distVar=chanparams[1])
    trainY,testY=centeredChannel(trainX,w),centeredChannel(testX,w)
    print 'testers loaded'
    
    pX=transpose(pixDist(trainX,res))
    pXX=pairDist(trainX,res)
    pY=transpose(pixDist(trainY,res))
    pYY=pairDist(trainY,res)
    f,adj=corScore(pX,pY,pXX,pYY,w)
    print 'distributions & scorefunc computed'
    
    s=shelve.open('infonet_vars.db')
    s['pX']=pX; s['pXX']=pXX
    s['pY']=pY; s['pYY']=pYY
    s['w']=w
    s['testX']=testX;   s['testY']=testY
    s['f']=f;   s['adj']=adj
    s.close()
    print 'all data saved'

