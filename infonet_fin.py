from numpy import *
from operator import eq
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shelve
from scipy.linalg import *
from scipy import ndimage
from scipy import stats

################################ data ##############################################
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
    #l is a list of filenames
    data=[]
    for i in range(len(l)):
        data.extend(readCsv('{0}.txt'.format(l[i])))
    return array(data)

def loadData(filename,nums=None,lendata=None,numClasses=None,loadwhat=None):
    if loadwhat==None:
        loadwhat=['testX','testY','f','adj']
    s=shelve.open(filename)
    loaded=[s[e] for e in loadwhat]
    if ('testX' in loadwhat) or ('testY' in loadwhat):
        testX=loaded[loadwhat.index('testX')]; testY=loaded[loadwhat.index('testY')]
        exlen=int(len(testX)/numClasses)
        X=zeros([len(nums)*lendata,len(testX[0])]);
        Y=zeros(shape(X))
        ind=[int(e) for e in nums]
        for i in range(len(nums)):
            X[i*lendata:(i+1)*lendata,:]=testX[(ind[i]-1)*exlen:(ind[i]-1)*exlen+lendata,:]
            Y[i*lendata:(i+1)*lendata,:]=testY[(ind[i]-1)*exlen:(ind[i]-1)*exlen+lendata,:]
        loaded[loadwhat.index('testX')]=X
        loaded[loadwhat.index('testY')]=Y
    print 'data loaded'
    return loaded

################################### distributions ###################################
def getNeighbors(pos,size):
    #pos is within [0,size^2-1]
    #[n s e w]
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


def allNeighbors(pixlen):
    size=int(math.sqrt(pixlen))
    nb={}
    for i in range(pixlen):
        nb[i]=getNeighbors(i,size)
    return nb

def pixDist(data,res):
    dist=zeros([res,len(data[0])])
    for p in range(res):
        for r in range(len(data)):
            dist[p]+=array(data[r])==p
    dist/=len(data)
    return dist  

def pairDist(data,res=16):
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
            

######################################## scoring ###############################
def indScore(pX,pY,Y,w,res):
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
        f[x1,:]=JY[0,:]/array([max(e,0.001) for e in pY[x1,:]])
    for img in range(len(Y)):
        for x1 in range(len(Y[0])):
            sc[img,x1]=f[x1,Y[img,x1]]
    sc=sum(sc,1)
    return sc

def corFunc(pX,pY,pXX,pYY,w):
    res=len(pX[0])
    size=int(sqrt(len(pX)))
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

def corScore(Y,f,adj):
    size=int(math.sqrt(len(Y[0])))
    nb=allNeighbors(len(Y[0]))
    sc=zeros(shape(Y))
    for img in range(len(Y)):
        for x1 in range(len(Y[0])):
            sc[img,x1]=f[x1,Y[img,x1]]
            for n in range(len(nb[x1])):
                x2=nb[x1][n]
                sc[img,x1]+=adj[x1][n][Y[img,x1],Y[img,x2]]
    sc=sum(sc,1)
    return sc

def largeScore(Y,f,adj):
    size=int(math.sqrt(len(Y[0])))
    nb=allNeighbors(len(Y[0]))
    sc=zeros(shape(Y))
    for img in range(len(Y)):
        for x1 in range(len(Y[0])):
            sc[img,x1]=f[x1,Y[img,x1]]
            scadj=[adj[x1][n][Y[img,x1],Y[img,nb[x1][n]]] for n in range(len(nb[x1]))]
            sc[img,x1]+=sum(scadj)
    sc=sum(sc,1)
    return sc

##################################### score-passing ##############################

def localFunc(pX,pY,pXX,pYY,w,x1,x2,f2,Y):
    #NOTE: this function updates x1 based on x2!!!
    #Y is a single image here
    #n is neighbor index of x1, 0-3
    #x1 is index of accepting node
    #notation: y2>>>y1 message passing
    res=len(pX[0])
    size=int(sqrt(len(pX)))
    #f=zeros(shape(pX))
    f=zeros(len(pX[0]))
    #adj=[None for _ in range(len(pX))]
    #x2=nb[n]

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
          
    #adj[x1]=[None for _ in range(len(nb))]
        
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

    #adj[x1][n]=zeros([res,res])
    adj=zeros([res,res])
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
                    #adj[x1][n][chi1,chi2]+=a[i,j]*term1*f2
                    adj[chi1,chi2]+=a[i,j]*term1*f2
                    
    #f[x1,:] = JY1[0,:]/[max(e,0.001) for e in pY[x1,:]]
    #sc=f[x1,Y[x1]]+adj[x1][n][Y[x1],Y[x2]]

    f = JY1[0,:]/[max(e,0.001) for e in pY[x1,:]]
    sc=f[Y[x1]]+adj[Y[x1],Y[x2]]
    return sc

def localInd(x1,pX,pY,Y,w):
    #Y is a single image
    res=shape(pX)[1]
    sc=zeros(shape(Y))
    JY=zeros([res,res])

    f=zeros(shape(pY))
        
    ymat=eye(res)*(1/array([max(e,0.001) for e in sqrt(pY[x1])]))
    xmat=eye(res)*sqrt(pX[x1])
    rmv=dot(transpose(sqrt(array([pY[x1]]))),sqrt(array([pX[x1]])))
    
    B=dot(ymat,dot(w,xmat)) - rmv
    u,s,v=svd(B)
    for i in range(res):
        for chi in range(res):
            JY[i,chi]=u[chi,i]
    f[x1,:]=JY[0,:]/array([max(e,0.001) for e in pY[x1,:]])
    
        
    sc[x1]=f[x1,Y[x1]]
    sc=sum(sc)
    return sc

    
####################################### channels #################################
def getChan(res,centerVar=0,distVar=0.1,sym=False):
    w=eye(res)*(1-centerVar)+distVar*abs(random.randn(res,res))
    for i in range(len(w[0])): #column-normalized
        w[:,i]/=sum(w[:,i])
    w=(w+w.T)/2 if sym else w
    return w

def centeredChannel(data,w):
    res=len(w)
    Y=zeros(data.shape)
    for i in range(len(data)):
        chan=[stats.rv_discrete(name='h',values=(arange(res),w[:,pix])) for pix in data[i]]
        Y[i,:]=[c.rvs() for c in chan]
    return Y

def lostBits(X,Y):
    trans=[[0,0],[0,1],[1,0],[1,1]]
    cng=zeros(shape(X))
    for img in range(len(X)):
        for pix in range(len(X[0])):
            bef=trans[X[img,pix]-1]
            aft=trans[Y[img,pix]-1]
            cng[img,pix]+=sum(abs(bef-aft))
    cng=sum(cng,0)
    cng/=len(X)
    return cng
        
def mutInfo(X,pX,pY,w,res):
    minf=zeros(shape(X))
    for img in range(len(X)):
        for pix in range(len(X[0])):
            for chi1 in range(res):
                for chi2 in range(res):
                    minf[img][pix]+=w[chi2,chi1]*pX[pix,chi1]*\
                            log2(w[chi2,chi1]/max(pY[pix,chi2],0.001))
    minf=sum(minf,0)
    minf/=len(X)
    return minf

################################# plotting #######################################
def showplot(scores,nums,showpics=False,X=None,Y=None):
    if showpics:
        lendata=int(len(scores[0])/len(nums))
        size=int(math.sqrt(len(X[0])))
        n=len(X)
        numY=int(sqrt(n))
        while(n%numY!=0):
            numY-=1
        numX=n/numY
        plt.figure(1)
        ind=[i[0] for i in sorted(enumerate(scores), key=lambda x:x[1])]
        imgs=[None for _ in range(numY*numX)]
        print 'getting pics... ('+repr(lendata)+' examples)'
        for i in range(numY):
            for j in range(numX):
                pltnum=numX*i+j+1
                imgs[pltnum-1]=plt.subplot(numY,numX,pltnum)
                imgs[pltnum-1].axes.get_xaxis().set_visible(False)
                imgs[pltnum-1].axes.get_yaxis().set_visible(False)
                imgs[pltnum-1].imshow(reshape(Y[ind[pltnum-1]],(size,size)),
                                      interpolation='none',cmap = cm.Greys_r)

        plt.figure(2)
        imgs=[None for _ in range(numY*numX)]
        for i in range(numY):
            for j in range(numX):
                pltnum=numX*i+j+1
                imgs[pltnum-1]=plt.subplot(numY,numX,pltnum)
                imgs[pltnum-1].axes.get_xaxis().set_visible(False)
                imgs[pltnum-1].axes.get_yaxis().set_visible(False)
                imgs[pltnum-1].imshow(reshape(X[ind[pltnum-1]],(size,size)),
                                      interpolation='none',cmap = cm.Greys_r)
        scatnum=3
    else:
        scatnum=1
    
    plt.figure(scatnum)
    titles=['unadjusted','NN adjusted','largest NN','propagated NN']
    pltcolors=['#FF0000','#FF7F00','#FFFF00','#00FF00','#0000FF','#FF1493','#8F00FF',\
               'k','w']
    lendata=int(len(scores[0])/len(nums))

    for ind in range(len(scores)):
        temp=[None for _ in range(len(nums))]
        ptemp=plt.subplot(len(scores),1,ind+1)
        for n in range(len(nums)):
            temp[n]=plt.scatter([scores[ind][i] for i in range(n*lendata,(n+1)*lendata)],
                        [n+1 for _ in range(n*lendata,(n+1)*lendata)],c=pltcolors[n])
            temp[n].axes.get_yaxis().set_visible(False)
        ptemp.set_title(titles[ind])
        plt.legend(temp,nums)
    plt.show()

