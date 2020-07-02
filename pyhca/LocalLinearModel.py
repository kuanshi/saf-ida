"""Local linear model"""

import numpy as np
from sklearn import model_selection as skms
from scipy import stats as spst

__author__ = 'Kuanshi Zhong'

class LocalLinearRegression:
    
    def __init__(self,modelname='LLM',data=[],kerneltype='Gaussian',
                 modelselection=['CV',5],lambdabound=[0.5,2],ndiv=50):
        """
        __init__: initialization
        - Input:
            modelname: model name
            data: training data
            kerneltype: defining kernel type (default is Gaussian)
            modelselection: model selection method (default is 5-CV)
            lambdabound: lower- and upper-bounds of tunning factor
            ndiv: number of lambda divisions
        """
        # data check
        if data.size==0:
            print("Please check the input dimension.")
            return
        # randomize data
        np.random.seed(0)
        np.random.shuffle(data)
        # initializing model
        self.X = data[:,0:-1]
        self.y = data[:,-1]
        self.kerneltype = kerneltype
        self.modelselection = modelselection
        self.lambdabound = lambdabound
        self.ndiv = ndiv
        # trainning data size
        self.ndata = len(self.y)
        # generating lambda
        self.lambdarange = np.linspace(self.lambdabound[0], 
                                       self.lambdabound[1],self.ndiv)
        # optimal lambda
        self.lambda_opt = []
        # searching optimal model
        self.__model_development()
        
    def __model_development(self):
        """
        __model_development: searching the optimal model given user inputs.
        """
        if self.modelselection[0]=='LOOCV':
            nf = self.ndata
        elif self.modelselection[0]=='CV':
            nf = self.modelselection[1]
        else:
            print("Please define a valid selection method: \
                'LOOCV' or ['CV',#].")
            return
        kf = skms.KFold(n_splits=nf)
        meanerr = np.zeros((1,self.ndiv))
        ltag = 0
        for lambda_trial in self.lambdarange:
            err = np.zeros((1,nf))
            tmptag = 0
            for tag_train,tag_test in kf.split(self.y):
                err[0,tmptag] = np.mean(np.square(self.y[tag_test].reshape([-1,1])- \
                           self.modeleval(X_train=self.X[tag_train,:],
                                     y_train=self.y[tag_train].reshape([-1,1]),
                                     kntype=self.kerneltype,lambda0=lambda_trial,
                                     x0=self.X[tag_test,:])))
                tmptag = tmptag+1
            meanerr[0,ltag] = np.mean(err)
            ltag = ltag+1
        # find optimal lambda
        self.lambda_opt = self.lambdarange[meanerr.argmin()]
        # compute mean squared error
        pred_y = self.modeleval(X_train=self.X,y_train=self.y,kntype=self.kerneltype,
                                lambda0=self.lambda_opt,x0=self.X,rflag=0).reshape((-1,1))
        self.mse = np.mean(np.square(self.y.reshape((-1,1))-pred_y))
        
    def modeleval(self,X_train=[],y_train=[],kntype=[],lambda0=[],x0=[],rflag=0):
        """
        modelevel: evaluating the model at given point.
        - Input: 
            X_train: training data X
            y_train: training data y
            kntype: kernel type
            lambda0: tunning parameter
            x0: evaluated X locations
            rflag: return-value flag, 0 - median, 1 - standard deviation, and
            2 - MSE
        - Output:
            y0: median
            sigma0: standard deviation
            MSE: mean squared error
        """
        # initialization
        if len(X_train)==0:
            X_train = self.X
        if len(y_train)==0:
            y_train = self.y
        if len(kntype)==0:
            kntype = self.kerneltype
        if not lambda0:
            lambda0 = self.lambda_opt
        nTrain = y_train.size
        nEval = np.size(x0,0)
        y0 = np.zeros((nEval,1))
        sigma0 = 0
        K = np.zeros((nEval,nTrain))
        L = np.zeros((nEval,nEval))
        # computing kernel
        for i in range(0,nEval):
            K[i,:] = self.__computekernel(X_train,x0[i,:],kntype,lambda0).transpose()
        # evaluating local linear model
        for i in range(0,nEval):
            W = np.diag(K[i,:])
            tmpM1 = np.linalg.inv(X_train.transpose().dot(W).dot(X_train))
            tmpM2 = X_train.transpose().dot(W)
            tmpM3 = tmpM2.dot(y_train)
            y0[i,0] = x0[i,:].reshape((1,-1)).dot(tmpM1.dot(tmpM3))
            if rflag==1:
                L[i,:] = x0[i,:].reshape((1,-1)).dot(tmpM1.dot(tmpM2))
        # return values
        if rflag==0:
            return y0
        elif rflag==1:
            sigma0 = np.sqrt(np.sum(np.square(y_train.reshape((-1,1))-y0))/(nTrain-2.0* \
                             np.trace(L)+np.trace(L.transpose()*L)))
            return sigma0
        else:
            return self.mse            
        
    def __computekernel(self,xr,xt,kn,tp):
        """
        __computekernel: computing kernel smoothers
        - Input:
            xr: a matrix of reference points
            xt: the target location
            kn: kernel type
            tp: tunning parameter
        - Output:
            Ks: kernel smoothers
        """
        # initialization
        nRef = xr.shape[0]
        Ks = np.zeros((nRef,1))
        # computing the kernel
        dist = np.sqrt(np.sum(np.square((xr-np.ones((nRef,1))*xt)),axis=1))
        if kn=='KNN':
            tmpdist = np.sort(dist)
            Ks[np.where(dist==tmpdist[0:tp]),0] = 1
        elif kn=='Nadaraya-Watson':
            Ks = np.max(np.array(0.75*(1-(dist/tp)),np.zeros((nRef,1))),axis=1)
        elif kn=='Tricube':
            Ks = np.max(np.array(np.power(1-np.power(dist/np,3),3),
                                 np.zeros((nRef,1))),axis=1)
        elif kn=='Gaussian':
            Ks = spst.norm.pdf(dist/tp,loc=0,scale=1)
        else:
            print("Please input a valid kernel type: 'KNN', \
                  'Nadaraya-Waston, 'Tricube', or 'Gaussian'.")
            return
        Ks = Ks/np.sum(Ks)
        # return value
        return Ks
