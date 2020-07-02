"""Global linear model"""

import numpy as np
from sklearn import linear_model as sklm

__author__ = 'Kuanshi Zhong'

class GlobalLinearRegression:
    
    def __init__(self,modelname='GLM',data=[],modeltype='OLS',modelpara=[]):
        """
        __init__: initialization
        - Input:
            modelname: model name
            data: training data
            modeltype: model type ('OLS' or 'ElasticNet')
            modelpara: [alpha,l1_ratio] (for 'ElasticNet')
        """
        # data check
        if data.size==0:
            print("Please check the input dimension.")
            return
        # randomize data
        np.random.shuffle(data)
        # initializing model
        self.X = data[:,0:-1]
        self.y = data[:,-1]
        self.modeltype = modeltype
        self.modelpara = modelpara
        # developing model
        self.__model_developement()
    
    def __model_developement(self):
        """
        __model_development: searching the optimal model given user inputs.
        """
        if self.modeltype=='OLS':
            self.model = sklm.LinearRegression().fit(self.X,self.y)
        elif self.modeltype=='ElasticNet':
            self.model = sklm.ElasticNet(alpha=self.modelpara[0],
                                         l1_ratio=self.modelpara[1]).fit(
                                                 self.X,self.y,)
        else:
            print("Please check the model type ('OLS' or 'ElasticNet').")
            return
        
    def modeleval(self,x0=[],rflag=0):
        """
        modelevel: evaluating the model at given point.
        - Input: 
            x0: evaluated X locations
            rflag: return-value flag, 0 - median, 1 - standard deviation, 
            2 - R^2 factor
        - Output:
            y0: median
            sigma0: standard deviation
            R2: R^2
        """
        y0 = self.model.predict(x0)
        sigma0 = np.std(self.y-self.model.predict(self.X))
        R2 = self.model.score(self.X,self.y)
        if rflag==0:
            return y0
        elif rflag==1:
            return sigma0
        else:
            return R2
