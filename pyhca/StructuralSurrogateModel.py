"""Surrogate models for structural performance metrics"""

import numpy as np
import json
from scipy import stats as spst
from matplotlib import pyplot as plt
from .GlobalLinearModel import GlobalLinearRegression
from .LocalLinearModel import LocalLinearRegression


__author__ = 'Kuanshi Zhong'

class SurrogateModel:

    def __init__(self,idadatafile=[],gmdatafile=[]):
        """
        __init__: initialization
        - Input:
            idadatafile: filename of IDA results
            gmdatafile: filename of the nested ground motion set data
        """
        self.idadatafile = idadatafile
        self.gmdatafile = gmdatafile
        # raw IDA results
        self.idadata = {}
        # ground motion data
        self.gmdata = {}
        # laoding data
        self.__load_data()
        # range of EDP variables
        self.rangeEDP = {}
        # Sa values exceeding different EDP levels
        self.SaEDP = {}
        # optimal lambda for collapse
        self.lambda_col_opt = []
        # collapse model
        self.col_model = []
        # EDP model
        self.edp_model = {}

    def __load_data(self):
        """
        __loadata: loading and storing ida and site data
        """
        print("Loading structural and ground motion data.")
        # IDA data
        with open(self.idadatafile) as f:
            tmpdata = json.load(f)
        self.dataid = tmpdata['Data ID']
        self.nameEDP = tmpdata['EDP name']
        self.nEDP = len(self.nameEDP)
        temp_nameGM = tmpdata['Ground motion name']
        self.nGM = len(temp_nameGM)
        for gmtag in temp_nameGM:
            self.idadata[gmtag] = tmpdata[gmtag]
        # Ground motion data
        if len(self.gmdatafile):
            with open(self.gmdatafile) as f:
                self.gmdata = json.load(f)
            # Sort the IDA data order in case it does not match the order
            # in gmdata
            idadata_sorted = dict()
            for tar_gmname in self.gmdata['Ground motion name']:
                idadata_sorted.update({tar_gmname: self.idadata[tar_gmname]})
            self.idadata = idadata_sorted
            self.nameGM = self.gmdata['Ground motion name']
            # computing SaRatio
            if 'SaRatio' in dict.keys(self.gmdata):
                self.__compute_saratio()
                # initializing optimal SaRatio period ranges
                self.optTra = {}
                self.optTrb = {}
        print("Data loaded.")

    def __compute_saratio(self):
        """
        __compute_saratio: computing SaRatio
        """
        print("Computing SaRatio.")
        # conditioning on T1
        self.T1 = self.gmdata['Conditional T1 (s)']
        # lower-bound period
        self.Tra = np.linspace(0.05,0.95,19)
        # upper-bound period
        self.Trb = np.linspace(1.05,3.00,40)
        # grid
        self.gTra,self.gTrb = np.meshgrid(self.Tra,self.Trb)
        # vector
        self.vTra = self.gTra.reshape([-1,1])
        self.vTrb = self.gTrb.reshape([-1,1])
        # PSA
        tmpT = np.array(self.gmdata['Spectral period (s)'])
        tmppsa = np.array(self.gmdata['Response spectra (g)'])
        tmpsaratio = []
        counttag = 0
        for tra in self.vTra:
            tmpTtag = np.intersect1d(np.where(tmpT>=np.round(tra*self.T1/0.01)*0.01),
                                     np.where(tmpT<=np.round(self.vTrb[counttag]*self.T1/0.01)*0.01))
            tmpvalue = np.divide(tmppsa[:,tmpT==self.T1].reshape(1,-1),
                      spst.gmean(tmppsa[:,tmpTtag],axis=1).reshape(1,-1))
            tmpsaratio.append(tmpvalue)
            counttag = counttag+1
        self.saratio_pool = np.array(tmpsaratio)
        print("SaRatio computed.")

    def get_collapse_im(self,cim='Sa (g)',cedp='SDRmax',climit=0.1):
        """
        get_collapse_im: collecting collapse intensity measures
        """
        print("Processing collapse data.")
        self.imcol = np.zeros((self.nGM,1))
        for gmtag in self.nameGM:
            tmptag = self.nameGM.index(gmtag)
            tmpim = np.array(self.idadata[gmtag][cim])
            tmpedp = np.array(self.idadata[gmtag][cedp])
            loctag = np.max(np.where(tmpedp<=climit))
            if loctag==np.size(self.idadata[gmtag][cim])-1:
                self.imcol[tmptag,0] = tmpim[loctag]
            else:
                self.imcol[tmptag,0] = np.interp(climit,
                          tmpedp[loctag:loctag+1],tmpim[loctag:loctag+1])
        self.imcol_median_raw = spst.gmean(self.imcol)
        self.imcol_std_raw = np.std(np.log(self.imcol), ddof=1)
        print("Collapse data processed.")
        print("Median collapse "+cim+" = "+str(self.imcol_median_raw))

    def get_edp_im(self,edpim='Sa (g)',**kwargs):
        """
        get_edp_im: computing intensity levels exceeding different EDP values
        - Input:
            edpim: the conditioning intensity measure
            *kwarg: EDPkeyword=[lb,ub,ndiv], e.g., SDR=[0.001,0.1,20]
        """
        print("Computing "+edpim+" for different EDPs.")
        # first initializing ranges of EDP values based on IDA data
        self.__get_edp_range()
        # updating user-defined ranges
        for key, value in kwargs.items():
            self.__get_edp_range(edpkw=key,lim=value)
        # loop over EDP
        for edptag in self.nameEDP:
            self.SaEDP[edptag] = {}
            for gmtag in self.nameGM:
                tmpsa = np.array(self.idadata[gmtag][edpim])[
                        np.argsort(self.idadata[gmtag][edptag])]
                tmpedp = np.sort(self.idadata[gmtag][edptag])
                tmpedp = tmpedp+1.0e-2*min(tmpedp)*np.random.rand(len(tmpedp))
                tmpsa = tmpsa.tolist()
                tmpsa.insert(0,0.0)
                tmpedp = tmpedp.tolist()
                tmpedp.insert(0,0.0)
                # interpolation with upper limits censored
                self.SaEDP[edptag][gmtag] = np.interp(
                        self.rangeEDP[edptag]['Range'].tolist(),
                        tmpedp,tmpsa,right=max(tmpsa))
        print(edpim+" computed.")

    def __get_edp_range(self,edpkw=[],lim=[]):
        """
        __get_edp_range: computing the range of EDP values from IDA data
        """
        if len(edpkw):
            for edptag in self.nameEDP:
                if edpkw in edptag:
                    self.rangeEDP[edptag]['Lower bound'] = lim[0]
                    self.rangeEDP[edptag]['Upper bound'] = min(lim[1],
                                 self.rangeEDP[edptag]['Upper bound'])
                    self.rangeEDP[edptag]['Number of divisions'] = lim[2]
                    self.rangeEDP[edptag]['Range'] = np.exp(np.linspace(
                            np.log(self.rangeEDP[edptag]['Lower bound']),
                            np.log(self.rangeEDP[edptag]['Upper bound']),
                            lim[2]))
        else:
            # default number of divisions
            self.ndiv = 20
            for edptag in self.nameEDP:
                tmpLB = float('inf')
                tmpUB = -float('inf')
                self.rangeEDP[edptag] = {}
                for gmtag in self.nameGM:
                    tmpLB = min(tmpLB,min(self.idadata[gmtag][edptag]))
                    tmpUB = max(tmpUB,max(self.idadata[gmtag][edptag]))
                self.rangeEDP[edptag]['Lower bound'] = tmpLB
                self.rangeEDP[edptag]['Upper bound'] = tmpUB
                self.rangeEDP[edptag]['Number of divisions'] = self.ndiv
                self.rangeEDP[edptag]['Range'] = np.exp(np.linspace(
                        np.log(tmpLB),np.log(tmpUB),self.ndiv))

    def compute_collapse_model(self,modeltag='LLM',
                           modelcoef=['Gaussian',['CV',5],[0.5,2],50]):
        """
        compute_collapse_model: searching the surrogate model
        with the optimal SaRatio
        - Input:
            modeltag: 'LLM' (default) - local linear model,
            'OLS' - global linear model with the ordinary least square method
            'ElasticNet' - global linear model with the elastic net method
            modelcoef: 'LLM' needs four - kernel type, selection method,
            [lambda_lowerbound,lambda_upperbound], and lambda division number;
            'OLS' does not require any; and 'ElasticNet' needs two - alpha
            and l1_ratio.
        """
        print("Computing collapse model.")
        # initializing the default parameters for ElasticNet
        if modeltag=='ElasticNet':
            modelcoef = [1.0,0.5]
        elif modeltag=='OLS':
            modelcoef = []
        else:
            pass
        # searching the optimal period of SaRatio
        if 'SaRatio' in self.gmdata['Key IM']:
            tmp_kim = self.gmdata['Key IM']
            tmperr = []
            tmpoptlambda = []
            counttag = 0
            for tra in self.vTra:
                tmpX = np.log(np.column_stack(
                        (self.saratio_pool[counttag].reshape((-1,1)),
                               np.array(self.gmdata[tmp_kim[
                                       tmp_kim!='SaRatio']]).reshape((-1,1)))))
                if modeltag=='LLM':
                    tmpmodel = LocalLinearRegression(
                            modelname='LLM',data=np.column_stack(
                                    (tmpX,np.log(self.imcol))),
                            kerneltype=modelcoef[0],
                            modelselection=modelcoef[1],
                            lambdabound=modelcoef[2],ndiv=modelcoef[3])
                    # using the CV mse as the error
                    tmperr.append(tmpmodel.mse)
                    tmpoptlambda.append(tmpmodel.lambda_opt)
                else:
                    tmpmodel = GlobalLinearRegression(
                            modelname='GLM',data=np.column_stack(
                                    (tmpX,np.log(self.imcol))),
                                    modeltype=modeltag,modelpara=modelcoef)
                    # using -R^2 as error to be minimized
                    tmperr.append(-tmpmodel.modeleval(tmpX,rflag=2))
                counttag = counttag+1
            # find min error
            opttag = np.argmin(tmperr)
            self.col_model_err = tmperr
            self.lambda_col_opt = tmpoptlambda
            # optimal period range
            self.optTra['Collapse'] = self.vTra[opttag]
            self.optTrb['Collapse'] = self.vTrb[opttag]
            # collapse model
            tmpX = np.log(np.column_stack(
                    (self.saratio_pool[opttag].reshape((-1,1)),
                     np.array(self.gmdata[tmp_kim[
                             tmp_kim!='SaRatio']]).reshape((-1,1)))))
            if modeltag=='LLM':
                self.col_model = LocalLinearRegression(
                        modelname='LLM',data=np.column_stack(
                                    (tmpX,np.log(self.imcol))),
                                kerneltype=modelcoef[0],
                                modelselection=modelcoef[1],
                                lambdabound=modelcoef[2],ndiv=modelcoef[3])
            else:
                self.col_model = GlobalLinearRegression(
                        modelname='GLM',data=np.column_stack(
                                    (tmpX,np.log(self.imcol))),
                        modeltype=modeltag,modelpara=modelcoef)
        else:
            tmp_kim = self.gmdata['Key IM']
            if modeltag=='LLM':
                self.col_model = LocalLinearRegression(
                        modelname='LLM',data=np.column_stack(
                        (np.log([self.gmdata[tmp_kim]]).reshape((-1,2)),
                        np.log(self.imcol))),kerneltype=modelcoef[0],
                        modelselection=modelcoef[1],lambdabound=modelcoef[2],
                        ndiv=modelcoef[3])
            else:
                self.col_model = GlobalLinearRegression(
                        modelname='GLM',data=np.column_stack(
                                (np.log(self.gmdata[tmp_kim]),
                                np.log(self.imcol))),
                                modeltype=modeltag,modelpara=modelcoef)
        print("Collapse model computed.")

    def compute_edp_model(self,modeltag='OLS',modelcoef=[]):
        """
        compute_edp_model: searching the surrogate model
        with the optimal SaRatio for different EDP
        - Input (similar to "get_collapse_model":
            modeltag: 'LLM' (default) - local linear model,
            'OLS' - global linear model with the ordinary least square method
            'ElasticNet' - global linear model with the elastic net method
            modelcoef: 'LLM' needs four - kernel type, selection method,
            [lambda_lowerbound,lambda_upperbound], and lambda division number;
            'OLS' does not require any; and 'ElasticNet' needs two - alpha
            and l1_ratio.
        """
        print("Computing EDP models.")
        # initializing the default parameters for ElasticNet
        if modeltag=='ElasticNet':
            modelcoef = [1.0,0.5]
        elif modeltag=='LLM':
            modelcoef = ['Gaussian',['CV',5],[0.5,2],50]
        else:
            pass
        # searching the optimal period of SaRatio
        if 'SaRatio' in self.gmdata['Key IM']:
            tmp_kim = self.gmdata['Key IM']

            # loop over all EDP variables
            for tagedp in self.nameEDP:
                self.edp_model[tagedp] = {'optTra':[], 'optTrb':[],
                              'model': []}
                tmpdiv = self.rangeEDP[tagedp]['Number of divisions']
                # loop over all levels
                for taglevel in range(0,tmpdiv):
                    # collect Sa values for taglevel
                    tmpy = []
                    tmpSaEDP = self.SaEDP[tagedp]
                    for taggm in tmpSaEDP.keys():
                        tmpy.append(tmpSaEDP[taggm][taglevel])
                    tmpy = np.log(tmpy).reshape((-1,1))
                    tmperr = []
                    tmpoptlambda = []
                    counttag = 0
                    # loop over all period ranges
                    for tra in self.vTra:
                        tmpX = np.log(np.column_stack(
                                (self.saratio_pool[counttag].reshape((-1,1)),
                                 np.array(self.gmdata[tmp_kim[
                                         tmp_kim!='SaRatio']]).reshape((-1,1)))))
                        if modeltag=='LLM':
                            pass
                        else:
                            tmpmodel = GlobalLinearRegression(
                                    modelname='GLM',data=np.column_stack((tmpX,tmpy)),
                                            modeltype=modeltag,modelpara=modelcoef)
                            # using -R^2 as error to be minimized
                            tmperr.append(-tmpmodel.modeleval(tmpX,rflag=2))
                        counttag = counttag+1
                    # find min error
                    opttag = np.argmin(tmperr)
                    self.edp_model_err = tmperr
                    self.lambda_epd_opt = tmpoptlambda
                    # optimal period range
                    self.edp_model[tagedp]['optTra'].append(self.vTra[opttag])
                    self.edp_model[tagedp]['optTrb'].append(self.vTrb[opttag])
                    # EDP model
                    tmpX = np.log(np.column_stack(
                            (self.saratio_pool[opttag].reshape((-1,1)),
                             np.array(self.gmdata[tmp_kim[
                                     tmp_kim!='SaRatio']]).reshape((-1,1)))))
                    if modeltag=='LLM':
                        pass
                    else:
                        self.edp_model[tagedp]['model'].append(GlobalLinearRegression(
                                modelname='GLM',data=np.column_stack((tmpX,tmpy)),
                                        modeltype=modeltag,modelpara=modelcoef))
        else:
            tmp_kim = self.gmdata['Key IM']
            if modeltag=='LLM':
                pass
            else:
                # loop over all EDP variables
                for tagedp in self.nameEDP:
                    tmpdiv = self.rangeEDP[tagedp]['Number of divisions']
                    # loop over all levels
                    for taglevel in range(0,tmpdiv):
                        # collect Sa values for taglevel
                        tmpy = []
                        tmpSaEDP = self.SaEDP[tagedp]
                        for taggm in tmpSaEDP.keys():
                            tmpy.append(tmpSaEDP[taggm][taglevel])
                            tmpy = np.log(tmpy).reshape((-1,1))
                            self.edp_model[tagedp]['model'] = GlobalLinearRegression(
                                    modelname='GLM',data=np.column_stack((
                                            np.log(self.gmdata[tmp_kim]),tmpy)),
                                            modeltype=modeltag,modelpara=modelcoef)
        print("EDP models computed.")

    def plot_raw_collapse(self):
        """
        plot_raw_collapse: plot the raw collapse Sa versus supplemental IMs
        - Input: none
        - Output: 1: success, 2: error
        """
        print('Plotting raw collapse Sa versus IMs.')
        if self.col_model:
            for i in range(np.shape(self.col_model.X)[1]):
                curfig = plt.figure(i)
                curax = curfig.gca()
                x = np.exp(self.col_model.X[:, i])
                y = np.exp(self.col_model.y)
                curax.plot(x,y,linestyle='None',marker='o', \
                           markerfacecolor='k',markeredgecolor='k')
                curax.grid()
                plt.xlabel(self.gmdata['Key IM'][i])
                plt.ylabel('Collapse Sa (g)')
                plt.title('Collapse Sa vs. '+self.gmdata['Key IM'][i])
                plt.show()
        else:
            print('No collapse models were found.')
            return 0
