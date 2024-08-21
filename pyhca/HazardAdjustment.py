"""Site specific hazard adjustment"""

import numpy as np
from scipy import stats as spst
from scipy import optimize as spop
from matplotlib import pyplot as plt

__author__ = 'Kuanshi Zhong'

class SiteAdjustment:
    
    def __init__(self,surrogate=[],site=[]):
        """
        __init__: initialization
        """
        # collect data from the surrage model
        # key intensity measures
        self.key_im = surrogate.gmdata['Key IM']
        # conditional period
        self.T1 = surrogate.T1
        # performance models
        # collapse
        if surrogate.col_model:
            # raw IDA collapse Sa data
            self.imcol_raw = surrogate.imcol
            self.imcol_median_raw = surrogate.imcol_median_raw
            self.imcol_std_raw = surrogate.imcol_std_raw
            # surrogate collapse model
            self.col_model = surrogate.col_model
            if 'SaRatio' in self.key_im:
                self.optTra_col = surrogate.optTra['Collapse']
                self.optTrb_col = surrogate.optTrb['Collapse']
            print("Collapse model received.")
        else:
            # raw IDA collapse Sa data
            self.imcol_raw = []
            self.imcol_median_raw = []
            self.imcol_std_raw = []
            # surrogate collapse model
            self.col_model = []
            print("No collapse model received.")
        # EDP variables
        if len(surrogate.edp_model):
            self.nameEDP = surrogate.nameEDP
            self.nEDP = surrogate.nEDP
            self.rangeEDP = surrogate.rangeEDP
            self.edp_model = surrogate.edp_model
            print("EDP models received.")
        else:
            self.nameEDP = []
            self.edp_model = {}
            print("No EDP model received.")
        # site
        self.site = site
        # general conditional intensity mesasure
        self.gcim = {}
        # return period
        self.RP = []
        # GCIM log mean & covariance matrix
        self.mean_sacol = {}
        self.mean_subimcol = {}
        self.sigma_subimcol = {}
        self.mean_saedp = {}
        self.mean_subimedp = {}
        self.sigma_subimedp = {}
        self.__compute_gcim()
        self.ssp_pool_set = []
        self.ssp = {}
        
    def __compute_gcim(self):
        """
        __compute_gcim: computing the general conditional IM targets
        """
        print("Computing GCIM targets.")
        # collecting intensity measures
        for tagsite in self.site.nameCase:
            self.gcim[tagsite] = {}
            tmpsite = self.site.SiteCase[tagsite]
            self.RP = tmpsite['Return period (yr)']
            self.nRP = len(self.RP)
            for tagim in self.key_im:
                if 'SaRatio' in tagim:
                    for tmpkey in tmpsite.keys():
                        # Sa at the conditioning period
                        if 'Sa(T1)' in tmpkey:
                            self.gcim[tagsite]['SaT1'] = np.array(tmpsite[tmpkey])
                        # conditional mean spectrum
                        elif 'PSA' in tmpkey:
                            self.gcim[tagsite]['PSA'] = np.array(tmpsite[tmpkey])
                            self.gcim[tagsite]['T'] = np.array(tmpsite['Spectral period (s)'])
                        # End if 'Sa(T1)'
                    # End for tmpkey
                else:
                    for tmpkey in tmpsite.keys():
                        if tagim in tmpkey:
                            # other measures
                            self.gcim[tagsite][tagim] = np.array(tmpsite[tmpkey])
                        # End if tagim
                    # End for tmpkey
                # End if 'SaRatio'
            # End for tagim
            # covariance
            self.gcim[tagsite]['COV'] = np.array(tmpsite['Covariance'])
        # End for tagsite
         # computing GCIM log mean & covariance matrix
        if 'SaRatio' in self.key_im:
            for tagsite in self.site.nameCase:
                tmpT = self.gcim[tagsite]['T']
                self.gcim[tagsite]['SaRatio'] = []
                # collapse model
                if self.col_model:
                    tmpTtag = np.intersect1d(np.where(tmpT>=np.round(
                            self.optTra_col*self.T1/0.01)*0.01),np.where(
                            tmpT<=np.round(self.optTrb_col*self.T1/0.01)*0.01))
                    numT = len(tmpTtag)
                    A = np.row_stack((np.column_stack((np.ones((1,numT))*(-1.0/numT),0.0)),
                                      np.column_stack((np.zeros((1,numT)),1.0))))
                    counttag = 0
                    self.mean_sacol[tagsite] = []
                    self.mean_subimcol[tagsite] = {}
                    self.sigma_subimcol[tagsite] = {}
                    for tagRP in self.RP:
                        tmpSa = self.gcim[tagsite]['SaT1'][counttag]
                        self.mean_sacol[tagsite].append(np.log(tmpSa))
                        b = np.row_stack((np.log(tmpSa),0.0))
                        tmpM = np.log(np.row_stack((
                                self.gcim[tagsite]['PSA'][counttag,tmpTtag].reshape((-1,1)),
                                self.gcim[tagsite]['Ds575'][counttag])))
                        self.mean_subimcol[tagsite][tagRP] = A.dot(tmpM)+b
                        tmpTag = np.column_stack((tmpTtag.reshape((1,-1)),len(tmpT)))
                        tmpS = self.gcim[tagsite]['COV'][counttag]
                        tmpS = tmpS[:,tmpTag]
                        tmpS = tmpS[tmpTag,:].reshape((len(tmpTtag)+1,len(tmpTtag)+1))
                        self.sigma_subimcol[tagsite][tagRP] = A.dot(tmpS).dot(A.transpose())
                        counttag = counttag+1
                    # End for tagRP
                # End if len(self.col_model)
                # EDP models
                if len(self.edp_model):
                    self.mean_saedp[tagsite] = {}
                    self.mean_subimedp[tagsite] = {}
                    self.sigma_subimedp[tagsite] = {}
                    # loop over EDP
                    for tagedp in self.nameEDP:
                        tmpdiv = self.rangeEDP[tagedp]['Number of divisions']
                        self.mean_saedp[tagsite][tagedp] = {}
                        self.mean_subimedp[tagsite][tagedp] = {}
                        self.sigma_subimedp[tagsite][tagedp] = {}
                        # loop over all levels
                        for taglevel in range(0,tmpdiv):
                            tmpTra = self.edp_model[tagedp]['optTra'][taglevel]
                            tmpTrb = self.edp_model[tagedp]['optTrb'][taglevel]
                            tmpTtag = np.intersect1d(np.where(tmpT>=np.round(
                                    tmpTra*self.T1/0.01)*0.01),np.where(tmpT<=np.round(
                                            tmpTrb*self.T1/0.01)*0.01))
                            numT = len(tmpTtag)
                            A = np.row_stack((np.column_stack((
                                    np.ones((1,numT))*(-1.0/numT),0.0)),np.column_stack((
                                           np.zeros((1,numT)),1.0))))
                            counttag = 0
                            self.mean_saedp[tagsite][tagedp][taglevel] = []
                            self.mean_subimedp[tagsite][tagedp][taglevel] = {}
                            self.sigma_subimedp[tagsite][tagedp][taglevel] = {}
                            for tagRP in self.RP:
                                tmpSa = self.gcim[tagsite]['SaT1'][counttag]
                                self.mean_saedp[tagsite][tagedp][taglevel].append(np.log(tmpSa))
                                b = np.row_stack((np.log(tmpSa),1))
                                tmpM = np.log(np.row_stack((
                                        self.gcim[tagsite]['PSA'][counttag,tmpTtag].reshape((-1,1)),
                                        self.gcim[tagsite]['Ds575'][counttag])))
                                self.mean_subimedp[tagsite][tagedp][taglevel][tagRP] = A.dot(tmpM)+b
                                tmpTag = np.column_stack((tmpTtag.reshape((1,-1)),len(tmpT)))
                                tmpS = self.gcim[tagsite]['COV'][counttag]
                                tmpS = tmpS[:,tmpTag]
                                tmpS = tmpS[tmpTag,:].reshape((len(tmpTtag)+1,len(tmpTtag)+1))
                                self.sigma_subimedp[tagsite][tagedp][taglevel][tagRP] = \
                                A.dot(tmpS).dot(A.transpose())
                                counttag = counttag+1
                            # End for tagRP
                        # End for taglevel
                    # End for tagedp
                # End if len(self.edp_model)
            # End for tagsite
        else:
            # this part can be extended (KZ)
            pass
        print("GCIM targets computed.")
        
    def site_specific_performance(self,setname=[('All','All')],rflag=0):
        """
        site_specific_performance: computing site-specific responses
        - Input: 
            setname - (the case tag, the response variable tag)
            rflag - rewrite flag (0: no rewriting, 1: overwriting old data)
            The default value is 'All' including all existing attributes
        """
        tmppool = []
        for tagcase in setname:
            if tagcase == ('All','All'):
                if len(self.ssp_pool_set)==0 or rflag:
                    for tmp1 in self.site.nameCase:
                        for tmp2 in self.nameEDP:
                            tmppool.append((tmp1,tmp2))
                            # removing old data
                            if (tmp1,tmp2) in self.ssp_pool_set:
                                self.ssp_pool_set.remove((tmp1,tmp2))
                        tmp2 = 'Collapse'
                        tmppool.append((tmp1,tmp2))
                        # removing old data
                        if (tmp1,tmp2) in self.ssp_pool_set:
                            self.ssp_pool_set.remove((tmp1,tmp2))
                else:
                    print("Please use rflag=1 for overwriting data.")
            # if tagcase already exists
            elif tagcase in self.ssp_pool_set:
                # checking rewrite flag
                if rflag:
                    tmppool.append(tagcase)
                    # remove the old data
                    self.ssp_pool_set.remove(tagcase)
                else:
                    print("Case existed: "+tagcase+", please use rflag=1 for overwriting.")
                    return
            elif tagcase[0] in self.site.nameCase and \
            (tagcase[1] in self.nameEDP or tagcase[1]=='Collapse'):
                tmppool.append(tagcase)
            else:
                print("Case not found: "+tagcase[0]+".")
                return
        for tagcase in tmppool:
            self.ssp_pool_set.append(tagcase)
            
        # computing site-specific responses
        for tagcase in self.ssp_pool_set:
            self.ssp[tagcase] = {}
            # collapse
            if tagcase[1] == 'Collapse':
                self.ssp[tagcase]['Collapse'] = {}
                print("Adjusting collapse fragility for Case: "+tagcase[0])
                # collecting Sa and sub-IM log mean and covariance
                tmpMsa = self.mean_sacol[tagcase[0]]
                tmpMsubim = self.mean_subimcol[tagcase[0]]
                tmpSsubim = self.sigma_subimcol[tagcase[0]]
                # loop over range
                rptag = 0
                prob = []
                for tagRP in self.RP:
                    range_im = []
                    dm = []
                    counttag = 0
                    # loop over intensity generating IM ranges
                    for tagIM in self.key_im:
                        range_im.append(np.linspace(
                                tmpMsubim[tagRP][counttag]- \
                                3*np.sqrt(tmpSsubim[tagRP][counttag,counttag]),
                                tmpMsubim[tagRP][counttag]+ \
                                3*np.sqrt(tmpSsubim[tagRP][counttag,counttag]),
                                50))
                        dm.append(np.mean(np.diff(range_im[-1].transpose())))
                        counttag = counttag+1
                    # generating IM grid
                    tmp_grid = np.meshgrid(*range_im)
                    v_im = []
                    counttag = 0
                    # coverting to vectorized sub-IM
                    for tagIM in self.key_im:
                        v_im.append(tmp_grid[counttag].reshape((-1,1)))
                        counttag = counttag+1
                    v_im = np.array(v_im).transpose().reshape((-1,2))
                    # computing probability mass function
                    pmf = spst.multivariate_normal.pdf(v_im,mean=tmpMsubim[tagRP].transpose()[0],
                                                       cov=tmpSsubim[tagRP])*np.prod(dm)
                    # computing cond. log mean and standard deviation of collapse
                    theta_sa = self.col_model.modeleval(x0=v_im,rflag=0)
                    beta_sa = self.col_model.modeleval(x0=self.col_model.X,rflag=1)
                    # computing exceeding probability
                    eprob = spst.norm.cdf(tmpMsa[rptag],
                                         loc=theta_sa,scale=beta_sa)
                    # estimating collapse probability
                    prob.append(pmf.reshape((1,-1)).dot(eprob.reshape((-1,1))))
                    rptag = rptag+1
                prob = np.array(prob).transpose()[0]
                # collecting estimated probability
                self.ssp[tagcase]['Collapse']['Sa (g)'] = np.exp(tmpMsa).tolist()
                self.ssp[tagcase]['Collapse']['Est. Pcol'] = prob.tolist()
                # MLE for log mean and standard dev. of collapse fragility
                self.ssp[tagcase]['Collapse']['Fragility'] = self.__mle_normpara(
                        np.array(tmpMsa).flatten(),prob).tolist()
                print("Adjusted median collapse Sa (g): "+ \
                      str(np.exp(self.ssp[tagcase]['Collapse']['Fragility'][0])))
                print("Adjusted dispersion of collapse Sa: "+ \
                      str(self.ssp[tagcase]['Collapse']['Fragility'][1]))
            else:
                print("Adjusting EDP: "+tagcase[1]+" for "+tagcase[0])
                # calling EDP
                tagedp = tagcase[1]
                self.ssp[tagcase][tagedp] = {}
                nRange = self.rangeEDP[tagedp]['Number of divisions']
                # initializing exceeding probability
                eprob = np.ones((self.nRP,nRange))
                # loop over all ranges
                for tagR in range(0,nRange):
                    tmpMsa = self.mean_saedp[tagcase[0]][tagedp][tagR]
                    tmpMsubim = self.mean_subimedp[tagcase[0]][tagedp][tagR]
                    tmpSsubim = self.sigma_subimedp[tagcase[0]][tagedp][tagR]
                    # loop over return periods
                    countRP = 0
                    for tagRP in self.RP:
                        range_im = []
                        dm = []
                        counttag = 0
                        # loop over intensity measures
                        for tagIM in self.key_im:
                            range_im.append(np.linspace(
                                    tmpMsubim[tagRP][counttag]- \
                                    3*np.sqrt(tmpSsubim[tagRP][counttag,counttag]),
                                    tmpMsubim[tagRP][counttag]+ \
                                    3*np.sqrt(tmpSsubim[tagRP][counttag,counttag]),
                                    50))
                            dm.append(np.mean(np.diff(range_im[-1].transpose())))
                            counttag = counttag+1
                        # generating IM grid
                        tmp_grid = np.meshgrid(*range_im)
                        v_im = []
                        counttag = 0
                        # coverting to vectorized sub-IM
                        for tagIM in self.key_im:
                            v_im.append(tmp_grid[counttag].reshape((-1,1)))
                            counttag = counttag+1
                        v_im = np.array(v_im).transpose().reshape((-1,2))
                        # computing probability mass function
                        pmf = spst.multivariate_normal.pdf(
                                v_im,mean=tmpMsubim[tagRP].transpose()[0],
                                cov=tmpSsubim[tagRP])*np.prod(dm)
                        # computing cond. log mean and standard deviation of collapse
                        theta_sa = self.edp_model[tagedp]['model'][tagR].modeleval(
                                x0=v_im,rflag=0)
                        beta_sa = self.edp_model[tagedp]['model'][tagR].modeleval(
                                x0=self.edp_model[tagedp]['model'][tagR].X,rflag=1)
                        tmp_prob = spst.norm.cdf(tmpMsa[countRP],
                                                 loc=theta_sa,scale=beta_sa)
                        eprob[countRP,tagR] = pmf.reshape((1,-1)).dot(
                                tmp_prob.reshape((-1,1)))
                        countRP = countRP+1
                    # End for tagRP                    
                # End for tagR
                # computing conditional CDF of EDP
                cond_CDF = np.ones((1,nRange))
                countRP = 0
                for tagRP in self.RP:
                    # normalization
                    cond_CDF = (1.0-eprob[countRP,:])/(1.0-eprob[countRP,-1])
                    cond_CDF[np.min(np.where(cond_CDF>=1.0)):] = 1.0
                    cond_CDF[0] = 0.0
                    ## estimating log mean and standard deviation of EDP
                    self.ssp[tagcase][tagedp][tagRP] = self.__mle_normpara(
                            np.log(self.rangeEDP[tagedp]['Range']),cond_CDF).tolist()
                    print("Adjusted median of "+tagcase[1]+" at RP"+str(tagRP)+": "+ \
                          str(np.exp(self.ssp[tagcase][tagedp][tagRP][0])))
                    print("Adjusted std of "+tagcase[1]+" at RP"+str(tagRP)+": "+ \
                          str(self.ssp[tagcase][tagedp][tagRP][1]))
                    countRP = countRP+1
                # End for tagRP
            # if tagcase
        # End for tagcase                
    
    def plot_result(self,setname=[]):
        """
        plot_result: plotting adjusted collapse and response results
        - Input:
            setname: (the case tag, the response variable tag)
        - Output:
            N/A
        """
        cpalettes = ['k','b','g','y','r','m']
        numC = len(cpalettes)
        if setname[0] == ('All','All'):
            setname = self.ssp_pool_set
        tag_fig = 0
        for tagcase in setname:
            if tagcase in self.ssp_pool_set:
                # starting to plot
                if tagcase[1] == 'Collapse':
                    # collapse plot
                    curfig = plt.figure(tag_fig)
                    curax = curfig.gca()
                    # raw fragility
                    x = np.sort(self.imcol_raw,axis=0)
                    y = np.arange(len(x))/float(len(x))
                    curax.plot(x,y,linestyle='None',marker='o', \
                               markerfacecolor='k',markeredgecolor='k', \
                               label='Raw')
                    x = np.arange(0.001,2,0.001)
                    y = spst.norm.cdf(np.log(x), \
                                      loc=np.log(self.imcol_median_raw), \
                                      scale=self.imcol_std_raw)
                    curax.plot(x,y,linestyle='-',color='k', \
                               label='Raw fragility')
                    cur_mean = self.ssp[tagcase]['Collapse']['Fragility'][0]
                    cur_std = self.ssp[tagcase]['Collapse']['Fragility'][1]
                    y = spst.norm.cdf(np.log(x),loc=cur_mean,scale=cur_std)
                    curax.plot(x,y,linestyle='-',color='b', \
                               label='Adjusted fragility')
                    curax.legend()
                    curax.grid()
                    plt.ylim(0,1)
                    plt.xlim(0,2)
                    plt.xlabel('Sa (g)')
                    plt.ylabel('Collapse probability')
                    plt.title(tagcase[0]+' '+tagcase[1])
                else:
                    # story response plot
                    curfig = plt.figure(tag_fig)
                    curax = curfig.gca()
                    LB = self.rangeEDP[tagcase[1]]['Lower bound']
                    UB = self.rangeEDP[tagcase[1]]['Upper bound']
                    x = np.arange(LB,UB,0.01*(UB-LB))
                    # loop over RP
                    ctag = 0
                    for tagRP in self.RP:
                        cur_mean = self.ssp[tagcase][tagcase[1]][tagRP][0]
                        cur_std = self.ssp[tagcase][tagcase[1]][tagRP][1]
                        y = spst.norm.pdf(np.log(x),loc=cur_mean,scale=cur_std)
                        curax.plot(x,y,linestyle='-', \
                                   color=cpalettes[np.mod(ctag,numC)], \
                                   label='Return period = '+str(tagRP))
                        ctag = ctag+1
                    curax.legend()
                    curax.grid()
                    plt.xlabel(tagcase[1])
                    plt.ylabel('PDF')
                    plt.title(tagcase[0]+' '+tagcase[1])
                tag_fig = tag_fig+1
            else:
                plt.show()
                print("Can't find setname = "+tagcase[0]+' '+tagcase[1]+".")
                return
            plt.show()
            
    def __mle_normpara(self,x,y):
        """
        __mle_para: estimating parameters for normal distribution using MLE
        - Input:
            x: variable values
            y: cumulative probability values
        - Output:
            optsol: [mu,sigma] where,
            mu: mean of the normal dist.
            sigma: standard deviation of the normal dist.
        """
        # initialzing optimal solution
        optsol = []
        # initial guess
        if np.max(y)>=0.5:
            # interpolating to get the first guess
            mu0 = np.interp(0.5,y.flatten(),x.flatten())
        else:
            # using the maximum as the first guess
            mu0 = np.max(x)
        # standard deviation starts with 0.2
        sigma0 = 0.2
        # convergence flag for optimization
        conv_flag = 0
        while not conv_flag and sigma0 < 1.0:
            x0 = [mu0,sigma0]
            res = spop.minimize(self.__loglik,x0,args=(x,y),
                                method='Nelder-Mead',options={'maxiter': 10000})
            conv_flag = res.success
            optsol = res.x
            sigma0 = sigma0+0.05
        return optsol
        
    def __loglik(self,t,xx,yy):
        """ 
        __loglik: computing the loglik value (negative)
        - Input:
            t: trial value
            xx: variable values
            yy: cumulative probability values
        - Output:
            loglik: negative log likelihood
        """
        # big sampling number
        bignum = 1000
        num_yy = np.around(bignum*yy).reshape((-1,1))
        # estimating cumulative probabaility values given t
        p = spst.norm.cdf(xx,loc=t[0],scale=t[1]).reshape((-1,1))
        # computing log likelihood value
        loglik = -np.sum(spst.binom.logpmf(num_yy,bignum,p))
        return loglik
