"""Nested ground motion selection"""

import os as os
import numpy as np
import json as json

__author__ = 'Kuanshi Zhong'

class NestedGroundMotionSet:

    def __init__(self,*argv):
        # ground motion set name
        if len(argv):
            self.name = argv
        else:
            self.name = 'NGMS'
        # ground motion name list
        self.gmname = []
        # ground motion tag
        self.gmtag = []
        # response spectra
        self.psa = []
        # ground motion IM
        self.gmim = []
        # scaling factor
        self.gmsf = []
        # error
        self.imerr = []
        # ground motion acceleration history data points
        self.acc = []
        # number of acceleration data points
        self.npts = []
        # ground motion sampling rate
        self.dt = []
        # ground motion response spectra
        # scaling limit flag
        self.sf_flag = 0
        # database flag
        self.gmdb_flag = 0
        # grid flag
        self.grid_flag = 0
        """ 
        Note these are identified possible key intensity measures.
        More user-defined measures can be added with individual functions.
        """
        self.key_im = ['SaRatio','Ds575','Ds595',
                       'PGA','PGV','PGD','Ia']
    
    def definekeyim(self,dim,mesh,**kwargs):
        """
        definekeyim: defining key intensity measures that form a multi-
        dimension grid.
        - Input:
            dim: dimension of the grid, a scalar
            mesh: number of grids for each dimension, a vector
            kwargs: 'intensity measure name' = [lowerbound, upperbound] 
        """
        self.dim = dim
        self.mesh = np.array(mesh)
        self.name_im = []
        self.bound_im = np.float64(np.array([[0]*2]*dim))
        tmptag = 0
        # Reading inputs
        for name_im, range_data in kwargs.items():
            if name_im in self.key_im:
                self.name_im.append(name_im)
                if name_im == 'SaRatio':
                    # SaRatio(Ta,T1,Tb) needs 5 inputs
                    self.T1 = range_data[0]
                    # lower-bound T
                    self.Ta = range_data[1]*range_data[0]
                    # upper-bound T
                    self.Tb = range_data[2]*range_data[0]
                    # range of SaRatio
                    self.bound_im[tmptag,:] = range_data[3:5]
                else:
                    # others need 2 inputs
                    self.bound_im[tmptag,:] = range_data
                tmptag = tmptag+1
            else:
                print("Please check the intensity measure name with ",
                      self.key_im)
                return
    
    def generategrid(self,**kwargs):
        """
        generategrid: generating the IM grid
        - Input:
            kwargs: different sampling method, i.e., 'orthogonal (default)', 
            'random'=nGM, and 'LHS'=nGM.
            Other methods can be added.
        """
        
        if len(kwargs) == 0:
            self.range_im = []
            for tag_dim in range(0,self.dim):
                self.range_im.append(np.exp(np.linspace(
                        np.log(self.bound_im[tag_dim,0]), 
                        np.log(self.bound_im[tag_dim,1]),
                        self.mesh[tag_dim])))
            self.grid_im = np.meshgrid(*self.range_im)
            # number of nested ground motion in total
            self.nGM = np.prod(self.mesh)
        elif 'random' in kwargs:
            for tmp1,tmp2 in kwargs.items:
                self.nGM = tmp2
            loc_rand = np.random.rand(self.nGM,self.dim)
            self.grid_im = [[0]*self.nGM]*self.dim
            for tag_dim in range(0,self.dim):
                self.grid_im[:][tag_dim] = (self.bound_im[tag_dim][1]-
                            self.bound_im[tag_dim][0])*loc_rand[:][tag_dim]
        else:
           for tmp1,tmp2 in kwargs.items:
                self.nGM = tmp2
           """
           To be developed.
           """
        self.grid_flag = 1
    
    def scalinglimit(self,sf_min,sf_max,sf_penalty,sf_t,sf_target):
        """
        scalinglimit: defining the scaling factor limitation and penalty.
        - Input:
            sf_min: minimum scaling factor
            sf_max: maximum scaling factor
            sf_penalty: per unit outside the scaling range
            sf_t: the reference period
            sf_target: reference Sa(sf_t)
        Note this ScalingLimit is optional
        """
        self.sf_min = sf_min
        self.sf_max = sf_max
        self.sf_penalty = sf_penalty
        self.sf_t = sf_t
        self.sf_target = sf_target
        self.sf_flag = 1
        
    def __loadgroundmotiondata(self,filename):
        """
        loadgroundmotiondata: loading ground motion data for selection.
        - Input:
            filename: the full directory to the json data.
        Note please follow the format as the example.
        """
        print("Loading ground motion database.")
        with open(filename) as f:
            tmpdata = json.load(f)
        self.gmdb_nGM = tmpdata['numgm']
        self.gmdb_period = np.array(tmpdata['psa_period'])
        self.gmdb_name = tmpdata['name']
        self.gmdb_dt = np.array(tmpdata['dt'])
        self.gmdb_pga = np.array(tmpdata['PGA'])
        self.gmdb_pgv = np.array(tmpdata['PGV'])
        self.gmdb_pgd = np.array(tmpdata['PGD'])
        self.gmdb_ds575 = np.array(tmpdata['Ds575'])
        self.gmdb_ds595 = np.array(tmpdata['Ds595'])
        self.gmdb_ia = np.array(tmpdata['Ia'])
        self.gmdb_psa = np.array(tmpdata['PSA'])
        self.gmdb_flag = 1
        print("Ground motion database loaded.")
        
    def __computesaratio(self):
        """
        computesaratio: computing SaRatio for the ground motion database.
        """
        print("Computing SaRatio.")
        if self.gmdb_flag:
            tagTr = np.intersect1d(np.where(self.gmdb_period>=np.round(self.Ta,2)),
                             np.where(self.gmdb_period<=np.round(self.Tb,2)))
            saavg = np.reshape(np.exp(np.mean(np.log(self.gmdb_psa[:,tagTr]),
                                              axis=1)),[-1,1])
            sat1 = self.gmdb_psa[:,self.gmdb_period==np.round(self.T1,2)]
            self.gmdb_saratio = sat1/saavg
            print("SaRatio computed.")
        else:
            print("Please first .LoadGroundMotionData(filename).")
            return
    
    def __computeerror(self):
        """
        computeerror: computing error to the target grid.
        """
        print("Processing data.")
        # compute IM values of the database
        self.__groundmotiondataim()
        # scaling limits
        self.sf_loss = np.zeros((1,self.gmdb_nGM))
        tmpt = np.where(self.gmdb_period==np.round(self.sf_t,2))
        self.gmdb_sf = self.sf_target/np.reshape(self.gmdb_psa[:,tmpt],[1,-1])
        if self.sf_flag:
            ptag1 = np.where(self.gmdb_psa[:,tmpt]<(self.sf_target/self.sf_max))
            self.sf_loss[0,ptag1] = np.power(self.sf_max- \
                   self.sf_target/self.gmdb_psa[ptag1,tmpt],4)*self.sf_penalty
            ptag2 = np.where(self.gmdb_psa[:,tmpt]>(self.sf_target/self.sf_min))
            self.sf_loss[0,ptag2] = np.power(self.sf_min- \
                   self.sf_target/self.gmdb_psa[ptag2,tmpt],4)*self.sf_penalty
        # if the grid exists
        if self.grid_flag:
            self.errmat = np.zeros((self.nGM,self.gmdb_nGM))
            # loop over IM
            for tag_dim in range(0,self.dim):
                tmpgrid = np.reshape(self.grid_im[tag_dim],[1,-1])
                tmpim = np.reshape(self.gmdb_im[:,tag_dim],[1,-1])
                # loop over seat
                for tag_seat in range(0,self.nGM):
                    if tag_dim == 0:
                        self.errmat[tag_seat,:] = self.errmat[tag_seat,:]+ \
                        np.square(np.log(tmpim)- \
                                  np.log(tmpgrid[0,tag_seat]))+self.sf_loss
                    else:
                        self.errmat[tag_seat,:] = self.errmat[tag_seat,:]+ \
                        np.square(np.log(tmpim)- \
                                  np.log(tmpgrid[0,tag_seat]))
            print("Data processed.")
        else:
            print("Please first .generategrid(self,**kwargs).")
            return
                
    def __groundmotiondataim(self):
        """
        groundmotiondataim: computing key IM for the groudn motion database.
        """
        self.gmdb_im = np.zeros((self.gmdb_nGM,self.dim))
        tmptag = 0
        for name_im in self.name_im:
            if name_im == 'SaRatio':
                self.__computesaratio()
                self.gmdb_im[:,[tmptag]] = self.gmdb_saratio
            elif name_im == 'Ds575':
                self.gmdb_im[:,tmptag] = self.gmdb_ds575
            elif name_im == 'Ds595':
                self.gmdb_im[:,tmptag] = self.gmdb_ds595
            elif name_im == 'PGA':
                self.gmdb_im[:,tmptag] = self.gmdb_pga
            elif name_im == 'PGV':
                self.gmdb_im[:,tmptag] = self.gmdb_pgv
            elif name_im == 'PGD':
                self.gmdb_im[:,tmptag] = self.gmdb_pgd
            elif name_im == 'Ia':
                self.gmdb_im[:,tmptag] = self.gmdb_ia
            else:
                print("Please first add the following IM in the database ",
                      name_im,".")
            tmptag = tmptag+1            
    
    def selectnestedrecord(self,gmdb_path=[]):
        """
        selectnestedrecord: selecting neseted ground motion records.
        """
        # loading database
        if len(gmdb_path)==0:
            print("Please define the ground motion database directory.")
        else:
            self.__loadgroundmotiondata(gmdb_path)
            self.__computeerror()
            # starting selection
            availtag = [i for i in range(0,self.gmdb_nGM)]
            for gmseat in range(0,self.nGM):
                self.gmtag.append(availtag[
                        np.argmin(self.errmat[gmseat,availtag])])
                availtag.remove(self.gmtag[-1])
                self.imerr.append(self.errmat[gmseat,self.gmtag[-1]])
                self.gmname.append(self.gmdb_name[self.gmtag[-1]])
                self.dt.append(self.gmdb_dt[self.gmtag[-1]])
                self.gmim.append(self.gmdb_im[self.gmtag[-1],:])
                self.psa.append(self.gmdb_psa[self.gmtag[-1]])
                self.gmsf.append(self.gmdb_sf[0,self.gmtag[-1]])
        print("Nested ground motion set selected.")
    
    def savedata(self,output_path=[]):
        """
        savedata: saving the information of selected nested ground motion.
        """
        if len(output_path)==0:
            output_path = os.getcwd()
        # collecting variables
        metadata = {}
        metadata['Conditional T1 (s)'] = self.T1
        metadata['Ground motion name'] = self.gmname
        metadata['Sampling rate (s)'] = self.dt
        metadata['Key IM'] = self.name_im
        tmptag = 0
        tmpim = np.array(self.gmim)
        for name_im in self.name_im:
            metadata[name_im] = np.array(tmpim[:,tmptag]).tolist()
            tmptag = tmptag+1
        metadata['Spectral period (s)'] = np.array(self.gmdb_period).tolist()
        metadata['Response spectra(g)'] = np.array(self.psa).tolist()
        if self.sf_flag:
            metadata['Scaling factor'] = np.array(self.gmsf).tolist()
        with open(output_path+'/'+self.name+'.json','w') as outfile:
            json.dump(metadata,outfile,indent=4)
        print("Data saved.")
                          