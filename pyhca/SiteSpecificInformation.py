"""Site specific information"""

__author__ = 'Kuanshi Zhong'

import json, bisect, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import numpy as np
import USGSHazardGMM, SignificantDurationModel, CorrelationModel

LOCAL_IM_GMPE = {"DS575": ["Bommer, Stafford & Alarcon (2009)", "Afshari & Stewart (2016)"],
                 "DS595": ["Bommer, Stafford & Alarcon (2009)", "Afshari & Stewart (2016)"],
				 "DS2080": ["Afshari & Stewart (2016)"]}

class SiteData:

    def __init__(self,longitude,latitude,siteclass,returnperiods,imt,cim,vs30=None):

        self.longitude = longitude
        self.latitude = latitude
        self.siteclass = siteclass
        self.returnperiods = returnperiods
        self.imt = imt
        if 'PSA' in list(cim.keys())[0] or 'SA' in list(cim.keys())[0]:
            # spectrum type, get the period
            Tcond = cim.get(list(cim.keys())[0]).get('Period',None)
            if Tcond is None:
                print('SiteData.__init__: warnning: no Period for conditional intenisty measure {}'.format(list(cim.keys())[0]))
                self.cond_im = None
                #return
            else:
                self.cond_im = 'SA'+str(Tcond).replace('.','P')
            # cross check
            if 'PSA' in list(imt.keys()):
                if Tcond not in imt.get('PSA').get('Periods'):
                    bisect.insort(imt['PSA']['Periods'], Tcond)
            elif 'SA' in list(imt.keys()):
                if Tcond not in imt.get('SA').get('Periods'):
                    bisect.insort(imt['SA']['Periods'], Tcond)
            else:
                print('SiteData.__init__: please include conditional IM {} in intensity measure list'.format(list(cim.keys())[0]))
                return
        elif 'PGA' in list(cim.keys())[0]:
            self.cond_im = 'PGA'
            if 'PGA' not in imt:
                print('SiteData.__init__: please include conditional IM {} in intensity measure list'.format('PGA'))
                return
        else:
            print('SiteData.__init__: conditonal intensity measure {} is not supported yet'.format(list(cim.keys())[0]))
            return
        if vs30 is None:
            print(self.siteclass)
            self.vs30 = float(USGSHazardGMM.SITECLASS_MAP.get(self.siteclass))
        else:
            self.vs30 = vs30
        # set gmm
        self.set_gmm(imt)

    def set_hazard_disagg(self, edition='E2014B', region='COUS'):
        # set up hazard disaggregation
        self.disagg_info = []
        for cur_rp in self.returnperiods:
            cur_hazdisagg = USGSHazardGMM.HazardDisagg(self.longitude, self.latitude, self.cond_im, self.siteclass, cur_rp, edition=edition, region=region)
            self.disagg_info.append(cur_hazdisagg)

    def run_hazard_disagg(self):
        # run hazard disaggregation
        for i, cur_rp in enumerate(self.returnperiods):
            self.disagg_info[i].fetch_data()

    def set_gmm(self, im):
        # set up ground motion prediction equations
        self.gmm = dict()
        for cur_im, cur_prop in im.items():
            tmp = cur_prop.get('GMPE', None)
            if tmp is None:
                print('SiteData.set_gmm: cannot find GMPE for {}'.format(cur_im))
                return
            self.gmm.update({cur_im: tmp})

    def set_im_calculator(self):
        self.im_calculator = []
        self.im = []
        for cur_imt, cur_prop in self.imt.items():
            if cur_imt.startswith('PSA') or cur_imt.startswith('SA'):
                for curT in cur_prop.get('Periods'):
                    self.im.append('SA({})'.format(curT))
            elif cur_imt.startswith('DS'):
                self.im.append(cur_imt+'H')
        # loop over all return periods
        for i, cur_rp in enumerate(self.returnperiods):
            cur_hazdisagg = self.disagg_info[i].hazarddisagg
            cur_mag = cur_hazdisagg.get('Total').get('Magnitude')
            cur_dist = cur_hazdisagg.get('Total').get('Distance')
            # loop over all imt
            im_calculators = dict()
            self.im_calculator.append(im_calculators)
            for cur_imt, cur_prop in self.imt.items():
                cur_periods = cur_prop.get('Periods', None)
                cur_gmm = self.gmm.get(cur_imt, None)
                if cur_gmm is None:
                    print('SiteData.compute_im: cannot find the ground motion model for {}'.format(cur_imt))
                    return
                if cur_gmm == "Bommer, Stafford & Alarcon (2009)":
                    cur_calculator = SignificantDurationModel.bommer_stafford_alarcon_ds_2009
                elif cur_gmm == "Afshari & Stewart (2016)":
                    cur_calculator = SignificantDurationModel.afshari_stewart_ds_2016
                else:
                    cur_calculator = USGSHazardGMM.USGS_GMM(cur_gmm, cur_mag, self.vs30, imt=cur_imt, periods=cur_periods, rrup=cur_dist)
                # collect
                im_calculators.update({cur_imt: cur_calculator})

    def run_im_calculator(self):
        
        self.im_target = []
        # loop over all return periods
        for i, cur_rp in enumerate(self.returnperiods):
            # current hazard disagg
            cur_hazdisagg = self.disagg_info[i].hazarddisagg
            # current magnitude/distance
            cur_mag = cur_hazdisagg.get('Total').get('Magnitude')
            cur_dist = cur_hazdisagg.get('Total').get('Distance')
            # get current im_calculator
            im_calculators = self.im_calculator[i]
            # initialize mean and standard deviation list
            im_mean = []
            im_std = []
            # loop over all intensity measure types
            for cur_imt, cur_prop in self.imt.items():
                cur_calculator = im_calculators.get(cur_imt)
                cur_gmm = self.gmm.get(cur_imt, None)
                if cur_gmm == "Bommer, Stafford & Alarcon (2009)":
                    theta, beta_tot, beta_tau, beta_phi = cur_calculator(magnitude=cur_mag, distance=cur_dist, 
                                                                         vs30=self.vs30, duration_type=cur_imt+'H')
                    # collect
                    im_mean.append(theta)
                    im_std.append(beta_tot)
                elif cur_gmm == "Afshari & Stewart (2016)":
                    theta, beta_tot, beta_tau, beta_phi = cur_calculator(magnitude=cur_mag, distance=cur_dist, 
                                                                         vs30=self.vs30, duration_type=cur_imt+'H')
                    # collect
                    im_mean.append(theta)
                    im_std.append(beta_tot)
                else:
                    cur_calculator.fetch_data()
                    theta = np.log(cur_calculator.im['Median']).tolist()
                    beta_tot = cur_calculator.im['Dispersion']
                    # collect
                    im_mean = im_mean+theta
                    im_std = im_std+beta_tot

            # correlation coefficient
            im_corr = np.zeros((len(self.im),len(self.im)))
            for j, im1 in enumerate(self.im):
                for k, im2 in enumerate(self.im):
                    if j > k:
                        continue
                    im_corr[j,k] = CorrelationModel.baker_bradley_correlation_2017(im1=im1, im2=im2)
                    im_corr[k,j] = im_corr[j,k]

            # collect
            self.im_target.append({'ReturnPeriod': cur_rp, 'Median': np.exp(im_mean).tolist(), 
                                   'StandardDev': im_std, 'Correlation': im_corr.tolist()})


class SiteInfo:
    
    def __init__(self,dataname='SiteData',sitedatafile=None,siteconfigfile=None,site_data_dict=None):
        """
        __init__: initialization
        """
        self.nCase = 0
        self.nameCase = []
        self.SiteCase = {}
        self.sitedatafile = sitedatafile
        self.siteconfigfile = siteconfigfile
        if self.sitedatafile is not None:
            self.__load_data()
        elif self.siteconfigfile is not None:
            self.__site_config()
        elif site_data_dict is not None:
            self.__load_site_info(site_data_dict)
        else:
            print('SiteInfo.__init__: no site information is provided - please define sitedatafile or siteconfigfile')
            return
        
    def __load_data(self):
        """
        __loadata: loading site data
        """
        print("SiteInfo: loading site data")
        # Site data
        if len(self.sitedatafile):
            with open(self.sitedatafile) as f:
                data = json.load(f)
        self.nCase = data['Number of cases']
        self.nameCase = data['Case name']
        for tagcase in self.nameCase:
            self.SiteCase[tagcase] = data[tagcase]
        print("SiteInfo: site data loaded.")

    def __load_site_info(self,data):
        self.nCase = data['Number of cases']
        self.nameCase = data['Case name']
        for tagcase in self.nameCase:
            self.SiteCase[tagcase] = data[tagcase]
        print("SiteInfo: site data loaded.")

    def __site_config(self):

        print('SiteInfo: configuring site')
        # load config file
        try:
            with open(self.siteconfigfile) as f:
                site_config = json.load(f)
        except:
            print('SiteInfo.__site_config: cannot load {}'.format(self.siteconfigfile))
            return 1
        # site number
        self.nCase = site_config.get('SiteNumber',1)
        # site name
        self.nameCase = site_config.get('SiteName')
        # check site name and number
        if len(self.nameCase) != self.nCase:
            print('SiteInfo.__site_config: site names are not consistent with site number {}'.format(self.nCase))
            return 1
        # get site properties
        for cur_site_name in self.nameCase:
            cur_site = site_config.get(cur_site_name, None)
            if cur_site is None:
                print('SiteInfo.__site_config: skipping site {} as no properties provided'.format(cur_site_name))
                continue
            # longitude and latitude
            cur_lon = cur_site.get('Longitude')
            cur_lat = cur_site.get('Latitude')
            # return period
            cur_rp = cur_site.get('ReturnPeirod')
            # intensity measure types
            cur_im = cur_site.get('IntensityMeasure')
            cur_imt = list(cur_im.keys())
            # conditional intensity measure types
            cur_cim = cur_site.get('ConditionalIntensityMeasure')
            cur_cimt = list(cur_cim.keys())[0]
            # site class (optional)
            cur_siteclass = cur_site.get('SiteClass', 'B')


        return
        
    def add_case(self,sitedatafile=[]):
        """
        add_case: adding cases into the current site data
        """
        print("Adding case(s).")
        # Site data
        if len(self.sitedatafile):
            with open(self.sitedatafile) as f:
                data = json.load(f)
        for tagcase in data['Case name']:
            # checking any duplication
            if tagcase in self.nameCase:
                print("Case name already existed: "+tagcase+".")
                return
            else:
                self.nameCase.append(tagcase)
                self.SiteCase[tagcase] = data[tagcase]
                self.nCase = self.nCase+1
                print("Case: "+tagcase+" added.")
        
    def remove_case(self,casename=[]):
        """
        remove_case: removing cases from the current site data
        """
        print("Removing case(s).")
        # Site data
        for tagcase in casename:
            # checking any duplication
            if tagcase in self.nameCase:
                self.nameCase.remove(tagcase)
                del self.SiteCase[tagcase]
                self.nCase = self.nCase-1
                print("Case: "+tagcase+" removed.")
            else:
                print("Case does not exist: "+tagcase)
                return
