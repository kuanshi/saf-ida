"""USGS NSHMP hazard and ground motion model"""

import os, json, requests
import numpy as np
from scipy import interpolate

__author__ = 'Kuanshi Zhong'

SITECLASS_MAP = {'A': '2000', 'B': '1150', 'B/C': '760', 'C': '537',
                 'C/D': '360', 'D': '259', 'D/E': '180'}
IMT = ['PGA', 'SA']
EDITION_MAP = {'COUS': ['E2008','E2014','E2014B'],
               'CEUS': ['E2008','E2014','E2014B'],
               'WUS': ['E2008','E2014','E2014B'],
               'AK': ['E2007']}
PERIOD_MAP = {'E2007': [0,0.1,0.2,0.3,0.5,1.0,2.0], 
              'E2008': [0,0.1,0.2,0.3,0.5,0.75,1.0,2.0,3.0], 
              'E2014': [0,0.1,0.2,0.3,0.5,1.0,2.0], 
              'E2014B': [0,0.1,0.2,0.3,0.5,0.75,1.0,2.0,3.0,4.0,5.0]}


class USGS_Hazard:

    def __init__(self, longitude, latitude, imt, siteclass, edition='E2014B',region='COUS'):
        """
        initialization
        - input
        -- longitude: site longitude (float)
        -- latitude: site latitude (float)
        -- imt: intensity measure type (string)
        -- siteclass: site class(string)
        """
        self.longitude = longitude
        self.latitude = latitude
        self.imt = imt
        self.siteclass = siteclass
        self.edition = edition
        self.region = region

    def _config(self):
        # check longitude
        if np.abs(self.longitude)>=360:
            print('USGS_Hazard._config_check: longtidue {} is not valid'.format(self.longitude))
            return 1
        # check latitude
        if np.abs(self.latitude)>=90:
            print('USGS_Hazard._config_check: latitude {} is not valid'.format(self.latitude))
            return 1
        # check intensity measure type
        self.period = 0
        if self.imt == 'PGA':
            pass
        elif self.imt.startswith('SA'):
            period_str = self.imt[2:]
            if len(period_str) < 1:
                print('USGS_Hazard._config_check: intensity measure {} is not valid'.format(self.imt))
                return 1
            else:
                self.period = float(period_str.replace('P','.'))
        else:
            print('USGS_Hazard._config_check: intensity measure {} is not valid'.format(self.imt))
            return 1
        # check site class
        if self.siteclass not in list(SITECLASS_MAP.keys()):
            print('USGS_Hazard._config_check: site class {} is not valid'.format(self.siteclass))
            return 1
        else:
            self.vs30 = SITECLASS_MAP.get(self.siteclass)
        # check region
        if self.region not in list(EDITION_MAP.keys()):
            print('USGS_Hazard._config_check: site region {} is not valid'.format(self.region))
            return 1
        # check edition
        if self.edition not in list(EDITION_MAP.get(self.region)):
            print('USGS_Hazard._config_check: edition {} is not valid'.format(self.edition))
            return 1
        # check period
        if self.period > np.max(PERIOD_MAP.get(self.edition)):
            print('USGS_Hazard._config_check: period {} exits the maximum period supported by {}'.format(self.period, self.edition))
            print('USGS_Hazard._config_check: the maximum suported period {} is used'.format(np.max(PERIOD_MAP.get(self.edition))))
            self.period = np.max(PERIOD_MAP.get(self.edition))
            self.imt = 'SA'+str(self.period).replace('.','P')
        # prepare job number and intensity levels
        avail_imt = ['PGA']+['SA'+str(x).replace('.','P') for x in PERIOD_MAP.get(self.edition)[1:]]
        if self.imt in avail_imt:
            self.list_imt = [self.imt]
            self.list_t = [self.period]
            self.nrun = 1
        else:
            tmp = next(x for x, val in enumerate(PERIOD_MAP.get(self.edition)) if val>self.period)
            self.list_imt = avail_imt[tmp-1:tmp+1]
            self.list_t = PERIOD_MAP.get(self.edition)[tmp-1:tmp+1]
            self.nrun = 2


class HazardCurve(USGS_Hazard):

    def __init__(self, longitude, latitude, imt, siteclass, edition='E2014B',region='COUS'):
        # inherit
        super().__init__(longitude, latitude, imt, siteclass, edition=edition, region=region)
        # configure
        if self._config():
            print('HazardCurve.__init__: configuration failed - please check inputs')
        else:
            print('HazardCurve: hazard curve configured')
        self.hazardcurve = {self.imt: [], 'Annual Frequency of Exceedence': dict()}

    def _config(self):
        # inherit
        super()._config()
        # prepare url request
        self.url = []
        for cur_imt in self.list_imt:
            self.url.append('https://earthquake.usgs.gov/nshmp-haz-ws/hazard/{}/{}/{}/{}/{}/{}'.format(self.edition, \
                self.region,self.longitude,self.latitude,cur_imt,self.vs30))
        # return
        return 0

    def fetch_data(self):
        """
        fetch data via prepared url
        """
        if len(self.url) < 1:
            print('HazardCurve.fetch_data: no url is found')
            return 0
        # list of im and afe
        im = []
        afe = []
        # fetch individual
        for cur_url in self.url:
            cur_req = requests.get(cur_url)
            cur_res = cur_req.json()
            # check status
            if cur_res.get('status')=='error':
                print('HazardCurve.fetch_data: {}'.format(cur_res.get('message')))
                return 1
            im.append(cur_res['response'][0]['metadata']['xvalues'])
            tmp = cur_res['response'][0]['data']
            afe_values = dict()
            for cur_y in tmp:
                afe_values.update({cur_y.get('component'): cur_y.get('yvalues')})
            afe.append(afe_values)
        # interpolation if multiple
        if self.nrun == 1:
            self.hazardcurve = {self.list_imt[0]: im[0],
                                'Annual Frequency of Exceedence': afe[0]}
        else:
            # log-log linear interp for 2 periods
            comp_list = list(afe[0].keys())
            afe_values = dict()
            for cur_comp in comp_list:
                y1 = afe[0].get(cur_comp)
                y2 = afe[1].get(cur_comp)
                f = interpolate.interp1d(np.array(self.list_t), np.array([y1,y2]).T)
                y = f(self.period)
                afe_values.update({cur_comp: y.tolist()})
            self.hazardcurve = {self.imt: im[0],
                                'Annual Frequency of Exceedence': afe_values}
        # return
        print('HazardCurve: hazard curve obtained')
        return 0
        

class HazardDisagg(USGS_Hazard):

    def __init__(self, longitude, latitude, imt, siteclass, returnperiod, edition='E2014B',region='COUS'):
        # inherit
        super().__init__(longitude, latitude, imt, siteclass, edition=edition, region=region)
        self.returnperiod = returnperiod
        # configure
        if self._config():
            print('HazardDisagg.__init__: configuration failed - please check inputs')
        else:
            print('HazardDisagg: hazard disaggregation configured')

    def _config(self):
        # inherit
        super()._config()
        # prepare url request
        self.url = []
        for cur_imt in self.list_imt:
            self.url.append('https://earthquake.usgs.gov/nshmp-haz-ws/deagg/{}/{}/{}/{}/{}/{}/{}'.format(self.edition, \
                self.region,self.longitude,self.latitude,cur_imt,self.vs30,self.returnperiod))
        # return
        return 0

    def fetch_data(self):
        """
        fetch data via prepared url
        """
        if len(self.url) < 1:
            print('HazardDisagg.fetch_data: no url is found')
            return 0
        # list of im and afe
        disagg = []
        # fetch individual
        for i, cur_url in enumerate(self.url):
            cur_req = requests.get(cur_url)
            cur_res = cur_req.json()
            # check status
            if cur_res.get('status')=='error':
                print('HazardDisagg.fetch_data: {}'.format(cur_res.get('message')))
                return 1
            tmp = cur_res['response'][0]['data']
            disagg_values = dict()
            for cur_comp in tmp:
                cur_data = cur_comp.get('summary')
                im_value = cur_data[0]['data'][2]['value']
                magnitude_mean = cur_data[3]['data'][0]['value']
                distance_mean = cur_data[3]['data'][1]['value']
                epsilon_mean = cur_data[3]['data'][2]['value']
                tmp_dict = {self.list_imt[i]: im_value, 'Magnitude': magnitude_mean,
                            'Distance': distance_mean, 'epsilon': epsilon_mean}
                disagg_values.update({cur_comp.get('component'): tmp_dict})
            disagg.append(disagg_values)
        # interpolation if multiple
        if self.nrun == 1:
            self.hazarddisagg = disagg[0]
        else:
            # log-log linear interp for 2 periods
            comp_list = list(disagg[0].keys())
            disagg_values = dict()
            item_list = list(disagg[0].get(comp_list[0]).keys())
            print(item_list)
            item_list.pop(item_list.index(self.list_imt[0]))
            for cur_comp in comp_list:
                tmp = dict()
                # IM value
                y1 = disagg[0].get(cur_comp).get(self.list_imt[0])
                y2 = disagg[1].get(cur_comp).get(self.list_imt[1])
                f = interpolate.interp1d(np.array(self.list_t), np.array([y1,y2]))
                y = f(self.period)
                tmp.update({self.imt: y.tolist()})    
                for cur_item in item_list:
                    # Magnitude
                    y1 = disagg[0].get(cur_comp).get(cur_item)
                    y2 = disagg[1].get(cur_comp).get(cur_item)
                    f = interpolate.interp1d(np.array(self.list_t), np.array([y1,y2]))
                    y = f(self.period)
                    tmp.update({cur_item: y.tolist()})
                disagg_values.update({cur_comp: tmp})
            self.hazarddisagg = disagg_values
        # return
        print('HazardDisagg: hazard disaggregation completed')
        return 0


class USGS_GMM:

    def __init__(self, gmm, magnitude, vs30, imt='PGA', periods=None, rrup=None, vsinf=None, rjb=None, rx=None, 
                 dip=None, width=None, ztop=None, zhyp=None, z1p0=None, z2p5=None, rake=None):
        self.imt = imt
        self.periods = periods
        self.gmm = gmm
        self.magnitude = magnitude
        self.vs30 = vs30
        self.vsinf = vsinf
        self.rrup = rrup
        self.rjb = rjb
        self.rx = rx
        self.dip = dip
        self.width = width
        self.ztop = ztop
        self.zhyp = zhyp
        self.z1p0 = z1p0
        self.z2p5 = z2p5
        self.rake = rake
        # load usgs_gmm_spectra_config (available ranges of parameters)
        gmm_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'usgs_gmm_spectra_config.json')
        try:
            with open(gmm_config_file) as f:
                tmp = json.load(f)
            self.usgs_gmm_config = tmp.get('parameters')
        except:
            self.usgs_gmm_config = dict()
            print('USGS_GMM.__init__: configuration failed - configuration file {} not found'.format(gmm_config_file))
            return
        # configure
        if self._config():
            print('USGS_GMM.__init__: configuration failed - please check inputs')
            return
        else:
            print('USGS_GMM: GMM configured')

    def _config(self):

        # check magnitude
        if self.magnitude < 0 or self.magnitude > 9.7:
            print('USGS_GMM._config: magnitude {} is out of valid range [0, 9.7]'.format(self.magnitude))
            return 1
        # check distance
        if not any([self.rrup, self.rjb, self.rx]):
            print('USGS_GMM._config: please specify at least one distance')
            return 1
        for cur_r in [self.rrup, self.rjb, self.rx]:
            if cur_r is not None:
                if cur_r < 0 or cur_r > 1000:
                    print('USGS_GMM._config: distance {} is out of valid range [0km, 1000km]'.format(cur_r))
                    return 1
        # check vs30
        if self.vs30 < 150 or self.vs30 > 2000:
            print('USGS_GMM._config: vs30 {} is out of valid range [150m/s, 2000m/s]'.format(self.vs30))
            return 1
        # check dip
        if self.dip is not None:
            if self.dip < 0 or self.dip > 90:
                print('USGS_GMM._config: dip {} is out of valid range [0, 90]'.format(self.dip))
                return 1
        # check width
        if self.width is not None:
            if self.width < 0 or self.width > 60:
                print('USGS_GMM._config: width {} is out of valid range [0km, 60km]'.format(self.width))
                return 1
        # check ztop
        if self.ztop is not None:
            if self.ztop < 0 or self.ztop > 700:
                print('USGS_GMM._config: ztop {} is out of valid range [0km, 700km]'.format(self.ztop))
                return 1
        # check zhyp
        if self.zhyp is not None:
            if self.zhyp < 0 or self.zhyp > 700:
                print('USGS_GMM._config: zhyp {} is out of valid range [0km, 700km]'.format(self.zhyp))
                return 1
        # check z1p0
        if self.z1p0 is not None:
            if self.z1p0 < 0 or self.z1p0 > 5:
                print('USGS_GMM._config: z1p0 {} is out of valid range [0km, 5km]'.format(self.z1p0))
                return 1
        # check z2p5
        if self.z2p5 is not None:
            if self.z2p5 < 0 or self.z2p5 > 10:
                print('USGS_GMM._config: z2p5 {} is out of valid range [0km, 10km]'.format(self.z2p5))
                return 1
        # check imt
        if self.imt not in ['PGA','SA']:
            print('USGS_GMM._config: intensity measure {} is not valid'.format(self.imt))
            return 1
        # check gmm
        gmm_avail = self.usgs_gmm_config.get('gmm').get('values')
        gmm_param = dict()
        self.gmm_id = None
        for cur_gmm in gmm_avail:
            if self.gmm == cur_gmm.get('label'):
                gmm_param = cur_gmm
                self.gmm_id = cur_gmm.get('id')
                break
        if len(gmm_param) == 0:
            print('USGS_GMM._config: {} is not valid'.format(self.gmm))
            return 1
        # check gmm-specific maximum period
        Tmax = float(gmm_param.get('supportedImts')[-1][2:].replace('P','.'))
        if self.periods is not None and np.max(self.periods) > Tmax:
            print('USGS_GMM._config: periods requested exceeding maximum available {}s by {}'.format(Tmax, self.gmm))
            return 1
        # prepare url request
        for cur_imt in self.imt:
            if self.imt == 'SA':
                self.url = 'https://earthquake.usgs.gov/nshmp-haz-ws/gmm/spectra?gmm={}&mw={}'.format(self.gmm_id,self.magnitude)
                if self.rjb is not None:
                    self.url = self.url + '&rjb={}'.format(self.rjb)
                if self.rrup is not None:
                    self.url = self.url + '&rrup={}'.format(self.rrup)
                if self.rx is not None:
                    self.url = self.url + '&rx={}'.format(self.rx)
                if self.dip is not None:
                    self.url = self.url + '&dip={}'.format(self.dip)
                if self.width is not None:
                    self.url = self.url + '&width={}'.format(self.width)
                if self.ztop is not None:
                    self.url = self.url + '&ztop={}'.format(self.ztop)
                if self.zhyp is not None:
                    self.url = self.url + '&zhyp={}'.format(self.zhyp)
                if self.rake is not None:
                    self.url = self.url + '&rake={}'.format(self.rake)
                if self.vs30 is not None:
                    self.url = self.url + '&vs30={}'.format(self.vs30)
                if self.vsinf is not None:
                    self.url = self.url + '&vsinf={}'.format(str(bool(self.vsinf)).lower())
                if self.z2p5 is not None:
                    self.url = self.url + '&z2p5={}'.format(self.z2p5)
                if self.z1p0 is not None:
                    self.url = self.url + '&z1p0={}'.format(self.z1p0)
            else:
                for cur_r in [self.rrup, self.rjb, self.rx]:
                    if cur_r is not None:
                        break
                rMin = np.min(cur_r)
                rMax = np.max(cur_r)
                self.url = 'https://earthquake.usgs.gov/nshmp-haz-ws/gmm/distance?&gmm={}&imt={}&Mw={}&rMin={}&rMax={}'.format(self.gmm_id,self.imt,self.magnitude,rMin,rMax)

        # return
        return 0

    def fetch_data(self):

        if len(self.url) < 1:
            print('USGS_GMM.fetch_data: no url is found')
            return 0
        self.im = dict()
        # fetch url
        cur_req = requests.get(self.url)
        cur_res = cur_req.json()
        # check status
        if cur_res.get('status')=='error':
            print('USGS_GMM.fetch_data: {}'.format(cur_res.get('message')))
            return 1
        # medians
        medians = cur_res.get('means')
        periods = medians.get('data')[0].get('data').get('xs')
        m_sa = medians.get('data')[0].get('data').get('ys')
        if self.periods is not None:
            f = interpolate.interp1d(np.log(periods), np.log(m_sa), fill_value=(np.log(m_sa[0]),np.log(m_sa[-1])))
            y = np.exp(f(np.log(self.periods)))
            im_median = y.tolist()
        else:
            im_median = m_sa
        # sigmas
        sigmas = cur_res.get('sigmas')
        s_sa = sigmas.get('data')[0].get('data').get('ys')
        if self.periods is not None:
            f = interpolate.interp1d(np.log(periods), np.log(s_sa), fill_value=(np.log(s_sa[0]),np.log(s_sa[-1])))
            y = np.exp(f(np.log(self.periods)))
            im_dispersion = y.tolist()
        else:
            im_dispersion = s_sa
        self.im.update({'Median': im_median, 'Dispersion': im_dispersion})
