# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Leland Stanford Junior University
# Copyright (c) 2024 The Regents of the University of California
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Kuanshi Zhong
#

import argparse, json, os, copy
from general import *
from pyngms import NestedGroundMotionSelection as NGMS
from pyhca import SiteSpecificInformation as SSInfo
from pyhca import StructuralSurrogateModel as SSM
from pyhca import HazardAdjustment as HA

class SAF_IDA:

    def __init__(self, dir_info = dict(), job_name = 'saf_ida'):

        # initiate a log file
        self.input_dir = dir_info.get('Input', './')
        self.output_dir = dir_info.get('Output', './')
        self.logfile_name = job_name+'.log'
        self.logfile = Logfile(logfile_dir=self.output_dir, logfile_name=self.logfile_name)

        # initiate a database
        self.db_name = job_name+'.h5'
        self.dbserver = DBServer(db_dir=self.output_dir, db_name=self.db_name)

        # initiate a nested ground motion set
        self.groundmotions = NGMS.NestedGroundMotionSet(job_name=job_name)

    def _parse_gms_config(self, gms_config):

        # initialize intensity measure grid
        self.groundmotions.definekeyim(0, [])

        # intensity measure grid
        im_grid_config = gms_config.get('IntensityMeasureGrid', None)
        if im_grid_config is None:
            err_msg = 'SAF_IDA._parse_gms_config: IntensityMeasureGrid not found in configuration.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # im grid dimension
        im_grid_dim = len(im_grid_config)
        if im_grid_dim == 0:
            err_msg = 'SAF_IDA._parse_gms_config: no intensity measure defined in IntensityMeasureGrid.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # im names
        im_names = list(im_grid_config.keys())
        for cur_im in im_names:
            im_info = im_grid_config.get(cur_im, None)
            if im_info is None:
                err_msg = 'SAF_IDA._parse_gms_config: empty in {}.'.format(cur_im)
                self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            # im grid mesh
            cur_mesh = im_info.get('Mesh', None)
            if cur_mesh is None:
                err_msg = 'SAF_IDA._parse_gms_config: IntensityMeasureGrid missing Mesh for {}.'.format(cur_im)
                self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            # get range
            cur_range = im_info.get('Range', None)
            if cur_range is None:
                err_msg = 'SAF_IDA._parse_gms_config: IntensityMeasureGrid missing Range for {}.'.format(cur_im)
                self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            # T1, Ta, and Tb
            if 'SaRatio' in cur_im:
                T1 = im_info.get('T1', None)
                Ta = im_info.get('Ta', None)
                Tb = im_info.get('Tb', None)
                if None in [T1, Ta, Tb]:
                    err_msg = 'SAF_IDA._parse_gms_config: SaRatio missing T1, Ta, Tb'
                    self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                    return 1
                # add im
                self.groundmotions.add_key_im(name=cur_im, vrange=cur_range, mesh=cur_mesh, T1=T1, Ta=Ta, Tb=Tb)
            else:
                # add im
                self.groundmotions.add_key_im(name=cur_im, vrange=cur_range, mesh=cur_mesh)
        
        # scaling limit if any
        scaling_config = gms_config.get('Scaling', None)
        if scaling_config is not None:
            # minimum scaling
            sf_min = scaling_config.get('Min', 0.001)
            sf_max = scaling_config.get('Max', 1.0e6)
            sf_penalty = scaling_config.get('Penalty', 0.0)
            sf_t = scaling_config.get('ReferenceT', None)
            sf_sa = scaling_config.get('ReferenceSa', None)
            if None in [sf_t, sf_sa]:
                err_msg = 'SAF_IDA._parse_gms_config: Scaling missing ReferenceT, ReferenceSa'
                self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            self.groundmotions.scalinglimit(sf_min, sf_max, sf_penalty, sf_t, sf_sa)            

        # ground motion database
        self.gmdb_file = gms_config.get('GroundMotionData', 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyngms', 'data', 'GroundMotionCharacteristics.csv'))

        # return
        return 0


    def _select_records(self):

        try:
            # generate grid
            self.groundmotions.generategrid()
        except:
            err_msg = 'SAF_IDA._select_records: failed to generate IM grid'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        run_msg = 'SAF_IDA._select_records: IM grid generated'
        self.logfile.write_msg(msg=run_msg)

        try:
            # select records        
            self.groundmotions.selectnestedrecord(gmdb_path=self.gmdb_file)
        except:
            err_msg = 'SAF_IDA._select_records: failed to select ground motions'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        run_msg = 'SAF_IDA._select_records: ground motions selected'
        self.logfile.write_msg(msg=run_msg)

        try:
            # save data
            self.groundmotions.savedata(output_path=self.output_dir)
        except:
            err_msg = 'SAF_IDA._select_records: failed to save data'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        run_msg = 'SAF_IDA._select_records: data saved'
        self.logfile.write_msg(msg=run_msg)

        return 0
    
    def _parse_site_config(self, site_config = None):

        # site name
        self.site_name = site_config.get('SiteName','MySite')
        # longitude and latitude
        self.lon = site_config.get('Longitude',None)
        if self.lon is None:
            err_msg = 'SAF_IDA._parse_site_config: Longitude not found.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        self.lat = site_config.get('Latitude', None)
        if self.lat is None:
            err_msg = 'SAF_IDA._parse_site_config: Latitude not found.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # site class and vs30
        self.site_class = site_config.get('SiteClass','B/C')
        self.vs30 = site_config.get('Vs30',None)
        # return periods to investigate
        self.return_periods = site_config.get('ReturnPeriods',None)
        if self.return_periods is None:
            err_msg = 'SAF_IDA._parse_site_config: ReturnPeriods not found.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # intensity measure types
        self.imt = site_config.get('IntensityMeasureType',None)
        if self.imt is None:
            err_msg = 'SAF_IDA._parse_site_config: IntensityMeasureType not found.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # conditional intensity measure
        self.cim = site_config.get('ConditionalIntensityMeasure',None)
        if self.cim is None:
            self.cim = {
                'SA': {
                    'Period': None
                }
            }
        # seismic hazard deaggregation info
        self.shd_edition = site_config.get('DeaggEdition', 'E2014B')
        self.shd_region = site_config.get('DeaggRegion','COUS')
        # intensity measure target type
        self.im_target_type = site_config.get('IMTargetType','CS')
        # create the site data object
        self.site_data = SSInfo.SiteData(self.lon, self.lat, self.site_class, self.return_periods, 
                                         self.imt, self.cim, self.vs30)
        # set hazard deaggregation
        self.site_data.set_hazard_disagg(edition=self.shd_edition, region=self.shd_region)
        self.site_data.run_hazard_disagg()

        # set intensity measure calculation
        self.site_data.set_im_calculator(tgt_type=self.im_target_type)
        self.site_data.run_im_calculator()

        # prepare site data dictionary
        self.site_data_dict = {
            'Data ID': 'Site data',
            'Number of cases': 1,
            'Case name': [self.site_name],
            self.site_name: {
                'Coord.': [self.lon, self.lat],
                'Number of intensity levels': len(self.return_periods),
                'Target type': self.im_target_type,
                'Intensity Measures': list(self.imt.keys()),
                'T1 (s)': self.cim.get(list(self.cim.keys())[0]).get('Period'),
                'Spectral period (s)': self.imt.get('SA').get('Periods'),
                'Return period (yr)': self.return_periods,
                'Sa(T1) (g)': [],
                'PSA (g)': [],
                'DS575': [],
                'DS595': [],
                'Covariance': []
            }
        }
        IM_Conversion = {
            'DS575': 'DS575',
            'DS595': 'DS595',
            'Ds575': 'DS575',
            'Ds595': 'DS595'
        }
        for i,cur_rp in enumerate(self.return_periods):
            cur_im_target = self.site_data.im_target[i]
            im_idx = 0
            if self.im_target_type in ['CS','CSD']:
                for cur_im in self.imt.keys():
                    cur_im_list = []
                    if cur_im == 'SA':
                        num_T = len(self.imt.get('SA').get('Periods'))
                        self.site_data_dict[self.site_name]['PSA (g)'].append(cur_im_target.get('ConditionalMean')[im_idx:im_idx+num_T])
                        im_idx = im_idx+num_T
                        if self.cim['SA'].get('Period') is None:
                            pass
                        else:
                            T1_idx = self.imt.get('SA').get('Periods').index(self.cim['SA'].get('Period'))
                            self.site_data_dict[self.site_name]['Sa(T1) (g)'].append(cur_im_target.get('ConditionalMean')[T1_idx])
                    elif cur_im.startswith('DS') or cur_im.startswith('Ds'):
                        self.site_data_dict[self.site_name][IM_Conversion.get(cur_im)].append(cur_im_target.get('ConditionalMean')[im_idx])
                        im_idx = im_idx+1
                    else:
                        pass
                # covariance
                self.site_data_dict[self.site_name]['Covariance'].append(cur_im_target.get('ConditionalCov'))
            else:
                for cur_im in self.imt.keys():
                    cur_im_list = []
                    if cur_im == 'SA':
                        num_T = len(self.imt.get('SA').get('Periods'))
                        self.site_data_dict[self.site_name]['PSA (g)'].append(cur_im_target.get('Median')[im_idx:im_idx+num_T])
                        im_idx = im_idx+num_T
                        if self.cim['SA'].get('Period') is None:
                            pass
                        else:
                            T1_idx = self.imt.get('SA').get('Periods').index(self.cim['SA'].get('Period'))
                            self.site_data_dict[self.site_name]['Sa(T1) (g)'].append(cur_im_target.get('Median')[T1_idx])
                    elif cur_im.startswith('DS') or cur_im.startswith('Ds'):
                        self.site_data_dict[self.site_name][IM_Conversion.get(cur_im)].append(cur_im_target.get('Median')[im_idx])
                        im_idx = im_idx+1
                    else:
                        pass
                # covariance
                self.site_data_dict[self.site_name]['Covariance'].append((np.diag(cur_im_target.get('StandardDev')).dot(np.array(cur_im_target.get('Correlation'))).dot(np.diag(cur_im_target.get('StandardDev')))).tolist())
        
        # return
        return 0
    
    def _parse_user_hazard_config(self, tgt_config = None):

        # site name
        self.site_name = tgt_config.get('TargetName','MySite')
        # longitude and latitude
        self.lon = tgt_config.get('Longitude',None)
        self.lat = tgt_config.get('Latitude', None)
        # site class and vs30
        self.site_class = tgt_config.get('SiteClass',None)
        self.vs30 = tgt_config.get('Vs30',None)
        # return periods to investigate
        self.return_periods = tgt_config.get('ReturnPeriods',None)
        if self.return_periods is None:
            err_msg = 'SAF_IDA._parse_user_hazard_config: ReturnPeriods not found.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # intensity measure target type
        self.im_target_type = tgt_config.get('IMTargetType','CS')
        # input config
        self.user_hazard_config = tgt_config.get('InputConfig',None)
        if self.user_hazard_config is None:
            err_msg = 'SAF_IDA._parse_user_hazard_config: InputConfig not found.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        self.user_hazard_input_type = self.user_hazard_config.get('Type',None)
        if self.user_hazard_input_type is None:
            err_msg = 'SAF_IDA._parse_user_hazard_config: Type not found in InputConfig.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        if self.user_hazard_input_type == 'IntensityMeasureCSV':
            self.user_hazard_input_file = self.user_hazard_config.get('Filename',[])
            if len(self.user_hazard_input_file) != len(self.return_periods):
                err_msg = 'SAF_IDA._parse_user_hazard_config: Input file number does not match the return period number.'
                self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            csv_headers = self.user_hazard_config.get('Header',[])
            if len(csv_headers) == 0:
                err_msg = 'SAF_IDA._parse_user_hazard_config: Header not found in InputConfig.'
                self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            self.imt = dict()
            for cur_header in csv_headers:
                self.imt.update({
                    cur_header.get('IM'): {
                        "Periods": cur_header.get('Periods')
                    }
                })
            #print(self.imt)
        # conditional intensity measure
        self.cim = tgt_config.get('ConditionalIntensityMeasure',None)
        if self.cim is None:
            self.cim = {
                'SA': {
                    'Period': None
                }
            }
        # prepare site data dictionary
        self.site_data_dict = {
            'Data ID': 'Site data',
            'Number of cases': 1,
            'Case name': [self.site_name],
            self.site_name: {
                'Coord.': [self.lon, self.lat],
                'Number of intensity levels': len(self.return_periods),
                'Target type': self.im_target_type,
                'Intensity Measures': list(self.imt.keys()),
                'T1 (s)': self.cim.get(list(self.cim.keys())[0]).get('Period'),
                'Spectral period (s)': self.imt.get('SA').get('Periods'),
                'Return period (yr)': self.return_periods,
                'Sa(T1) (g)': [],
                'PSA (g)': [],
                'DS575': [],
                'DS595': [],
                'Covariance': []
            }
        }
        IM_Conversion = {
            'DS575': 'DS575',
            'DS595': 'DS595',
            'Ds575': 'DS575',
            'Ds595': 'DS595'
        }
        for i,cur_rp in enumerate(self.return_periods):
            cur_inputfile = os.path.join(self.input_dir,self.user_hazard_input_file[i])
            df_im_realizations = pd.read_csv(cur_inputfile,header=0)
            im_idx = 0
            for cur_im in self.imt.keys():
                if cur_im == 'SA':
                    num_T = len(self.imt.get('SA').get('Periods'))
                    self.site_data_dict[self.site_name]['PSA (g)'].append(list(np.exp(np.log(df_im_realizations.iloc[:,im_idx:im_idx+num_T]).mean(axis=0).tolist())))
                    im_idx = im_idx+num_T
                    if self.cim['SA'].get('Period') is None:
                        pass
                    else:
                        T1_idx = self.imt.get('SA').get('Periods').index(self.cim['SA'].get('Period'))
                        self.site_data_dict[self.site_name]['Sa(T1) (g)'].append(np.exp(np.log(df_im_realizations.iloc[:,T1_idx]).mean()))
                elif cur_im.startswith('DS') or cur_im.startswith('Ds'):
                    self.site_data_dict[self.site_name][IM_Conversion.get(cur_im)].append(np.exp(np.log(df_im_realizations.iloc[:,im_idx]).mean()))
                    im_idx = im_idx+1
                else:
                    pass
            # covariance
            self.site_data_dict[self.site_name]['Covariance'].append(np.cov(np.log(df_im_realizations).T).tolist())
        # return
        return 0
    
    def _parse_user_im_config(self, tgt_config = None):

        # site name
        self.site_name = tgt_config.get('TargetName','MySite')
        # longitude and latitude
        self.lon = tgt_config.get('Longitude',None)
        self.lat = tgt_config.get('Latitude', None)
        # site class and vs30
        self.site_class = tgt_config.get('SiteClass',None)
        self.vs30 = tgt_config.get('Vs30',None)
        # return periods to investigate
        self.return_periods = tgt_config.get('ReturnPeriods',None)
        if self.return_periods is None:
            num_return_period = 1
        else:
            num_return_period = len(self.return_periods)
        # intensity measure target type
        self.im_target_type = tgt_config.get('IMTargetType',None)
        if self.im_target_type is None:
            err_msg = 'SAF_IDA._parse_user_im_config: IMTargetType not found.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # input config
        self.user_im_config = tgt_config.get('InputConfig',None)
        if self.user_im_config is None:
            err_msg = 'SAF_IDA._parse_user_im_config: InputConfig not found.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        self.user_im_tgt_type = self.user_im_config.get('Type',None)
        if self.user_im_tgt_type is None:
            err_msg = 'SAF_IDA._parse_user_im_config: Type not found in InputConfig.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        if self.user_im_tgt_type == 'IntensityMeasureMeanLog':
            self.user_im_tgt = self.user_im_config.get('MeanLog',[])
            if len(self.user_im_tgt) == 0:
                err_msg = 'SAF_IDA._parse_user_im_config: Input MeanLog number is zero.'
                self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            self.imt = dict()
            for cur_imt in self.user_im_tgt:
                if len(cur_imt.get('Value')) != num_return_period:
                    err_msg = 'SAF_IDA._parse_user_im_config: Input MeanLog size does not match return period number.'
                    self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                    return 1
                self.imt.update({
                    cur_imt.get('IM'): {
                        "Value": cur_imt.get('Value')
                    }
                })
            self.user_im_std = [None for i in range(num_return_period)]
            self.user_im_corr = [None for i in range(num_return_period)]
        if self.user_im_tgt_type == 'IntensityMeasureDist':
            self.user_im_tgt = self.user_im_config.get('MeanLog',[])
            if len(self.user_im_tgt) == 0:
                err_msg = 'SAF_IDA._parse_user_im_config: Input MeanLog number is zero.'
                self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            self.imt = dict()
            for cur_imt in self.user_im_tgt:
                if len(cur_imt.get('Value')) != num_return_period:
                    err_msg = 'SAF_IDA._parse_user_im_config: Input MeanLog size does not match return period number.'
                    self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                    return 1
                self.imt.update({
                    cur_imt.get('IM'): {
                        "Value": cur_imt.get('Value')
                    }
                })
            self.user_im_std = self.user_im_config.get('StdLog',[])
            if len(self.user_im_std) != num_return_period:
                err_msg = 'SAF_IDA._parse_user_im_config: Input StdLog number does not match return period number.'
                self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            self.user_im_corr = self.user_im_config.get('CorrelationCoeffMatrix',[])
            if len(self.user_im_corr) != num_return_period:
                err_msg = 'SAF_IDA._parse_user_im_config: Input CorrelationCoeffMatrix number does not match return period number.'
                self.logfile.write_msg(msg=err_msg, msg_type='WARNING')
                self.user_im_corr = [None for i in range(num_return_period)]
        # conditional intensity measure
        self.cim = tgt_config.get('ConditionalIntensityMeasure',None)
        if self.cim is None:
            self.cim = {
                'SA': {
                    'Period': None
                }
            }
        # prepare site data dictionary
        self.site_data_dict = {
            'Data ID': 'Site data',
            'Number of cases': 1,
            'Case name': [self.site_name],
            self.site_name: {
                'Coord.': [self.lon, self.lat],
                'Number of intensity levels': num_return_period,
                'Target type': self.im_target_type,
                'Intensity Measures': list(self.imt.keys()),
                'T1 (s)': self.cim.get(list(self.cim.keys())[0]).get('Period'),
                'Return period (yr)': self.return_periods,
                'Sa(T1) (g)': [],
                'DS575': [],
                'DS595': [],
                'Covariance': []
            }
        }
        IM_Conversion = {
            'DS575': 'DS575',
            'DS595': 'DS595',
            'Ds575': 'DS575',
            'Ds595': 'DS595'
        }
        for j,cur_im in enumerate(self.imt.keys()):
            if cur_im == 'SA' or cur_im.startswith('DS') or cur_im.startswith('Ds'):
                pass
            else:
                self.site_data_dict[self.site_name][cur_im] = []
        for i in range(num_return_period):
            im_idx = 0
            num_secondary_im = len(self.imt.keys())
            for j,cur_im in enumerate(self.imt.keys()):
                if cur_im == 'SA':
                    num_secondary_im = num_secondary_im-1
                    if self.cim['SA'].get('Period') is None:
                        pass
                    else:
                        self.site_data_dict[self.site_name]['Sa(T1) (g)'].append(np.exp(self.user_im_tgt[j].get('Value')[i]))
                elif cur_im.startswith('DS') or cur_im.startswith('Ds'):
                    self.site_data_dict[self.site_name][IM_Conversion.get(cur_im)].append(np.exp(self.user_im_tgt[j].get('Value')[i]))
                    im_idx = im_idx+1
                else:
                    self.site_data_dict[self.site_name][cur_im].append(np.exp(self.user_im_tgt[j].get('Value')[i]))
            # covariance
            if self.user_im_std[i] is None:
                cur_sigma = np.zeros((num_secondary_im,num_secondary_im))
            else:
                cur_sigma = np.diag(self.user_im_std[i])
            if self.user_im_corr[i] is None:
                cur_corr = np.identity(num_secondary_im)
            else:
                cur_corr = np.array(self.user_im_corr[i])    
            self.site_data_dict[self.site_name]['Covariance'].append(np.dot(np.dot(cur_sigma,cur_corr),cur_sigma).tolist())
        # return
        return 0

    def create_groundmotionset(self, gms_config = None):
        """
        create a nested ground motion set
        input:
            gms_config: a dictionary of ground motion selection configuration
        """

        # load gms configuration
        if self._parse_gms_config(gms_config):
            err_msg = 'SAF_IDA.config_groundmotions: error in parsing ground motion selection configurations.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        self.logfile.write_msg(msg='SAF_IDA.config_groundmotions: ground motion selection configured.')

        # select records
        if self._select_records():
            err_msg = 'SAF_IDA.config_groundmotions: error in selecting ground motions.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        self.logfile.write_msg(msg='SAF_IDA.config_groundmotions: ground motion selection completed.')

        # return
        return 0

    def get_site_specific_hazard(self, site_config = None):
        # load site configuration
        if self._parse_site_config(site_config):
            err_msg = 'SAF_IDA.get_site_specific_hazard: error in parsing site configurations.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        self.logfile.write_msg(msg='SAF_IDA.get_site_specific_hazard: site configured.')

        # return
        return 0
    
    def get_user_defined_hazard(self, tgt_config = None):
        # load site configuration
        if self._parse_user_hazard_config(tgt_config):
            err_msg = 'SAF_IDA.get_user_defined_hazard: error in parsing user-defined configurations.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        self.logfile.write_msg(msg='SAF_IDA.get_user_defined_hazard: user-defined hazard target configured.')

        # return
        return 0
    
    def get_user_defined_im(self, tgt_config = None):
        # load site configuration
        if self._parse_user_im_config(tgt_config):
            err_msg = 'SAF_IDA.get_user_defined_im: error in parsing user-defined intensity measure.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        self.logfile.write_msg(msg='SAF_IDA.get_user_defined_im: user-defined IM target configured.')

        # return
        return 0

    def model_training(self, input_dir, train_config = None):
        # load info
        self.ida_datafile = os.path.join(input_dir,train_config.get('IDADataFile'))
        self.ida_gmdatafile = os.path.join(input_dir, train_config.get('IDAGMFile'))
        self.collapse_im = train_config.get('CollapseIM','Sa (g)')
        self.collapse_edp = train_config.get('CollapseEDP','SDRmax')
        self.collapse_limit = train_config.get('CollapseLimit',0.10)
        self.saf_model = SSM.SurrogateModel(idadatafile=self.ida_datafile,gmdatafile=self.ida_gmdatafile,
                                            train_config=train_config)
        # collecting collapse IM
        self.saf_model.get_collapse_im(cim=self.collapse_im,cedp=self.collapse_edp,climit=self.collapse_limit)
        # EDP type
        self.edp_type = train_config.get('EDPType',[])
        # EDP ranges
        self.edp_range = train_config.get('EDPRange',dict())
        # EDP IM
        self.edp_im = train_config.get('EDPIM','Sa (g)')
        # collecting EDP IM (it's hard-coded for SDR and PFA now - to fix this soon, KZ)
        if len(self.edp_type)>0:
            self.saf_model.get_edp_im(edpim=self.edp_im,SDR=self.edp_range.get('SDR',[-np.inf,np.inf]),PFA=self.edp_range.get('PFA',[-np.inf,np.inf]))
        # collaspse model
        self.col_model_type = train_config.get('CollapseModelType','OLS')
        self.col_model_param = train_config.get('CollapseModelParam',[])
        self.saf_model.compute_collapse_model(modeltag=self.col_model_type,modelcoef=self.col_model_param)
        # EDP model
        if len(self.edp_type)>0:
            self.edp_model_type = train_config.get('EDPModelType','OLS')
            self.edp_model_param = train_config.get('EDPModelParam',[])
            self.saf_model.compute_edp_model(modeltag=self.edp_model_type,modelcoef=self.edp_model_param)

        # return
        return 0

    def model_prediction(self, pred_config, output_dir):
        # load info
        self.pred_response = [(pred_config.get('Site'),pred_config.get('Response'))]
        if pred_config.get('TargetType') in ['SiteSpecific','UserDefined','UserDefinedHazard']:
            # site
            cur_site = SSInfo.SiteInfo(dataname=self.site_name,site_data_dict=self.site_data_dict)
            # prediction
            self.site_adj = HA.SiteAdjustment(surrogate=self.saf_model,site=cur_site)
            self.site_adj.site_specific_performance(setname=self.pred_response)
            # save
            filename = pred_config.get('ResultFilename',None)
            self.save_to_file(filename=filename)
        else:
            # site
            cur_site = SSInfo.SiteInfo(dataname=self.site_name,site_im_dict=self.site_data_dict)
            # prediction
            self.site_adj = HA.SiteAdjustment(surrogate=self.saf_model,site=cur_site)
            self.site_adj.site_specific_performance_user(setname=self.pred_response)
    

    def save_to_file(self, filename=None, outdir=None):
        # output directory
        if outdir is None:
            outdir = self.output_dir
        # file path
        if filename is None:
            filename = 'saf-ida.json'
        outpath = os.path.join(outdir,filename)
        # convert data to json first
        self.res = dict()
        res_list = list(self.site_adj.ssp.keys())
        for cur_res in res_list:
            self.res.update(self.site_adj.ssp.get(cur_res))
        # get the file format
        if outpath.endswith('.csv'):
            tmp = dict()
            # csv
            for key1, value1 in self.res.items():
                if key1 == 'Collapse':
                    tmp.update({'Collapse Capacity': value1.get('Fragility')})
                else:
                    if type(value1) is dict:
                        for key2, value2 in value1.items():
                            cur_key = '{}-{}'.format(key1,key2)
                            tmp.update({cur_key:value2})
                    else:
                        tmp.update({cur_key:np.array(value1).flatten().tolist()})
            tmp_df = pd.DataFrame.from_dict(tmp)
            tmp_df.to_csv(outpath)

        elif outpath.endswith('.json'):
            # json
            with open(outpath,'w') as f:
                json.dump(self.res, f, indent=2)
        else:
            err_msg = 'SAF_IDA.save_to_file: the file format is not supported (please reselect from csv and json).'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        
        # return
        return 0


def run_saf_ida(job_name = 'saf_ida', job_config = ''):

    # read job configuration
    if not os.path.exists(job_config):
        print('run_saf_ida: configuration file not found: {}'.format(job_config))
        return 1
    try:
        with open(job_config) as f:
            job_info = json.load(f)
    except:
        print('run_saf_ida: malformatted configuration file: {}'.format(job_config))
        return 1

    # directory
    dir_info = job_info['Directory']
    work_dir = dir_info['Work']
    input_dir = dir_info['Input']
    output_dir = dir_info.get('Output',None)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__),'Output'))
    try:
        os.mkdir(f"{output_dir}")
    except:
        print('runSAF_IDA: output directory already exists.')

    # job type
    job_type = job_info.get('Type', None)
    if job_type is None:
        err_msg = 'run_saf_ida: Please specity "Type" in the configuraiton file.'
        saf_ida_job.logfile.write_msg(msg=err_msg, msg_type='ERROR')
        return 1

    # create SAF_IDA job
    saf_ida_job = SAF_IDA(dir_info=dir_info, job_name=job_name)

    # run jobs
    if 'GroundMotionSelection' in job_type:
        # get the ground motion selection config.
        gms_config = job_info.get('GroundMotionSelection', None)
        if gms_config is None:
            err_msg = 'run_saf_ida: GroundMotionSelection not found in job configuration.'
            saf_ida_job.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1

        # create ground motions
        saf_ida_job.create_groundmotionset(gms_config=gms_config)
    
    if 'SiteSpecificHazard' in job_type:
        # KZ: 03/02/25 - extending this to user-defined target
        tgt_type = job_info.get('Prediction',dict()).get('TargetType','SiteSpecific')
        if tgt_type == 'SiteSpecific':
            # get the site config.
            site_config = job_info.get('SiteSpecificHazard', None)
            if site_config is None:
                err_msg = 'run_saf_ida: SiteSpecificHazard not found in job configuration.'
                saf_ida_job.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            # create site specific hazard information data
            saf_ida_job.get_site_specific_hazard(site_config=site_config)
        elif tgt_type in ['UserDefined','UserDefinedHazard']:
            tgt_config = job_info.get('UserDefinedHazard',None)
            if tgt_config is None:
                err_msg = 'run_saf_ida: UserDefinedHazard not found in job configuration.'
                saf_ida_job.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            # create user-defined hazard information data
            saf_ida_job.get_user_defined_hazard(tgt_config=tgt_config)
        elif tgt_type in ['UserDefinedIM','UserDefinedIntensityMeasure']:
            tgt_config = job_info.get('UserDefinedIM',None)
            if tgt_config is None:
                err_msg = 'run_saf_ida: UserDefinedIM not found in job configuration.'
                saf_ida_job.logfile.write_msg(msg=err_msg, msg_type='ERROR')
                return 1
            # create user-defined hazard information data
            saf_ida_job.get_user_defined_im(tgt_config=tgt_config)
        else:
            err_msg = 'run_saf_ida: TargetType not supported yet - please contact us.'
            saf_ida_job.logfile.write_msg(msg=err_msg, msg_type='ERROR')
        # kz: save
        if tgt_config.get('SaveTarget',False):
            if tgt_config.get('SaveCOV',False):
                pass
            else:
                tmp = copy.deepcopy(saf_ida_job.site_data_dict)
                del tmp[saf_ida_job.site_name]['Covariance']
            with open (os.path.join(output_dir,'site_specific_target.json'),'w') as f:
                json.dump(tmp,f,indent=2) 

    if 'Training' in job_type:
        # get training config
        train_config = job_info.get('Training', None)
        if train_config is None:
            err_msg = 'run_saf_ida: Training not found in job configuration.'
            saf_ida_job.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # create a training run
        saf_ida_job.model_training(input_dir=input_dir, train_config=train_config)
        if train_config.get('PlotData',False):
            saf_ida_job.saf_model.plot_raw_collapse(logscale=[False,False],outdir=output_dir)

    if 'Prediction' in job_type:
        # get training config
        pred_config = job_info.get('Prediction', None)
        if pred_config is None:
            err_msg = 'run_saf_ida: Prediction not found in job configuration.'
            saf_ida_job.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # create a training run
        saf_ida_job.model_prediction(pred_config=pred_config, output_dir=output_dir)
        # kz: save
        if saf_ida_job.col_model_type in ['OLS']:
            col_a0, col_ai = saf_ida_job.saf_model.col_model.get_coef()
        else:
            col_a0 = None
            col_ai = []
        with open (os.path.join(output_dir,'collapse_model_coef.json'),'w') as f:
            json.dump({
                "ModelType": saf_ida_job.col_model_type,
                "Scale": "LogLog",
                "IMResponse": saf_ida_job.collapse_im,
                "IMPredictor": [1]+saf_ida_job.saf_model.im_predictor,
                "Coefficients": [col_a0]+list(col_ai)
            },f,indent=2) 


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name')
    parser.add_argument('--job_config')
    args = parser.parse_args()

    # run saf-ida
    run_saf_ida(job_name = args.job_name, job_config = args.job_config)