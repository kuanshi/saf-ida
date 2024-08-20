# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Leland Stanford Junior University
# Copyright (c) 2021 The Regents of the University of California
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

import argparse, json, os
from general import *
from pyngms import NestedGroundMotionSelection as NGMS
from pyhca import SiteSpecificInformation as SSInfo

class SAF_IDA:

    def __init__(self, dir_info = dict(), job_name = 'saf_ida'):

        # initiate a log file
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
        self.site_data.set_im_calculator()
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
                'Ds575 (s)': [],
                'Ds595 (s)': [],
                'Covariance': []
            }
        }
        IM_Conversion = {
            'DS575': 'Ds575 (s)',
            'DS595': 'Ds595 (s)'
        }
        for i,cur_rp in enumerate(self.return_periods):
            cur_im_target = self.site_data.im_target[i]
            im_idx = 0
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
                elif cur_im.startswith('DS'):
                    self.site_data_dict[self.site_name][IM_Conversion.get(cur_im)].append(cur_im_target.get('Median')[im_idx])
                    im_idx = im_idx+1
                else:
                    pass
            # covariance
            self.site_data_dict[self.site_name]['Covariance'].append((np.diag(cur_im_target.get('StandardDev')).dot(np.array(cur_im_target.get('Correlation'))).dot(np.diag(cur_im_target.get('StandardDev')))).tolist())
        
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

    def get_site_specific_hazard(self, site_config = None):
        # load site configuration
        if self._parse_site_config(site_config):
            err_msg = 'SAF_IDA.get_site_specific_hazard: error in parsing site configurations.'
            self.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        self.logfile.write_msg(msg='SAF_IDA.get_site_specific_hazard: site configurated.')


def run_saf_ida(job_name = 'saf_ida', job_config = ''):

    # read job configuration
    if not os.path.exists(args.job_config):
        print('run_saf_ida: configuration file not found: {}'.format(args.job_config))
        return 1
    try:
        with open(args.job_config) as f:
            job_info = json.load(f)
    except:
        print('run_saf_ida: malformatted configuration file: {}'.format(args.job_config))
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
        # get the site config.
        site_config = job_info.get('SiteSpecificHazard', None)
        if site_config is None:
            err_msg = 'run_saf_ida: SiteSpecificHazard not found in job configuration.'
            saf_ida_job.logfile.write_msg(msg=err_msg, msg_type='ERROR')
            return 1
        # create site specific hazard information data
        saf_ida_job.get_site_specific_hazard(site_config=site_config)


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name')
    parser.add_argument('--job_config')
    args = parser.parse_args()

    # run saf-ida
    run_saf_ida(job_name = args.job_name, job_config = args.job_config)