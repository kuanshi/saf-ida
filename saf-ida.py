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


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name')
    parser.add_argument('--job_config')
    args = parser.parse_args()

    # run saf-ida
    run_saf_ida(job_name = args.job_name, job_config = args.job_config)