# Constants, variables, and methods that are commonly used

import os
from datetime import datetime
import numpy as np
import pandas as pd

class Logfile:
    def __init__(self, logfile_dir = './', logfile_name = 'saf_ida.log', screen_msg = True):
        """
        Initializing the logfile
        - logfile_dir: default is the same path of the PLoM package
        - logfile_name: default is the "saf_ida.log"
        - screen_msg: default is to show message on screen
        """
        self.logfile_dir = logfile_dir
        self.logfile_name = logfile_name
        self.logfile_path = os.path.join(self.logfile_dir, self.logfile_name)
        self.screen_msg = screen_msg
        # start the log
        self.write_msg(msg = '--NEW LOG STARTING FROM THIS LINE--', mode='w')            
    
    def write_msg(self, msg = '', msg_type = 'RUNNING', msg_level = 0, mode='a'):
        """
        Writing running messages
        - msg: the message
        - msg_type: the type of message 'RUNNING', 'WARNING', 'ERROR'
        - msg_level: how many indent tags
        """
        indent_tabs = ''.join(['\t']*msg_level)
        decorated_msg = '{} {} {}-MSG {} '.format(datetime.utcnow(), indent_tabs, msg_type, msg)
        if self.screen_msg:
            print(decorated_msg)
        with open(self.logfile_path, mode) as f:
            f.write('\n'+decorated_msg)
    
    def delete_logfile(self):
        """
        Deleting the log file
        """
        if os.path.exists(self.logfile_path):
            os.remove(self.logfile_path)
        else:
            print('The logfile {} does not exist.'.format(self.logfile_path))


class DBServer:
    def __init__(self, db_dir = './', db_name = 'saf_ida.h5'):
        """
        Initializing the database
        - db_dir: default is the same path of the PLoM package
        - db_name: default is "saf_ida.h5"
        """
        self.db_dir = db_dir
        self.db_name = db_name
        self.db_path = os.path.join(self.db_dir, self.db_name)
        if os.path.exists(self.db_path):
            # deleting the old database
            os.remove(self.db_path)
        self.init_time = datetime.utcnow()
        self.item_name_list = []
        self.basic()
        self.dir_export = self._create_export_dir()
            
    
    def basic(self):
        """
        Writing basic info
        """
        df = pd.DataFrame.from_dict({
            'InitializedTime': [self.init_time],
            'LastEditedTime': [datetime.utcnow()],
            'DBName': [self.db_name],
        }, dtype=str)
        store = pd.HDFStore(self.db_path, 'a')
        df.to_hdf(store, 'basic', mode='a')
        store.close()


    def _create_export_dir(self):
        """
        Creating a export folder
        """
        dir_export = os.path.join(self.db_dir,'DataOut')
        try:
            os.makedirs(dir_export, exist_ok=True)
            return dir_export
        except:
            return None


    def get_item_adds(self):
        """
        Returning the full list of data items
        """
        return self._item_adds


    def add_item(self, item_name = None, col_names = None, item = [], data_shape = None, data_type='Data'):
        """
        Adding a new data item into database
        """
        if data_type == 'Data':
            if item.size > 1:
                df = pd.DataFrame(item, columns = col_names)
                dshape = pd.DataFrame(data_shape, columns=['DS_'+item_name])
            else:
                if col_names is None:
                    col_names = item_name
                df = pd.DataFrame.from_dict({
                    col_names: item.tolist()
                })
                dshape = pd.DataFrame.from_dict({
                    'DS_'+col_names: (1,)
                })
            if item_name is not None:
                store = pd.HDFStore(self.db_path, 'a')
                # data item
                df.to_hdf(store, item_name, mode='a')
                # data shape
                dshape.to_hdf(store, 'DS_'+item_name, mode='a')
                store.close()
        elif data_type == 'ConstraintsFile':
            # constraints filename
            cf = pd.DataFrame.from_dict({
                'ConstraintsFile': item
            }, dtype=str)
            store = pd.HDFStore(self.db_path, 'a')
            cf.to_hdf(store, 'constraints_file', mode='a')
            store.close()
        else:
            # Not supported data_type
            return False


    def get_item(self, item_name = None, table_like=False, data_type='Data'):
        """
        Getting a specific data item
        """
        if data_type == 'Data':
            if item_name is not None:
                store = pd.HDFStore(self.db_path, 'r')
                try:
                    item = store.get(item_name)
                    item_shape = tuple([x[0] for x in self.get_item_shape(item_name=item_name).values.tolist()])
                    if not table_like:
                        item = item.to_numpy().reshape(item_shape)                
                except:
                    item = None
                finally:
                    store.close()

                return item

            return item.values.tolist()[0][0]


    def remove_item(self, item_name = None):
        """
        Removing an item
        """
        if item_name is not None:
            store = pd.HDFStore(self.db_path, 'r')
            try:
                store.remove(item_name)
            except:
                item = None
            finally:
                store.close()


    def get_item_shape(self, item_name = None):
        """
        Getting the shape of a specific data item
        """
        
        if item_name is not None:
            store = pd.HDFStore(self.db_path, 'r')
            try:
                item_shape = store.get('DS_'+item_name)
            except:
                item_shape = None
            store.close()

            return item_shape


    def get_name_list(self):
        """
        Returning the keys of the database
        """
        store = pd.HDFStore(self.db_path, 'r')
        try:
            name_list = store.keys()
        except:
            name_list = []
        store.close()
        return name_list


    def export(self, data_name = None, filename = None, file_format = 'csv'):
        """
        Exporting the specific data item
        - data_name: data tag
        - format: data format
        """
        d = self.get_item(item_name = data_name[1:], table_like=True)
        if d is None:
            return 1
        if filename is None:
            filename = os.path.join(self.dir_export,str(data_name).replace('/','')+'.'+file_format)
        else:
            filename = os.path.join(self.dir_export,filename.split('.')[0]+'.'+file_format)
        if file_format == 'csv' or 'txt':
            d.to_csv(filename, header=True, index=True)
        elif file_format == 'json':
            with open(filename, 'w') as f:
                json.dump(d, f)
        else:
            return 2
        return filename
        
