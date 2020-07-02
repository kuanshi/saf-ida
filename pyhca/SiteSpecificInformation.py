"""Site specific information"""

import json

__author__ = 'Kuanshi Zhong'

class SiteInfo:
    
    def __init__(self,dataname='SiteData',sitedatafile=[]):
        """
        __init__: initialization
        """
        self.sitedatafile = sitedatafile
        self.nCase = 0
        self.nameCase = []
        self.SiteCase = {}
        self.__load_data()
        
    def __load_data(self):
        """
        __loadata: loading site data
        """
        print("Loading site data.")
        # Site data
        if len(self.sitedatafile):
            with open(self.sitedatafile) as f:
                data = json.load(f)
        self.nCase = data['Number of cases']
        self.nameCase = data['Case name']
        for tagcase in self.nameCase:
            self.SiteCase[tagcase] = data[tagcase]
        print("Site data loaded.")
        
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
