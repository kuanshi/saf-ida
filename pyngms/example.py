import numpy as np
from NestedGroundMotionSelection import NestedGroundMotionSet

a = NestedGroundMotionSet()
a.definekeyim(2,[7,7],SaRatio=[1,0.2,3,np.exp(-0.2),np.exp(1.0)],Ds575=[np.exp(1.0),np.exp(4.5)])
a.generategrid()
a.scalinglimit(0.5,5,10,1,0.9)
a.selectnestedrecord(gmdb_path='./data/GroundMotionCharacteristics.json')
a.savedata()