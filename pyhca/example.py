from StructuralSurrogateModel import SurrogateModel
from SiteSpecificInformation import SiteInfo
from HazardAdjustment import SiteAdjustment

# building surrogate model object
a = SurrogateModel(idadatafile='./data/IDAData.json',
                          gmdatafile='./data/NGMS_T1.71G7x7.json')
# collecting collapse Sa
a.get_collapse_im(cim='Sa (g)',cedp='SDRmax',climit=0.1)

# defining EDP ranges
lb_sdr = 5.0e-4
up_sdr = 0.1
lb_pfa = 1.0e-4
up_pfa = 10.0
ndiv = 20

# getting im at EDP levels
a.get_edp_im(edpim='Sa (g)',SDR=[lb_sdr,up_sdr,ndiv],PFA=[lb_pfa,up_pfa,ndiv])

# getting collapse model
a.compute_collapse_model(modeltag='LLM',modelcoef=['Gaussian',['CV',5],[0.5,2],20])
#a.compute_collapse_model(modeltag='OLS',modelcoef=[]) # OLS model

# getting EDP models
a.compute_edp_model(modeltag='OLS',modelcoef=[])

# getting site info
b = SiteInfo(sitedatafile='./data/SiteData.json')

# building site specific response object
c = SiteAdjustment(surrogate=a,site=b)
# computing site specific performance metrics
c.site_specific_performance(setname=[('All','All')])
#c.site_specific_performance(setname=[('Los Angeles-1','Collapse')])

# plotting
c.plot_result(setname=[('Los Angeles-1','Collapse')])
c.plot_result(setname=[('Los Angeles-1','SDRmax')])
