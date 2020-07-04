# Site-Specific Adjustment Framework for Incremental Dynamic Analysis (SHAF-IDA)
The Site-specific Hazard Adjustment Framework for Incremental Dynamic Analysis (SHAF-IDA) is developed for efficiently estimate site-specific structural performance, e.g., esitmating probabilistic distributions of engineering demand parameters (EDP), evalutating fragilities of damage measures (DM), and assessing collapse risks.

The multiple stripe analysis (MSA) ([Jalayer, 2003](https://ui.adsabs.harvard.edu/abs/2003PhDT........34J/abstract)) is one state-of-the-art method to conduct site-specific seismic performance assessments that is recommended by FEMA P-58 ([FEMA, 2012](https://www.fema.gov/media-library/assets/documents/90380)). As illustrated in the Figure below, engineers first build the structural model and obtain the fundamental period T<sub>1</sub>, then select different ground motion suites for different intensity levels to match the correponding earthquake scenarios (e.g., [Goulet et al., 2007](https://onlinelibrary.wiley.com/doi/abs/10.1002/eqe.694)) or conditional intensity measure targets (e.g., [Bradley, 2010](https://onlinelibrary.wiley.com/doi/full/10.1002/eqe.995); [Chandramohan et al., 2016](https://onlinelibrary.wiley.com/doi/full/10.1002/eqe.2711)). For different sites, the ground motion selection and structural analysis need to be repeated. When being applied to the regional assessment, the MSA can be time-consuming due to re-selecting records and re-analyzing structures.  Similarly, the MSA is not flexible for dealing the time-dependent seismic hazard (e.g., aftershocks and induced earthquakes).

![](https://github.com/kuanshi/shaf-ida/blob/master/doc/image/MSA_vs_SHAF-IDA.png)
<img align="center" width="50%" height="50%" src="https://github.com/kuanshi/shaf-ida/blob/master/doc/image/MSA_vs_SHAF-IDA.png">
