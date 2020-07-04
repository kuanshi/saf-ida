# Nested Ground Motion Set
The nested ground motion set is designed for the IDA to select a suite of records whose key intensity measures closely form an orthogonal grid.  This orthogonal grid (target) can be in high dimensions where multiple intensity measures are considered.  For instance, if using the spectral shape measure SaRatio ([Baker and Cornell, 2006](https://onlinelibrary.wiley.com/doi/10.1002/eqe.571); [Eads et al., 2016](https://onlinelibrary.wiley.com/doi/full/10.1002/eqe.2575)) and ground motion duration Ds<sub>5-75</sub> ([Bommer and Martinez-Pereira, 1999](https://www.tandfonline.com/doi/abs/10.1080/13632469909350343), the figure below shows one example grid target (grey dots) as well as the selected ground motion records (red dots) for a conditioning period T<sub>1</sub> = 2s. The algorithm can also penalize large scaling factors as specified by users. In addition to covering a wide range of possible ground motion characteristics, the orthogonal grid is designated to avoid strong correlations among the intensity measures in the generic ground motion set. This correlation issue is termed collinearity ([Baker and Cornell, 2005](https://onlinelibrary.wiley.com/doi/abs/10.1002/eqe.474)): if two intensity measures are highly correlated, it is difficult to separate their effects on, with confidence, the corresponding coefficients in the response prediction model.

<p align="center">
 <img width="40%" height="40%" src="https://github.com/kuanshi/shaf-ida/blob/master/doc/image/NGMS_GRID.png">
</p>

The class NestedGroundMotionSet in this folder now supports to select ground motion records from the [NGA-West](https://peer.berkeley.edu/nga-west) and [NGA-Sub](https://www.risksciences.ucla.edu/nhr3/gmdata/preliminary-nga-subduction-records) ground motion databases to fit upto a 7-D grid target with intensity measures: SaRatio(T<sub>1</sub>), Ds<sub>5-75</sub>, Ds<sub>5-95</sub>, PGA, PGV, PGD, and Ia. These intensity measures of the two ground motion databases are pre-calculated in the [GroundMotionCharacteristics.json](https://github.com/kuanshi/shaf-ida/tree/master/pyngms/data). Other intensity measures can be included by modifying the JSON data file and the including the intensity name tag into the class.

The example.py provides a simple demo for selecting the nested ground motion set with SaRatio and Ds<sub>5-75</sub>, including five major steps:

1. Creating the new NestedGroundMotionSet object:
```python
a = NestedGroundMotionSet()
```
2. Setting up the number of dimensions, numbers of divisions, and the names and ranges of key intensity measure:
```python
a.definekeyim(2,[7,7],SaRatio=[1,0.2,3,np.exp(-0.2),np.exp(1.0)],Ds575=[np.exp(1.0),np.exp(4.5)])
```
3. Generating the grid:
```python
a.generategrid()
```
4. Defining the scaling factor limits: minimum scaling factor, maximum scaling factor, T<sub>1</sub>, Sa(T<sub>1</sub>) to scale to (optional):
```python
a.scalinglimit(0.5,5,10,1,0.9)
```
5. Defining the ground motion database and conducting the selection
```python
a.selectnestedrecord(gmdb_path='./data/GroundMotionCharacteristics.json')
```

