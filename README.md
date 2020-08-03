## Sparse Gaussian Processes using Pseudo Inputs

##### Implementation in Python

This is repository contains a python implementation of Sparse Gaussian Processes using Pseudo Inputs published in NIPS 2005. [[LINK]](http://www.gatsby.ucl.ac.uk/~snelson/SPGP_talk.pdf)

In order to use this package, please import the package as follows -

```python
import spgp
```

This package contains two functions:

* **spgp.utilityfn**: Contains the function for likelihood calculation and estimating the mean & variance using learned SPGP hyper-params.
* **spgp.minimize**: Uses Carl's Rasmussen's implementation for finding a local minimum of a nonlinear multivariate function.

This repository also contains an example implementation for a 2D spatial regression problem. This example uses ```plotly``` for generating the outputs and these are saved as an offline plot in the ```output``` folder.



To use these codes, please refer the following publications:

1. Rajat Mishra, Mandar Chitre, and Sanjay Swarup. "Online Informative Path Planning using Sparse Gaussian Processes." *2018 OCEANS-MTS/IEEE Kobe Techno-Oceans (OTO)*. IEEE, 2018.
2. Edward Snelson and Zoubin Ghahramani. "Sparse Gaussian Processes using pseudo-inputs." *Advances in neural information processing systems*. 2006.