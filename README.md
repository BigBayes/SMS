# smssample
This repo contains the scripts used in the following paper:

**Distributed Bayesian Posterior Sampling via Moment Sharing**

Minjie Xu, Balaji Lakshminarayanan, Yee Whye Teh, Jun Zhu and Bo Zhang

*Advances in Neural Information Processing Systems (NIPS), 2014.*

[Link to PDF](http://papers.nips.cc/paper/5596-distributed-bayesian-posterior-sampling-via-moment-sharing.pdf)

Please cite the above paper if you use this code.

If you have any questions/comments/suggestions, please contact Minjie ([chokkyvista06@gmail.com](mailto:chokkyvista06@gmail.com)) or Balaji 
([balaji@gatsby.ucl.ac.uk](mailto:balaji@gatsby.ucl.ac.uk)).

Code released under MIT license (see COPYING for more info).

Copyright (c) 2015, Minjie Xu and Balaji Lakshminarayanan

*Note*: spikeslab_gibbs.m has been modified from Michalis Titsias's paired Gibbs sampler for Bayesian sparse linear regression with spike and slab prior. 

----------------------------------------------------------------------------

**List of scripts in the folder**:

- smssample.m (main script that does the bulk of the computation)

Examples of approximating families:

- approxfam_gaussian.m
- approxfam_spikeslab.m

Examples of model specification:

- gaussian.m, gaussian_gendata.m
- bayeslogreg.m, bayeslogreg_gendata.m
- spikeslab.m, spikeslab_gendata.m
- sparsebayeslogreg.m, sparsebayeslogreg_gendata.m

Examples of base samplers:

- sampler_nuts_da.m
- sampler_spikeslab.m (calls spikeslab_gibbs.m)

Utilities:

- nuts_da.m
- nuts_da.cpp
- sigmoid.m
- vislsm.m
- viserrs.m

**Demo scripts**:

- bayeslogreg_test.m (Bayesian logistic regression)
- spikeslab_test.m (Bayesian linear regression with spike-and-slab prior over weights)

*Note*: You may need to write additional scripts to aggregate the results and generate the plots.

----------------------------------------------------------------------------

**How do I use your scripts for my model/sampler?**

See above for examples of model specifications, approximating families and samplers.

- Create your model specification MODEL.m
- Create your approximating family approxfam_XYZ.m
- Create your sampler sampler_ABC.m
- Create a wrapper script that will invoke the above and call smsample.m (see demo scripts above)
