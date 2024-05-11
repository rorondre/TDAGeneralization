# Code for our experimental results

We provide the code base that we used to produce our main experiments and figures. All our implementation has been done in python, tested with python `3.9`.

## Installation

You can create a virtual environment and install the required packages through the provided `requirements.txt` file.
````
python -m pip install -r requirements.txt
````

| :memo:        | This will install a cpu-only version of pytorch, feel free to install torch and torchvision according to your CUDA version.        |
|---------------|:------------------------|


## Tests

We provide some pytests to assess that the installation went well>
````
pytest -x -vv
````
All tests should pass.


## Main experiments

The code for the main experiments can be found in the folder `PHDim`, several models, in particular Vision transformers, can be found in the folder `models`. The folder `analysis` contain the source code to compute all the quantities that appear in our work, namely $E_\alpha$, magnitude, positive magnitude as well as the persistent homology dimension.

The general command to run one experiment is the following (linux):

````
PYTHONPATH=$PWD python -m PHDim [--args ARGS]
````

Additional arguments can be passed to the function, they can be found in PHDim/__main__.py. Calling this script will launch individual trainings on a specified grid of hyperparameters. Each of these individual trainings are implemented in the script `train_risk_analysis.py`.


# References

Our code is built on the two following studies, and their released code.

````
@article{DBLP:journals/corr/abs-2111-13171,
  author    = {Tolga Birdal and
               Aaron Lou and
               Leonidas J. Guibas and
               Umut Simsekli},
  title     = {Intrinsic Dimension, Persistent Homology and Generalization in Neural
               Networks},
  journal   = {CoRR},
  volume    = {abs/2111.13171},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.13171},
  eprinttype = {arXiv},
  eprint    = {2111.13171},
  timestamp = {Wed, 01 Dec 2021 15:16:43 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-13171.bib}
}
````

````
@inproceedings{dupuis2023generalization,
  title={Generalization bounds using data-dependent fractal dimensions},
  author={Dupuis, Benjamin and Deligiannidis, George and Simsekli, Umut},
  booktitle={International Conference on Machine Learning},
  pages={8922--8968},
  year={2023},
  organization={PMLR}
}
````

````
@misc{andreeva_metric_2023,
  title = {Metric {{Space Magnitude}} and {{Generalisation}} in {{Neural Networks}}},
  author = {Andreeva, Rayna and Limbeck, Katharina and Rieck, Bastian and Sarkar, Rik},
  year = {2023},
  month = may,
  number = {arXiv:2305.05611},
  eprint = {2305.05611},
  primaryclass = {cs, math, stat},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2305.05611}
}
````