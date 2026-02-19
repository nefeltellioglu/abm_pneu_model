# Individual-based pneumococcal transmission model

## Installation

The code can be installed as a package. 

You can set up a Python virtual environment to run the code. Once you activate the virtual environment, you can run  `pip install -e .` after setting the current working directory as the code repository.

```shell
python3 -m venv venv
source venv/bin/activate
# set working directory as code repository then
pip install -e .
```

## Manuscript

The manuscript that explains the model in more detail is available as a preprint: [Tellioglu et al.](https://www.medrxiv.org/content/10.1101/2025.05.22.25327965v4). 

## Code guide

The background for the pathogen burden, need for a novel ABM model, structure of the repository and how to run example scenarios are all described in [introduction.ipynb](src/pneu_abm/code_guide/introduction.ipynb) Notebook.