# MPS-model-code
Model code for the MPS crash course. There is a branch for each week.

## Week 0

This branch provides a template for the folder structure. You should place the core files in the `src/` folder, tests in `test\`, and exercises in `exercises\`. Python modules personally give me a headache, so I wanted to give you this template as a starting point. Just remember to add
```python
from fix_pathing import root_dir
```
to the top of any test or exercise files.

I am assuming familiarity with python, but I have also provided yaml files for setting up your conda environment with the necessary packages. Run these from your terminal using 
```
conda env create -f environment.yaml
```
It will create an environement called `mps_course`.
There is a separate yaml file for if you have a mac with Apple Silicon. This will make the code run much more efficiently (not necessarily faster).
