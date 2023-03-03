import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import feather
from pathlib import Path
from pandas.io.json import json_normalize
from tqdm import tqdm
import fastai
from PetData import *

from fastai.tabular import *

pets = get_data()

dep_var = 'AdoptionSpeed'
cont_names, cat_names = cont_cat_split(pets, 50, dep_var=dep_var)
cat_names.remove('Filename')
cat_names.remove('PicturePath')

miss = FillMissing(cat_names, cont_names)

df = miss.apply_train(pets)

df.columns
