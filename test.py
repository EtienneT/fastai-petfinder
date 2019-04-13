#%% [markdown]
# # Image, tabular and text data in the same deep learning model
# 
# Deep learning has advanced tremendously in the last 2-3 years.  Researchers are always pushing more and more the boundaries of the state of the art in various sub-domain.  To do that researchers also have to specialize and imerse themselves in one domain of deep learning.  We often see deep learning models handling image data or text data or structured data.  But we rarely see them used together when you have a dataset that contain them all.  Datasets in the real world are much messier than academic datasets.  Being able to leverage everything you have in your data can yield very interesting results.
# 
# # Transfer Learning
# 
# Transfer learning in deep learning has also become very popular in recent years, more specially for image data with pre-trained ImageNet models that leverage models trained on millions of images that are expensive to train but allow you to re-use that knowledge for other tasks.  More recently text data started having its own transfer learning moment with pre-trained models like ULMFit, BERT and GPT-2.

#%%
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import feather
from fastai.text import *

from petfinder.data import *


#%%
path = 'C:\\work\\ML\\PetFinder\\'
bs=64

pets = get_data()
petsTest = get_data(True)

# pets['IsTest'] = False
# petsTest['IsTest'] = True

# pets = pd.concat([pets, petsTest])

# pets = feather.read_dataframe(path + 'pets.feather')
data_lm = load_data(path, 'data_lm_descriptions.pkl', bs=bs)

#%%
from fastai.tabular import *
from fastai.vision import *
from fastai.metrics import *
from fastai.text import *

dep_var = 'AdoptionSpeed'
cont_names, cat_names = cont_cat_split(pets, 50, dep_var=dep_var)
procs = [FillMissing, Categorify, Normalize]
cat_names.remove('Filename')
cat_names.remove('PicturePath')
cat_names.remove('PetID')
cat_names.remove('Description')


#%%
# cont_names, cat_names

#%%
from petfinder.model import *

#%%
from fastai.callbacks import *

bs = 32
size = 224
np.random.seed(42)

data_lm = load_data(path, 'data_lm_descriptions.pkl', bs=bs)
vocab = data_lm.vocab

imgList = ImageList.from_df(pets, path=path, cols='PicturePath')
tabList = TabularList.from_df(pets, cat_names=cat_names, cont_names=cont_names, procs=procs, path=path)
textList = TextList.from_df(pets, cols='Description', path=path, vocab=vocab)

norm, denorm = normalize_custom_funcs(*imagenet_stats)

if os.path.isfile(path + 'mixed_img_tab_text.pkl') != True :
    mixed = (MixedItemList([imgList, tabList, textList], path, inner_df=tabList.inner_df)
            .random_split_by_pct(.1)
            .label_from_df(cols='AdoptionSpeed', label_cls=CategoryList)
            .transform([[get_transforms()[0], [], []], [get_transforms()[1], [], []]], size=size))

    outfile = open(path + 'mixed_img_tab_text.pkl', 'wb')
    pickle.dump(mixed, outfile)
    outfile.close()
else:
    infile = open(path + 'mixed_img_tab_text.pkl','rb')
    mixed = pickle.load(infile)
    infile.close()


#%%
# data_text = textList.random_split_by_pct(.1).label_from_df(cols='AdoptionSpeed').databunch(bs=bs)
# data_text.save('text-classification-databunch.pkl')
data_text = load_data(path, 'text-classification-databunch.pkl')


#%%
data = mixed.databunch(bs=bs, collate_fn=collate_mixed, num_workers=0)
data.add_tfm(norm) # normalize images


#%%
cat_names = mixed.train.x.item_lists[1].cat_names
cont_names = mixed.train.x.item_lists[1].cont_names


#%%
# from fastai.callbacks.tensorboard import LearnerTensorboardWriter

learn = image_tabular_text_learner(data, len(cont_names), len(vocab.itos), data_text)

learn.callback_fns +=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.005, patience=3)]
# learn.callback_fns += [(partial(LearnerTensorboardWriter, base_dir=Path(path + 'logs\\'), name='mixed-metadata'))]


#%%
data.c

learn.lr_find()

learn.load('mixed-300')

# imgList = ImageList.from_df(petsTest, path=path, cols='PicturePath')
# tabList = TabularList.from_df(petsTest, cat_names=cat_names, cont_names=cont_names, procs=procs, path=path)
# textList = TextList.from_df(petsTest, cols='Description', path=path, vocab=vocab)

# norm, denorm = normalize_custom_funcs(*imagenet_stats)

# mixedTest = (MixedItemList([imgList, tabList, textList], path, inner_df=tabList.inner_df))

# learn = load_learner(path, 'mixed.pkl', test=mixedTest)

pets['IsTest'] = False
petsTest['IsTest'] = True

petsAll = pd.concat([pets, petsTest])
petsAll[pets.IsTest == True].AdoptionSpeed = -1

imgListTest = ImageList.from_df(petsAll, path=path, cols='PicturePath')
tabListTest = TabularList.from_df(petsAll, cat_names=cat_names, cont_names=cont_names, procs=procs, path=path)
textListTest = TextList.from_df(petsAll, cols='Description', path=path, vocab=vocab)

mixedTest = (MixedItemList([imgListTest, tabListTest, textListTest], path, inner_df=tabListTest.inner_df)
            .split_from_df(col='IsTest')
            .label_from_df(cols='AdoptionSpeed', label_cls=CategoryList)
            .transform([[get_transforms()[0], [], []], [get_transforms()[1], [], []]], size=size))

#%%
# learn.lr_find()