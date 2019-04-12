import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import feather
from pathlib import Path
from pandas.io.json import json_normalize
from tqdm import tqdm

__all__ = ['get_data', 'quadratic_weighted_kappa']

def get_data(isTest:bool=False, useMetadata:bool=False):
    name = 'train'
    if(isTest):
        name = 'test'

    p = Path('.')

    petsFeather = 'pets_' + name + '.feather'
    if os.path.isfile(petsFeather) != True:
        pets = pd.read_csv(name + '\\' + name + '.csv')

        pImages = p / (name + '_images')
        pSentiments = p / (name + '_sentiment')

        images = [x for x in pImages.iterdir()]
        images = pd.DataFrame([x for x in map(lambda x: (x.name.split('.')[0].split('-')[0], x.name), images)], columns=['PetID', 'Filename'])

        petsImages = pd.merge(pets, images, how='left', on='PetID')

        petsImages['NoImage'] = petsImages['Filename'].isna()
        petsImages['Filename'] = petsImages['Filename'].fillna('..\\NoImage.jpg')

        byRescuerCount = pets.groupby(['RescuerID']).PetID.nunique().reset_index().rename({'PetID': 'RescuerDogCount'}, axis=1)
        petsImages = pd.merge(petsImages, byRescuerCount, how='left', on='RescuerID')

        cat = ['Type', 'Name', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'RescuerID']
        cont = ['Age', 'Fee', 'Quantity', 'RescuerDogCount', 'VideoAmt', 'PhotoAmt']
        for x in cat:
            petsImages[x] = petsImages[x].astype('category')
        for x in cont:
            petsImages[x] = petsImages[x].astype('float')
            
        petsImages['PicturePath'] = petsImages.apply(lambda x: str(name + '_images\\' + x['Filename']), axis=1)
        sentimentJsons = [x for x in pSentiments.iterdir()]

        petsSentiments = pd.DataFrame()
        sentiment_feather = name + '_sentiments.feather'
        if os.path.isfile(sentiment_feather) != True:
            for s in tqdm(sentimentJsons, desc='Sentiments'):
                with open(s, encoding='utf8') as json_data:
                    d = json.load(json_data)
                    df = json_normalize(d['sentences'])
                    line = {}
                    m = df.mean().to_dict()
                    line['PetID'] = s.name.split('.')[0]
                    line['AvgSentenceSentimentMagnitude'] = m['sentiment.magnitude']
                    line['AvgSentenceSentimentScore'] = m['sentiment.score']
                    line['SentimentMagnitude'] = d['documentSentiment']['magnitude']
                    line['SentimentScore'] = d['documentSentiment']['score']
                    petsSentiments = petsSentiments.append(line, ignore_index=True)

            petsSentiments = petsSentiments.reset_index(drop=True)
            petsSentiments.to_feather(sentiment_feather)
        else:
            petsSentiments = feather.read_dataframe(sentiment_feather)

        pets = pd.merge(petsImages, petsSentiments, how='left', on='PetID')

        if(useMetadata):
            petsMetadata = pd.DataFrame()
            meta_feather = name + '_metadata.feather'
            if os.path.isfile(meta_feather) != True:
                pMetadata = p / (name + '_metadata')
                metadataJsons = [x for x in pMetadata.iterdir()]

                lst = []
                errors = []
                for s in tqdm(metadataJsons, desc='Metadata'):
                    with open(s, encoding='utf8') as json_data:
                        try:
                            d = json.load(json_data)
                            df = json_normalize(d['labelAnnotations'])
                            df = df.set_index('description').T
                            df['PetID'] = s.name.split('-')[0]
                            lst.append(df.loc['score'].to_dict())
                        except:
                            errors.append(s.name)
                petsMetadata = pd.DataFrame(lst)
                petsMetadata = petsMetadata.groupby('PetID').mean()
                petsMetadata = petsMetadata.fillna(0)

                petsMetadata = petsMetadata.reset_index()
                petsMetadata.to_feather(meta_feather)
            else:
                petsMetadata = feather.read_dataframe(meta_feather)

            pets = pd.merge(pets, petsMetadata, how='left', on='PetID')

        petsImages['NoDescription'] = petsImages['Description'].isna()
        pets['Description'] = pets['Description'].fillna('No description')

        pets = pets.reset_index(drop=True)

        pets.to_feather(petsFeather)

        return pets
    else:
        pets = feather.read_dataframe(petsFeather)

        return pets

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = Cmatrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)