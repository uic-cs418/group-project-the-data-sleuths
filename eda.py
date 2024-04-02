import pandas as pd
import numpy as np
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")


def filter_columns(df,cols):
    if(len(cols) > 0):
        df = df[cols]
    return df

def filter_rows(df):
    df = df[(df['CATEGORY']=='species') & (df['PROTOCOL TYPE']=='Traveling') | (df['PROTOCOL TYPE']=='Stationary') & (df['ALL SPECIES REPORTED']==1)]
    return df

# Native column: 1 = is native to chicago, 0 = not native to chicago
def exotic_trans(row):
    if row == np.nan:
        return 0
    else:
        return 1

def obs_trans(row):
    if row == 'X':
        return 1
    else:
        return row