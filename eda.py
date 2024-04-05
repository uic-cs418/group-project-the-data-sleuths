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

### Shannon index by year and month
def shannon_index(species_abundance):
    total_count = sum(species_abundance)
    proportions = [count / total_count for count in species_abundance]
    shannon_index = -sum(p * np.log(p) for p in proportions if p != 0)
    return shannon_index