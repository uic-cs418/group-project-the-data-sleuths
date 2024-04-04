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
<<<<<<< HEAD
    return shannon_index

def distribution(df,col):
    df_new = df.groupby(col)[col].size().reset_index(name='Size')
    df_new['Percent'] = (df_new['Size'] / df_new['Size'].sum()) * 100
    df_new = df_new.sort_values(by="Percent",ascending=False)
    print("\033[1m{} distribution\033[0m\n".format(col))
    print(df_new)
    print("\n\n")
=======
    return shannon_index
>>>>>>> abeb75099dee86ea4162e81b07d506264f2bda99
