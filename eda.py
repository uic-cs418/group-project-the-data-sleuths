import pandas as pd
import numpy as np
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore")

from shapely.geometry import Point


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

def distribution(df,col):
    df_new = df.groupby(col)[col].size().reset_index(name='Size')
    df_new['Percent'] = (df_new['Size'] / df_new['Size'].sum()) * 100
    df_new = df_new.sort_values(by="Percent",ascending=False)
    print("\033[1m{} distribution\033[0m\n".format(col))
    print(df_new)
    print("\n\n")
    return df_new

def monthly_distribution(df):
    monthdf = df.groupby(['OBSERVATION YEAR','OBSERVATION MONTH']).size().reset_index(name='count')
    avg_monthdf = monthdf.groupby(['OBSERVATION MONTH'])['count'].mean().reset_index(name='count')
    avg_monthdf = avg_monthdf.sort_values(by='count',ascending=False)
    print("\033[1m{} distribution\033[0m\n".format("Monthly"))
    print(avg_monthdf)
    print("\n\n")

def join_datasets(df,com_areas):
    com_areas = com_areas[['community','geometry']]
    geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)
    ebird_gdf = gpd.sjoin(geo_df, com_areas, how='left', op='within')
    return ebird_gdf

def agg_comm(series):
    return list(series)

def aggregate_data(df,cols):
    grouped = df.groupby(cols).agg(agg_comm).reset_index()
    return grouped

def modify_ebird_dataset(ebird_gdf):
    ebird_gdf['OBSERVATION DATE'] = pd.to_datetime(ebird_gdf['OBSERVATION DATE'])
    ebird_gdf['OBSERVATION MONTH'] =  ebird_gdf['OBSERVATION DATE'].apply(lambda x:x.month)
    ebird_gdf['OBSERVATION DAY'] =  ebird_gdf['OBSERVATION DATE'].apply(lambda x:x.day)
    ebird_gdf['OBSERVATION YEAR'] =  ebird_gdf['OBSERVATION DATE'].apply(lambda x:x.year)
    ebird_gdf = ebird_gdf.drop(columns=["index_right","LATITUDE","LONGITUDE"])
    ebird_gdf['COUNT'] = ebird_gdf['COUNT'].astype(int)
    return ebird_gdf

def categorize_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'