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


community_location_map = {"Albany Park": "Far North Side",
"Archer Heights": "Southwest Side",
"Armour Square": "South Side",
"Ashburn": "Far Southwest Side",
"Auburn Gresham": "Far Southwest Side",
"Austin": "West Side",
"Avalon Park": "Far Southeast Side",
"Avondale": "North Side",
"Belmont Cragin": "Northwest Side",
"Beverly": "Far Southwest Side",
"Bridgeport": "South Side",
"Brighton Park": "Southwest Side",
"Burnside": "Far Southeast Side",
"Calumet Heights": "Far Southeast Side",
"Chatham": "Far Southeast Side",
"Chicago Lawn": "Southwest Side",
"Clearing": "Southwest Side",
"Douglas": "South Side",
"Dunning": "Northwest Side",
"East Garfield Park": "West Side",
"East Side": "Far Southeast Side",
"Edgewater": "Far North Side",
"Edison Park": "Far North Side",
"Englewood": "Southwest Side",
"Forest Glen": "Far North Side",
"Fuller Park": "South Side",
"Gage Park": "Southwest Side",
"Garfield Ridge": "Southwest Side",
"Grand Boulevard": "South Side",
"Greater Grand Crossing": "South Side",
"Hegewisch": "Far Southeast Side",
"Hermosa": "Northwest Side",
"Hyde Park": "South Side",
"Irving Park": "Northwest Side",
"Jefferson Park": "Far North Side",
"Kenwood": "South Side",
"Lake View": "North Side",
"Lincoln Park": "North Side",
"Lincoln Square": "Far North Side",
"Logan Square": "Far North Side",
"Loop": "Central",
"Lower West Side": "West Side",
"McKinley Park": "Southwest Side",
"Montclare": "Northwest Side",
"Morgan Park": "Far Southwest Side",
"Mount Greenwood": "Far Southwest Side",
"Near North Side": "Central",
"Near South Side": "Central",
"Near West Side": "West Side",
"New City": "Southwest Side",
"North Center": "North Side",
"North Lawndale": "West Side",
"North Park": "Far North Side",
"Norwood Park": "Far North Side",
"Oakland": "South Side",
"O'Hare": "Far North Side",
"Portage Park": "Northwest Side",
"Pullman": "Far Southeast Side",
"Riverdale": "Far Southeast Side",
"Rogers Park": "Far North Side",
"Roseland": "Far Southeast Side",
"South Chicago": "Far Southeast Side",
"South Deering": "Far Southeast Side",
"South Lawndale": "West Side",
"South Shore": "South Side",
"Uptown": "Far North Side",
"Washington Heights": "Far Southwest Side",
"Washington Park": "South Side",
"West Elsdon": "Southwest Side",
"West Englewood": "Southwest Side",
"West Garfield Park": "West Side",
"West Lawn": "Southwest Side",
"West Pullman": "Far Southeast Side",
"West Ridge": "Far North Side",
"West Town": "West Side",
"Woodlawn": "South Side",
"Wrightwood": "Far Southwest Side"}