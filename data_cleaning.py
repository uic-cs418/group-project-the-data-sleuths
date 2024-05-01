
import eda, ml, pandas as pd, numpy as np, geopandas as gpd, warnings, textwrap; warnings.filterwarnings("ignore")

def create_cook_county_dataset():
    dataset_files = ['xaa','xab','xac','xad','xae','xaf','xag','xah','xai','xaj','xak','xal','xam','xan','xao','xap','xaq','xar','xas','xat','xau','xav','xaw','xax','xay','xaz','xba','xbb','xbc']
    cook_county_df = pd.DataFrame()
    for file in dataset_files:
        df = pd.read_csv('data/{file}'.format(file=file),sep='\t')
        df = df[df["COUNTY"].str.lower() == "cook"]
        cook_county_df = pd.concat([cook_county_df,df])
    cook_county_df.to_csv('data/ebd_cook_county.tsv', index=False,sep='\t')

def filter_dataset_columns():
    cook_county_df = pd.read_csv('data/ebd_cook_county.tsv',sep='\t')
    df = cook_county_df
    # Filter columns
    req_cols = ['CATEGORY', 'COMMON NAME', 'SCIENTIFIC NAME', 'OBSERVATION COUNT', 'EXOTIC CODE', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE', 'PROTOCOL TYPE', 'ALL SPECIES REPORTED']
    df = df[req_cols]
    # We will keep only species level observations (removing subspecies and genus level observations). We will also filter out incomplete checklists and incidental observations to manage bias towards specific species.
    df = df[(df['CATEGORY']=='species') & (df['PROTOCOL TYPE']=='Traveling') | (df['PROTOCOL TYPE']=='Stationary') & (df['ALL SPECIES REPORTED']==1)]
    return df

def data_transformation(df):
    df['NATIVE'] = df['EXOTIC CODE'].apply(lambda row:0 if row == np.nan else 1) # Native column: 1 = is native to chicago, 0 = not native to chicago
    df['COUNT'] = df['OBSERVATION COUNT'].apply(lambda row: 1 if row == 'X' else row) # Assume all 'X' observations have a count of 1 bird
    req_cols = ['COMMON NAME', 'SCIENTIFIC NAME', 'NATIVE', 'COUNT', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE']
    df = df[req_cols] # remove unnecessary columns
    return df

def agg_neighborhood(df):
    com_areas = gpd.read_file('data/neighborhoods/geo_export_f5325bf0-9c6d-49a5-a5d9-0e5bf24fa856.shp')
    ebird_gdf = eda.join_datasets(df,com_areas)
    return ebird_gdf

def save_final_dataset(ebird_gdf):
    ebird_gdf = eda.modify_ebird_dataset(ebird_gdf)
    ebird_gdf.to_csv("data/final_dataset.tsv",index=False,sep='\t')