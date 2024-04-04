{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b05252f6-aef3-4546-819d-968dd35accf4",
   "metadata": {},
   "source": [
    "# Chicago Luxury Effect"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "822f602d-f370-41ee-bada-4a4dc033a902",
   "metadata": {},
   "source": [
    "### Sections\n",
    "#### 1. Project introduction\n",
    "#### 2. Data cleaning\n",
    "#### 3. Exploratory data analysis\n",
    "#### 4. Vizualizations\n",
    "#### 5. ML analysis\n",
    "#### 6. Reflection"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ecd466f5-0b88-41ce-aedf-05861b386af6",
   "metadata": {},
   "source": [
    "## Section 1. Project Introduction\n",
    "This project explores the relationship between socio-economic indicators and bird diversity in Chicago communities. Utilizing a dataset containing metrics such as housing conditions, poverty rates, and per capita income alongside bird diversity measures, we aim to investigate whether affluent neighborhoods exhibit higher bird diversity."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f41283fa-f44e-4960-b1bd-ed46c0eb2ac1",
   "metadata": {},
   "source": [
    "## Section 2. Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1a44ee38-bb94-41af-86f6-1c02ddba63f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import geopandas as gpd\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "from eda import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8533c7ed-0c4f-4145-a35c-aafa7b3b5625",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('data/ebd_US-IL_200801_201212_relJan-2024.txt',sep='\\t')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "970fd878-679a-4b09-a0b5-da28031e918b",
   "metadata": {},
   "source": [
    "### 2.1 Filtering eBird Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "547f421c-59fa-4b1e-9eaf-881dc3c9d8c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filter columns\n",
    "req_cols = ['CATEGORY', 'COMMON NAME', 'SCIENTIFIC NAME', 'OBSERVATION COUNT', 'EXOTIC CODE', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE', 'PROTOCOL TYPE', 'ALL SPECIES REPORTED']\n",
    "df = df[req_cols]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9e0b63a3-4759-4b9f-a4fb-c8e4d2db5d53",
   "metadata": {},
   "source": [
    "We will keep only species level observations (removing subspecies and genus level observations). We will also filter out incomplete checklists and incidental observations to manage bias towards specific species."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "20a591d8-bcb4-47b4-8bdc-22a2358fe6ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[(df['CATEGORY']=='species') & (df['PROTOCOL TYPE']=='Traveling') | (df['PROTOCOL TYPE']=='Stationary') & (df['ALL SPECIES REPORTED']==1)]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d93eae8-b16f-4e5f-bcab-418418f7941f",
   "metadata": {},
   "source": [
    "### 2.2 ebird Dataset Transformation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3ff8efc8-bf5c-4292-8527-638e414913bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Native column: 1 = is native to chicago, 0 = not native to chicago\n",
    "df['NATIVE'] = df['EXOTIC CODE'].apply(lambda row:0 if row == np.nan else 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e18901f7-9015-45a4-a6f3-27a7caabc10e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assume all 'X' observations have a count of 1 bird\n",
    "df['COUNT'] = df['OBSERVATION COUNT'].apply(lambda row: 1 if row == 'X' else row)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f0e32455-bd5b-475d-b318-359218437479",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>COMMON NAME</th>\n",
       "      <th>SCIENTIFIC NAME</th>\n",
       "      <th>NATIVE</th>\n",
       "      <th>COUNT</th>\n",
       "      <th>LATITUDE</th>\n",
       "      <th>LONGITUDE</th>\n",
       "      <th>OBSERVATION DATE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>American Crow</td>\n",
       "      <td>Corvus brachyrhynchos</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>American Goldfinch</td>\n",
       "      <td>Spinus tristis</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>American Kestrel</td>\n",
       "      <td>Falco sparverius</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Bald Eagle</td>\n",
       "      <td>Haliaeetus leucocephalus</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Blue Jay</td>\n",
       "      <td>Cyanocitta cristata</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          COMMON NAME           SCIENTIFIC NAME  NATIVE COUNT   LATITUDE  \\\n",
       "1       American Crow     Corvus brachyrhynchos       1     1  38.850907   \n",
       "2  American Goldfinch            Spinus tristis       1     1  38.850907   \n",
       "3    American Kestrel          Falco sparverius       1     1  38.850907   \n",
       "4          Bald Eagle  Haliaeetus leucocephalus       1     1  38.850907   \n",
       "5            Blue Jay       Cyanocitta cristata       1     1  38.850907   \n",
       "\n",
       "   LONGITUDE OBSERVATION DATE  \n",
       "1 -89.256706       2008-01-01  \n",
       "2 -89.256706       2008-01-01  \n",
       "3 -89.256706       2008-01-01  \n",
       "4 -89.256706       2008-01-01  \n",
       "5 -89.256706       2008-01-01  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# remove unnecessary columns\n",
    "req_cols = ['COMMON NAME', 'SCIENTIFIC NAME', 'NATIVE', 'COUNT', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE']\n",
    "df = filter_columns(df,req_cols)\n",
    "df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8399ed44-de96-4c08-9a0b-30f56ed29e6e",
   "metadata": {},
   "source": [
    "### 2.3 Aggregate eBird data based on neighborhood"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "81890d20-3813-4c9b-bd1b-5c1ddbceadab",
   "metadata": {},
   "outputs": [],
   "source": [
    "com_areas = gpd.read_file('data/neighborhoods/geo_export_f5325bf0-9c6d-49a5-a5d9-0e5bf24fa856.shp')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "52308051-69d2-4953-bac8-2535c7e0b856",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>community</th>\n",
       "      <th>geometry</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>DOUGLAS</td>\n",
       "      <td>POLYGON ((-87.60914 41.84469, -87.60915 41.844...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>OAKLAND</td>\n",
       "      <td>POLYGON ((-87.59215 41.81693, -87.59231 41.816...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>FULLER PARK</td>\n",
       "      <td>POLYGON ((-87.62880 41.80189, -87.62879 41.801...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>GRAND BOULEVARD</td>\n",
       "      <td>POLYGON ((-87.60671 41.81681, -87.60670 41.816...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>KENWOOD</td>\n",
       "      <td>POLYGON ((-87.59215 41.81693, -87.59215 41.816...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         community                                           geometry\n",
       "0          DOUGLAS  POLYGON ((-87.60914 41.84469, -87.60915 41.844...\n",
       "1          OAKLAND  POLYGON ((-87.59215 41.81693, -87.59231 41.816...\n",
       "2      FULLER PARK  POLYGON ((-87.62880 41.80189, -87.62879 41.801...\n",
       "3  GRAND BOULEVARD  POLYGON ((-87.60671 41.81681, -87.60670 41.816...\n",
       "4          KENWOOD  POLYGON ((-87.59215 41.81693, -87.59215 41.816..."
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "com_areas = filter_columns(com_areas,['community','geometry'])\n",
    "com_areas.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8721706a-7b2a-4e02-bc82-d678470dd46f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>COMMON NAME</th>\n",
       "      <th>SCIENTIFIC NAME</th>\n",
       "      <th>NATIVE</th>\n",
       "      <th>COUNT</th>\n",
       "      <th>LATITUDE</th>\n",
       "      <th>LONGITUDE</th>\n",
       "      <th>OBSERVATION DATE</th>\n",
       "      <th>geometry</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>American Crow</td>\n",
       "      <td>Corvus brachyrhynchos</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>American Goldfinch</td>\n",
       "      <td>Spinus tristis</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>American Kestrel</td>\n",
       "      <td>Falco sparverius</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Bald Eagle</td>\n",
       "      <td>Haliaeetus leucocephalus</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Blue Jay</td>\n",
       "      <td>Cyanocitta cristata</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          COMMON NAME           SCIENTIFIC NAME  NATIVE COUNT   LATITUDE  \\\n",
       "1       American Crow     Corvus brachyrhynchos       1     1  38.850907   \n",
       "2  American Goldfinch            Spinus tristis       1     1  38.850907   \n",
       "3    American Kestrel          Falco sparverius       1     1  38.850907   \n",
       "4          Bald Eagle  Haliaeetus leucocephalus       1     1  38.850907   \n",
       "5            Blue Jay       Cyanocitta cristata       1     1  38.850907   \n",
       "\n",
       "   LONGITUDE OBSERVATION DATE                    geometry  \n",
       "1 -89.256706       2008-01-01  POINT (-89.25671 38.85091)  \n",
       "2 -89.256706       2008-01-01  POINT (-89.25671 38.85091)  \n",
       "3 -89.256706       2008-01-01  POINT (-89.25671 38.85091)  \n",
       "4 -89.256706       2008-01-01  POINT (-89.25671 38.85091)  \n",
       "5 -89.256706       2008-01-01  POINT (-89.25671 38.85091)  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from shapely.geometry import Point\n",
    "\n",
    "geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]\n",
    "geo_df = gpd.GeoDataFrame(df, geometry=geometry)\n",
    "geo_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a6aa46cf-f005-4c0d-9db5-b34a8a8e4362",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>COMMON NAME</th>\n",
       "      <th>SCIENTIFIC NAME</th>\n",
       "      <th>NATIVE</th>\n",
       "      <th>COUNT</th>\n",
       "      <th>LATITUDE</th>\n",
       "      <th>LONGITUDE</th>\n",
       "      <th>OBSERVATION DATE</th>\n",
       "      <th>geometry</th>\n",
       "      <th>index_right</th>\n",
       "      <th>community</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>American Crow</td>\n",
       "      <td>Corvus brachyrhynchos</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>American Goldfinch</td>\n",
       "      <td>Spinus tristis</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>American Kestrel</td>\n",
       "      <td>Falco sparverius</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Bald Eagle</td>\n",
       "      <td>Haliaeetus leucocephalus</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Blue Jay</td>\n",
       "      <td>Cyanocitta cristata</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.850907</td>\n",
       "      <td>-89.256706</td>\n",
       "      <td>2008-01-01</td>\n",
       "      <td>POINT (-89.25671 38.85091)</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          COMMON NAME           SCIENTIFIC NAME  NATIVE COUNT   LATITUDE  \\\n",
       "1       American Crow     Corvus brachyrhynchos       1     1  38.850907   \n",
       "2  American Goldfinch            Spinus tristis       1     1  38.850907   \n",
       "3    American Kestrel          Falco sparverius       1     1  38.850907   \n",
       "4          Bald Eagle  Haliaeetus leucocephalus       1     1  38.850907   \n",
       "5            Blue Jay       Cyanocitta cristata       1     1  38.850907   \n",
       "\n",
       "   LONGITUDE OBSERVATION DATE                    geometry  index_right  \\\n",
       "1 -89.256706       2008-01-01  POINT (-89.25671 38.85091)          NaN   \n",
       "2 -89.256706       2008-01-01  POINT (-89.25671 38.85091)          NaN   \n",
       "3 -89.256706       2008-01-01  POINT (-89.25671 38.85091)          NaN   \n",
       "4 -89.256706       2008-01-01  POINT (-89.25671 38.85091)          NaN   \n",
       "5 -89.256706       2008-01-01  POINT (-89.25671 38.85091)          NaN   \n",
       "\n",
       "  community  \n",
       "1       NaN  \n",
       "2       NaN  \n",
       "3       NaN  \n",
       "4       NaN  \n",
       "5       NaN  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ebird_gdf = gpd.sjoin(geo_df, com_areas, how='left', op='within')\n",
    "ebird_gdf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "e728d162-de74-4b41-b196-2ea45dc2088f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# remove observations outside of Chicago and also remove more unnecessary columns\n",
    "ebird_gdf = ebird_gdf[ebird_gdf['community'].notna()]\n",
    "ebird_gdf = ebird_gdf.drop(columns=[\"index_right\",\"LATITUDE\",\"LONGITUDE\",\"OBSERVATION DATE\", \"COMMON NAME\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "5d478667-7ff1-42b7-a9e1-9d11179d703d",
   "metadata": {},
   "outputs": [],
   "source": [
    "ebird_gdf['COUNT'] = ebird_gdf['COUNT'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "97b53756-9b5a-43e7-a33a-523c9793eb24",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# aggregate \n",
    "def agg_comm(series):\n",
    "    return list(series)\n",
    "\n",
    "grouped = ebird_gdf.groupby('community').agg(agg_comm)\n",
    "grouped['NATIVE'] = grouped['NATIVE'].apply(lambda x: sum(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "c85021d3-1267-4a89-be5e-f0bc5cb3cce2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>SCIENTIFIC NAME</th>\n",
       "      <th>NATIVE</th>\n",
       "      <th>COUNT</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>community</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ALBANY PARK</th>\n",
       "      <td>[Spinus tristis, Megaceryle alcyon, Cyanocitta...</td>\n",
       "      <td>1281</td>\n",
       "      <td>[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ARMOUR SQUARE</th>\n",
       "      <td>[Turdus migratorius, Branta canadensis, Aegoli...</td>\n",
       "      <td>35</td>\n",
       "      <td>[1, 1, 1, 1, 6, 3, 1, 1, 5, 2, 5, 1, 1, 1, 1, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AUSTIN</th>\n",
       "      <td>[Branta canadensis, Branta canadensis, Dryobat...</td>\n",
       "      <td>16232</td>\n",
       "      <td>[375, 380, 1, 19, 2, 1, 6, 7, 1, 1, 48, 6, 1, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AVONDALE</th>\n",
       "      <td>[Corvus brachyrhynchos, Branta canadensis, Buc...</td>\n",
       "      <td>97</td>\n",
       "      <td>[1, 45, 1, 12, 1, 1, 7, 3, 1, 1, 2, 34, 1, 77,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>BELMONT CRAGIN</th>\n",
       "      <td>[Setophaga ruticilla, Setophaga ruticilla, Tur...</td>\n",
       "      <td>123</td>\n",
       "      <td>[2, 2, 10, 10, 1, 1, 2, 2, 1, 1, 2, 2, 10, 10,...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                  SCIENTIFIC NAME  NATIVE  \\\n",
       "community                                                                   \n",
       "ALBANY PARK     [Spinus tristis, Megaceryle alcyon, Cyanocitta...    1281   \n",
       "ARMOUR SQUARE   [Turdus migratorius, Branta canadensis, Aegoli...      35   \n",
       "AUSTIN          [Branta canadensis, Branta canadensis, Dryobat...   16232   \n",
       "AVONDALE        [Corvus brachyrhynchos, Branta canadensis, Buc...      97   \n",
       "BELMONT CRAGIN  [Setophaga ruticilla, Setophaga ruticilla, Tur...     123   \n",
       "\n",
       "                                                            COUNT  \n",
       "community                                                          \n",
       "ALBANY PARK     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, ...  \n",
       "ARMOUR SQUARE   [1, 1, 1, 1, 6, 3, 1, 1, 5, 2, 5, 1, 1, 1, 1, ...  \n",
       "AUSTIN          [375, 380, 1, 19, 2, 1, 6, 7, 1, 1, 48, 6, 1, ...  \n",
       "AVONDALE        [1, 45, 1, 12, 1, 1, 7, 3, 1, 1, 2, 34, 1, 77,...  \n",
       "BELMONT CRAGIN  [2, 2, 10, 10, 1, 1, 2, 2, 1, 1, 2, 2, 10, 10,...  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grouped = grouped.drop(columns=['geometry'])\n",
    "grouped.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a147ad7-284f-45db-b083-6a5ecfafcee5",
   "metadata": {},
   "source": [
    "### 2.4 Join eBird and census datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "6535c2fe-df18-420e-bbe4-57e73be99067",
   "metadata": {},
   "outputs": [],
   "source": [
    "census_df = pd.read_csv(\"data/Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012_20240228.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "615bc596-f97f-4833-8319-efe66597aad0",
   "metadata": {},
   "outputs": [],
   "source": [
    "census_df['COMMUNITY AREA NAME'] = census_df['COMMUNITY AREA NAME'].str.upper()\n",
    "census_df = census_df.rename(columns={'PER CAPITA INCOME ': 'PER CAPITA INCOME'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "f55a6f31-61e4-4811-9598-a561cc4365db",
   "metadata": {},
   "outputs": [],
   "source": [
    "final_df = census_df.merge(grouped, left_on='COMMUNITY AREA NAME', right_on='community')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f51cca52-e460-4f97-85e9-8a855e412049",
   "metadata": {},
   "source": [
    "### 2.5 Transform dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "bd27982e-6ae4-4a91-9e00-1b1ac3b26e2d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Community Area Number</th>\n",
       "      <th>COMMUNITY AREA NAME</th>\n",
       "      <th>PERCENT OF HOUSING CROWDED</th>\n",
       "      <th>PERCENT HOUSEHOLDS BELOW POVERTY</th>\n",
       "      <th>PERCENT AGED 16+ UNEMPLOYED</th>\n",
       "      <th>PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA</th>\n",
       "      <th>PERCENT AGED UNDER 18 OR OVER 64</th>\n",
       "      <th>PER CAPITA INCOME</th>\n",
       "      <th>HARDSHIP INDEX</th>\n",
       "      <th>SCIENTIFIC NAME</th>\n",
       "      <th>NATIVE</th>\n",
       "      <th>COUNT</th>\n",
       "      <th>PER CAPITA INCOME IN K</th>\n",
       "      <th>PovertyFlag</th>\n",
       "      <th>shannon_index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.0</td>\n",
       "      <td>ROGERS PARK</td>\n",
       "      <td>7.7</td>\n",
       "      <td>23.6</td>\n",
       "      <td>8.7</td>\n",
       "      <td>18.2</td>\n",
       "      <td>27.5</td>\n",
       "      <td>23939</td>\n",
       "      <td>39.0</td>\n",
       "      <td>[Spinus tristis, Turdus migratorius, Turdus mi...</td>\n",
       "      <td>676</td>\n",
       "      <td>[1, 46, 4, 9, 1, 3, 1, 9, 12, 1, 2, 1, 2, 1, 7...</td>\n",
       "      <td>23.939</td>\n",
       "      <td>Poor</td>\n",
       "      <td>5.498251</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.0</td>\n",
       "      <td>WEST RIDGE</td>\n",
       "      <td>7.8</td>\n",
       "      <td>17.2</td>\n",
       "      <td>8.8</td>\n",
       "      <td>20.8</td>\n",
       "      <td>38.5</td>\n",
       "      <td>23040</td>\n",
       "      <td>46.0</td>\n",
       "      <td>[Sturnus vulgaris, Aquila chrysaetos, Passer d...</td>\n",
       "      <td>255</td>\n",
       "      <td>[7, 1, 5, 4, 100, 2, 2, 2, 1, 1, 1, 25, 25, 25...</td>\n",
       "      <td>23.040</td>\n",
       "      <td>Poor</td>\n",
       "      <td>4.148305</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Community Area Number COMMUNITY AREA NAME  PERCENT OF HOUSING CROWDED  \\\n",
       "0                    1.0         ROGERS PARK                         7.7   \n",
       "1                    2.0          WEST RIDGE                         7.8   \n",
       "\n",
       "   PERCENT HOUSEHOLDS BELOW POVERTY  PERCENT AGED 16+ UNEMPLOYED  \\\n",
       "0                              23.6                          8.7   \n",
       "1                              17.2                          8.8   \n",
       "\n",
       "   PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA  \\\n",
       "0                                          18.2   \n",
       "1                                          20.8   \n",
       "\n",
       "   PERCENT AGED UNDER 18 OR OVER 64  PER CAPITA INCOME  HARDSHIP INDEX  \\\n",
       "0                              27.5              23939            39.0   \n",
       "1                              38.5              23040            46.0   \n",
       "\n",
       "                                     SCIENTIFIC NAME  NATIVE  \\\n",
       "0  [Spinus tristis, Turdus migratorius, Turdus mi...     676   \n",
       "1  [Sturnus vulgaris, Aquila chrysaetos, Passer d...     255   \n",
       "\n",
       "                                               COUNT  PER CAPITA INCOME IN K  \\\n",
       "0  [1, 46, 4, 9, 1, 3, 1, 9, 12, 1, 2, 1, 2, 1, 7...                  23.939   \n",
       "1  [7, 1, 5, 4, 100, 2, 2, 2, 1, 1, 1, 25, 25, 25...                  23.040   \n",
       "\n",
       "  PovertyFlag  shannon_index  \n",
       "0        Poor       5.498251  \n",
       "1        Poor       4.148305  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_df[\"PER CAPITA INCOME IN K\"] = final_df.apply(lambda x: x[\"PER CAPITA INCOME\"] / 1000, axis=1)\n",
    "final_df[\"PovertyFlag\"] = final_df.apply(lambda x: \"Poor\" if x[\"PER CAPITA INCOME\"] < 40000 else \"Rich\", axis=1)\n",
    "final_df[\"shannon_index\"] = final_df[\"COUNT\"].apply(shannon_index)\n",
    "final_df.head(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "016cb014-bec9-44b4-a3dc-5eb247852cce",
   "metadata": {},
   "source": [
    "## Section 3: EDA"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f5c3d12-626e-4d1e-9b4d-605ab6b7a3a2",
   "metadata": {},
   "source": [
    "## Section 4: Vizualizations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "79aef352-51c0-4f63-b140-88fe0581107c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.patches as mpatches\n",
    "import seaborn as sns\n",
    "import plotly.express as px\n",
    "import altair as alt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8d1252e2-5046-4f3b-816a-6239f0ca2111",
   "metadata": {},
   "source": [
    "\n",
    "## 4.1 (_Does Income play a role in bird diversity of a community?_)\n",
    "\n",
    "(_The graph depicts bird diversity across Chicago's communities, revealing a correlation between higher per capita income and increased bird diversity, while also highlighting the fluctuating levels of diversity within low-income neighborhoods._)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "00723616-2a72-420e-8b5f-d437ab317c31",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "customdata": [
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ],
          [
           "Poor"
          ]
         ],
         "hovertemplate": "=%{customdata[0]}<br>Per Capita Income Of Community (in K)=%{x}<br>Shannon Index (Diversity)=%{y}<extra></extra>",
         "legendgroup": "Poor",
         "marker": {
          "color": "#00cc96",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Poor",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          23.939,
          23.04,
          35.787,
          37.524,
          32.875,
          27.751,
          26.576,
          21.323,
          24.336,
          27.249,
          26.282,
          15.461,
          20.039,
          31.908,
          13.781,
          15.957,
          12.961,
          12.034,
          10.402,
          16.444,
          16.148,
          23.791,
          19.252,
          35.911,
          13.785,
          39.056,
          18.672,
          19.398,
          20.588,
          14.685,
          17.104,
          16.563,
          8.201,
          22.677,
          26.353,
          16.954,
          22.694,
          12.765,
          25.113,
          17.285,
          39.523,
          34.381,
          33.385
         ],
         "xaxis": "x",
         "y": [
          5.498250890741741,
          4.148304992713281,
          9.626278060658764,
          7.116321256137552,
          3.9153514486576757,
          3.2306552081901403,
          7.685654440615552,
          6.303148587128158,
          3.2199066974014308,
          7.564594567988713,
          1.983204953102925,
          4.313047762239837,
          3.6939685760916166,
          5.369094335565102,
          1.0986122886681096,
          8.175104203683162,
          4.815892878058631,
          7.42385123421606,
          2.2809369026289956,
          4.07194649913427,
          2.6530793717777166,
          5.947795013347923,
          5.0508446101811035,
          5.341765777619535,
          2.4499119632014286,
          7.572882955586853,
          8.057085031441177,
          6.132057244471824,
          2.364846178214263,
          7.5815977952628755,
          6.565295609119809,
          6.265919106847263,
          7.056429149637764,
          8.551901564638195,
          1.0397207708399179,
          5.678240105185059,
          4.254697773361497,
          2.233261435925072,
          2.3693821196946767,
          1.040839837423239,
          4.925730798652334,
          4.33952307102356,
          2.470602399227253
         ],
         "yaxis": "y"
        },
        {
         "customdata": [
          [
           "Rich"
          ],
          [
           "Rich"
          ],
          [
           "Rich"
          ],
          [
           "Rich"
          ],
          [
           "Rich"
          ],
          [
           "Rich"
          ],
          [
           "Rich"
          ],
          [
           "Rich"
          ],
          [
           "Rich"
          ]
         ],
         "hovertemplate": "=%{customdata[0]}<br>Per Capita Income Of Community (in K)=%{x}<br>Shannon Index (Diversity)=%{y}<extra></extra>",
         "legendgroup": "Rich",
         "marker": {
          "color": "#ab63fa",
          "symbol": "circle"
         },
         "mode": "markers",
         "name": "Rich",
         "orientation": "v",
         "showlegend": true,
         "type": "scatter",
         "x": [
          57.123,
          60.058,
          71.551,
          88.669,
          44.164,
          43.198,
          44.689,
          65.526,
          59.077
         ],
         "xaxis": "x",
         "y": [
          6.540096699830579,
          7.465086829778779,
          9.101899831523921,
          5.314265960002228,
          5.778297523841608,
          6.729630810771817,
          6.537283453679603,
          6.839295885282258,
          8.841776022409961
         ],
         "yaxis": "y"
        }
       ],
       "layout": {
        "height": 600,
        "legend": {
         "title": {
          "text": ""
         },
         "tracegroupgap": 0
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Income vs. Bird Diversity"
        },
        "width": 800,
        "xaxis": {
         "anchor": "y",
         "autorange": true,
         "domain": [
          0,
          1
         ],
         "range": [
          3.125460205739036,
          93.74453979426096
         ],
         "title": {
          "text": "Per Capita Income Of Community (in K)"
         },
         "type": "linear"
        },
        "yaxis": {
         "anchor": "x",
         "autorange": true,
         "domain": [
          0,
          1
         ],
         "range": [
          0.46612483852408393,
          10.199873992974599
         ],
         "title": {
          "text": "Shannon Index (Diversity)"
         },
         "type": "linear"
        }
       }
      },
      "image/png": "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",
      "text/html": [
       "<div>                            <div id=\"428c65e5-8a99-456c-b90c-f157b8245182\" class=\"plotly-graph-div\" style=\"height:600px; width:800px;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"428c65e5-8a99-456c-b90c-f157b8245182\")) {                    Plotly.newPlot(                        \"428c65e5-8a99-456c-b90c-f157b8245182\",                        [{\"customdata\":[[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"],[\"Poor\"]],\"hovertemplate\":\"=%{customdata[0]}<br>Per Capita Income Of Community (in K)=%{x}<br>Shannon Index (Diversity)=%{y}<extra></extra>\",\"legendgroup\":\"Poor\",\"marker\":{\"color\":\"#00cc96\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Poor\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[23.939,23.04,35.787,37.524,32.875,27.751,26.576,21.323,24.336,27.249,26.282,15.461,20.039,31.908,13.781,15.957,12.961,12.034,10.402,16.444,16.148,23.791,19.252,35.911,13.785,39.056,18.672,19.398,20.588,14.685,17.104,16.563,8.201,22.677,26.353,16.954,22.694,12.765,25.113,17.285,39.523,34.381,33.385],\"xaxis\":\"x\",\"y\":[5.498250890741741,4.148304992713281,9.626278060658764,7.116321256137552,3.9153514486576757,3.2306552081901403,7.685654440615552,6.303148587128158,3.2199066974014308,7.564594567988713,1.983204953102925,4.313047762239837,3.6939685760916166,5.369094335565102,1.0986122886681096,8.175104203683162,4.815892878058631,7.42385123421606,2.2809369026289956,4.07194649913427,2.6530793717777166,5.947795013347923,5.0508446101811035,5.341765777619535,2.4499119632014286,7.572882955586853,8.057085031441177,6.132057244471824,2.364846178214263,7.5815977952628755,6.565295609119809,6.265919106847263,7.056429149637764,8.551901564638195,1.0397207708399179,5.678240105185059,4.254697773361497,2.233261435925072,2.3693821196946767,1.040839837423239,4.925730798652334,4.33952307102356,2.470602399227253],\"yaxis\":\"y\",\"type\":\"scatter\"},{\"customdata\":[[\"Rich\"],[\"Rich\"],[\"Rich\"],[\"Rich\"],[\"Rich\"],[\"Rich\"],[\"Rich\"],[\"Rich\"],[\"Rich\"]],\"hovertemplate\":\"=%{customdata[0]}<br>Per Capita Income Of Community (in K)=%{x}<br>Shannon Index (Diversity)=%{y}<extra></extra>\",\"legendgroup\":\"Rich\",\"marker\":{\"color\":\"#ab63fa\",\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Rich\",\"orientation\":\"v\",\"showlegend\":true,\"x\":[57.123,60.058,71.551,88.669,44.164,43.198,44.689,65.526,59.077],\"xaxis\":\"x\",\"y\":[6.540096699830579,7.465086829778779,9.101899831523921,5.314265960002228,5.778297523841608,6.729630810771817,6.537283453679603,6.839295885282258,8.841776022409961],\"yaxis\":\"y\",\"type\":\"scatter\"}],                        {\"template\":{\"data\":{\"histogram2dcontour\":[{\"type\":\"histogram2dcontour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"choropleth\":[{\"type\":\"choropleth\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"histogram2d\":[{\"type\":\"histogram2d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmap\":[{\"type\":\"heatmap\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmapgl\":[{\"type\":\"heatmapgl\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"type\":\"contourcarpet\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"contour\":[{\"type\":\"contour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"surface\":[{\"type\":\"surface\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"mesh3d\":[{\"type\":\"mesh3d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"parcoords\":[{\"type\":\"parcoords\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatter3d\":[{\"type\":\"scatter3d\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}]},\"layout\":{\"autotypenumbers\":\"strict\",\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"hovermode\":\"closest\",\"hoverlabel\":{\"align\":\"left\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"bgcolor\":\"#E5ECF6\",\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"ternary\":{\"bgcolor\":\"#E5ECF6\",\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]]},\"xaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"yaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"geo\":{\"bgcolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"subunitcolor\":\"white\",\"showland\":true,\"showlakes\":true,\"lakecolor\":\"white\"},\"title\":{\"x\":0.05},\"mapbox\":{\"style\":\"light\"}}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Per Capita Income Of Community (in K)\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Shannon Index (Diversity)\"}},\"legend\":{\"title\":{\"text\":\"\"},\"tracegroupgap\":0},\"title\":{\"text\":\"Income vs. Bird Diversity\"},\"height\":600,\"width\":800},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('428c65e5-8a99-456c-b90c-f157b8245182');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "required_cols = [\"shannon_index\", \"PER CAPITA INCOME IN K\", \"PovertyFlag\"]\n",
    "\n",
    "# Extract the required columns\n",
    "vis_df = final_df[required_cols]\n",
    "\n",
    "# Create a scatter plot with tooltips using Plotly Express\n",
    "fig = px.scatter(vis_df, x='PER CAPITA INCOME IN K', y='shannon_index', color='PovertyFlag',\n",
    "                 color_discrete_map={0: '#636EFA', 1: '#FFA15A'},\n",
    "                 title=\"Income vs. Bird Diversity\",\n",
    "                 labels={\"PER CAPITA INCOME IN K\": \"Per Capita Income Of Community (in K)\",\n",
    "                         \"shannon_index\": \"Shannon Index (Diversity)\",\n",
    "                        \"PovertyFlag\":\"\"},\n",
    "                 hover_data={\"PovertyFlag\": True},\n",
    "                 width=800, height=600)\n",
    "\n",
    "# Show the interactive plot\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "70a8068f-3635-4fd1-a75e-203bb6ae3a06",
   "metadata": {},
   "source": [
    "## Section 6: Reflection\n",
    "### What is hardest part of the project that you’ve encountered so far?\n",
    " Identifying community of a bird observation using the bird coordinates and the community boundaries.\n",
    "### What are your initial insights?\n",
    " Rich communities definitely have the higher bird diversity but the vice versa is not exactly true."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
