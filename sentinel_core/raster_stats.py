# -*- coding: utf-8 -*-
"""
Created on Sat May 20

@author: Bori Dániel
"""

#%%Imports

from sentinel_core.sentinel_functions import *
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from rasterio.plot import show
import rasterstats
import pandas as pd
import geopandas as gpd
import logging
from rasterio.mask import mask

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

all_stats = gpd.GeoDataFrame()
stats = ['min', 'max', 'mean', 'count',
          'sum', 'std', 'median', 'majority',
          'minority', 'unique', 'range']

image_path = r"G:\Geodata\sentinel_cropped\2017\S2A_MSIL2A_20170108T095401_N0204_R079_T33TYM_20170108T095355.SAFE\GRANULE\L2A_T33TYM_A008084_20170108T095355\IMG_DATA\R10m\L2A_T33TYM_20170108T095401_TCI_10m.tif"

vector = r"G:\Geodata\vector\clc50\CLC_50.shp"
vector = gpd.read_file(vector, crs=23700).to_crs(32633)

clc_diss = vector

vector2 = gpd.read_file(r"G:\Geodata\vector\shp\nagyszekely_hatar.shp")
vec = vector2.geometry

src = rasterio.open(image_path)
geom = vec.to_crs('32633')
print(f"geom bounds: {geom.bounds}")

result = rasterstats.zonal_stats(vectors = clc_diss['geometry'], raster = src.read(2), stats=['max', 'min', 'mean', 'count'], affine=src.transform, geojson_out=True)
geostats = gpd.GeoDataFrame.from_features(result)

fig, axs = plt.subplots(1,3)
show(src.read(), transform=src.transform, ax=axs[0])
clc_diss.plot(ax = axs[1])
geostats.plot(ax = axs[2], column='mean', legend=True)
plt.show()