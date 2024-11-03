# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:22:35 2023

@author: Bori Dániel (ÖMKi)
"""

import geopandas as gpd
import fiona, rasterio
import pandas as pd
from rasterstats import zonal_stats
from rasterio.plot import show
import matplotlib.pyplot as plt
import rasterio.plot as rplt
from rasterio.features import rasterize
import os, re
import numpy as np
from sentinel_core.sentinel_config import *
import os
import logging
from datetime import datetime
from colorama import Fore as F
from colorama import Style
import re
import time
import rasterio
import rasterio.mask
import fiona
import logging

class CustomFormatter(logging.Formatter):
    reset = F.WHITE
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: F.WHITE + format + reset,
        logging.INFO: F.LIGHTCYAN_EX + format + reset,
        logging.WARNING: F.YELLOW + format + reset,
        logging.ERROR: F.LIGHTRED_EX + format + reset,
        logging.CRITICAL: F.LIGHTRED_EX + format + reset
    }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logging(log_filename='legelo_adatbetoltes', log_folder = 'log', console_level=logging.DEBUG):

    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    # Copying current config file
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(log_folder,
                                              fr'{log_filename}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(console_level)
    formatter = CustomFormatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def c_print(f, text, **kwargs):
    print(f"{Style.BRIGHT}{f}{text}", **kwargs)

def get_file(path, image_type, ext = 'tif'):
    expr = ""
    root_expr = r'.*'
    filename = ""
    match image_type:
        case "TCI" | "B02" | "B03" | "B04" | "B08" |"WVP" | "AOT" :            
            expr = fr'.*T{Config.tile_id}_\d{{8}}T\d{{6}}_{image_type}_10m.{ext}'
        case "CLDPRB":
            expr = fr'MSK_CLDPRB_20m.{ext}|.*_CLD_20m[.]{ext}'
        case "NDVI":
            folder_expr = r'^python_results$'
            expr = r'.*\d{8}T\d{6}_NDVI.tif$'
        case _:
            print("image_type parameter of get_file function might be wrong or not implemented")
            return None
    for root, folder, files in os.walk(path):        
        if re.match(root_expr, root):
            for file in files:
                # print(os.path.join(root, file))
                if re.match(expr, file):
                    return os.path.join(root, file)


def plot_raster_with_grid(raster_dataset, ax, xlabel='Easting', ylabel='Northing'):
    """
    Plot a satellite raster with a coordinate grid.

    Parameters:
        raster_dataset (rasterio.DatasetReader): Opened rasterio dataset.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        xlabel (str): Label for the x-axis. Default is 'Easting'.
        ylabel (str): Label for the y-axis. Default is 'Northing'.

    Returns:
        None
    """

    # Read the raster bands
    red_band = raster_dataset.read(1)
    green_band = raster_dataset.read(2)
    blue_band = raster_dataset.read(3)

    # Stack the bands to create an RGB image
    rgb_image = np.dstack((red_band, green_band, blue_band))

    # Plot the RGB image
    ax.imshow(rgb_image)

    # Add grid and labels
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Satellite Raster with Coordinate Grid')


def write_raster(data, src, dest_path, datatype=None):
    """
    Write raster data to file
    
    data: np.ndarray
    src: raster to inherit profile from (rasterio.DatasetReader)
    datatype: raster data type eg. rasterio.float64
    dest_path: full path to destination file
    
    
    """
    
    if not datatype:
        datatype = data.dtype
    
    with rasterio.Env():
        profile = src.profile
    
        profile.update(
            dtype=datatype,
            count=1,
            compress='lzw')
    
        with rasterio.open(dest_path, 'w', **profile) as dst:
            dst.write(data.astype(datatype), 1)
    
    
def enum_items(source):
    print("\n")
    for ele in enumerate(source): 
        print(ele)
 
def list_columns(df):
    field_list = list(df)
    enum_items(field_list)
    return field_list


# For loading feature classes into geopandas dataframe
def loadfc_as_gpd(fgdb):
    layers = fiona.listlayers(fgdb)
    enum_items(layers)
    index = int(input("Which index to load? "))
    fcgpd = gpd.read_file(fgdb,layer=layers[index])
    return fcgpd
 
# For loading shapefiles into geopandas dataframe
def loadshp_as_gpd(shp):
    data = gpd.read_file(shp)
    return data


# For re-projecting input vector layer to raster projection
def reproject(fcgpd, raster):
    proj = raster.crs.to_proj4()
    print("Original vector layer projection: ", fcgpd.crs)
    reproj = fcgpd.to_crs(proj)
    print("New vector layer projection (PROJ4): ", reproj.crs)
    fig, ax = plt.subplots(figsize=(15, 15))
    rplt.show(raster, ax=ax)
    reproj.plot(ax=ax, facecolor='none', edgecolor='red')
    fig.show()
    return reproj


# For dissolving geopandas dataframe by selected field
def dissolve_gpd(df):
    field_list = list_columns(df)
    index = int(input("Dissolve by which field (index)? "))
    dgpd = df.dissolve(by=field_list[index])
    return dgpd

 
# For calculating zonal statistics
def get_zonal_stats(vector, raster, stats):
    # Run zonal statistics, store result in geopandas dataframe
    result = zonal_stats(vector, raster, stats=stats, geojson_out=True)
    geostats = gpd.GeoDataFrame.from_features(result)
    return geostats

# For generating raster from zonal statistics result
def stats_to_raster(zdf, raster, stats, out_raster, no_data='y'):
    meta = raster.meta.copy()
    out_shape = raster.shape
    transform = raster.transform
    dtype = raster.dtypes[0]
    field_list = list_columns(stats)
    index = int(input("Rasterize by which field? "))
    zone = zdf[field_list[index]]
    shapes = ((geom,value) for geom, value in zip(zdf.geometry, zone))
    burned = rasterize(shapes=shapes, fill=0, out_shape=out_shape, transform=transform)
    show(burned)
    meta.update(dtype=rasterio.float32, nodata=0)
    # Optional to set nodata values to min of stat
    if no_data == 'y':
        cutoff = min(zone.values)
        print("Setting nodata cutoff to: ", cutoff)
        burned[burned < cutoff] = 0 
    with rasterio.open(out_raster, 'w', **meta) as out:
        out.write_band(1, burned)
    print("Zonal Statistics Raster generated")

def date_from_folder(product_folder):
    datetime = product_folder.split('_')[2]
    return f"{datetime[0:4]}-{datetime[4:6]}-{datetime[6:8]}"


def rasterio_clip(shapefile_path, raster_path):
    with fiona.open(shapefile_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    for i, shape in enumerate(shapes):
        with rasterio.open(raster_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, [shape], crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
        plt.imshow(out_image)

from collections import defaultdict

def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d

import xml.etree.ElementTree as ET
def plot_bands(root_path):
    """
    Plots band information
    Params:
    root_path: file path to sentinel xml to get spectral information from

    """
    tree = ET.parse(root_path)
    root = tree.getroot()

    prod_characteristics = etree_to_dict(root[0][1])['Product_Image_Characteristics']
    spectral_info = prod_characteristics['Spectral_Information_List']

    spectral_info_df = pd.DataFrame(spectral_info['Spectral_Information'])
    spectral_info_df['spectral_values'] = spectral_info_df['Spectral_Response'].apply(lambda x: np.array(x['VALUES'].split(' '), dtype=np.float64))
    spectral_info_df['min_spectrum'] = spectral_info_df['Wavelength'].apply(lambda x: int(x['MIN']['#text']))
    spectral_info_df['central_spectrum'] = spectral_info_df['Wavelength'].apply(lambda x: float(x['CENTRAL']['#text']))

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['midnightblue', 'blue', 'green', 'red', 'darkred', 'dimgrey',
                                                        'teal', 'orange', 'blueviolet', 'peru', 'khaki', 'cadetblue', 'black'])
    res_type = {'10': 'solid',
                '20': 'dashed',
                '60': 'dotted'}
    for i in range(0,13,1):
        arr = np.zeros(2500)
        row = spectral_info_df.iloc[i]
        min = row['min_spectrum']
        spectral_values = row['spectral_values']
        arr[min:min+len(spectral_values)] = spectral_values
        plt.plot(arr, label=f"{row['@physicalBand']} - {row['RESOLUTION']}m2 - {row['central_spectrum']:.1f} nm", linewidth=2, linestyle=res_type[row['RESOLUTION']])
    plt.xlim(xmin=400, xmax=2600)
    plt.legend()
    plt.show()

def get_images_dict(root_path):
    """

    :param root_path: path to MSIL2A xml
    :return: dictionary of image file paths
    """
    tree = ET.parse(root_path)
    root = tree.getroot()
    try:
        prod_inf = etree_to_dict(root[0][0])['Product_Info']
    except KeyError:
        prod_inf = etree_to_dict(root[0][0])['L2A_Product_Info']

    try:
        prod_org = prod_inf['Product_Organisation']['Granule_List']
    except KeyError:
        prod_org = prod_inf['L2A_Product_Organisation']['Granule_List']
    try:
        image_list = prod_org['Granule']['IMAGE_FILE']
    except KeyError:
        image_list = prod_org['Granule']['IMAGE_FILE_2A']
    img_dict = {f"{x.split('_')[6]}_{x.split('_')[7]}": x for x in image_list}
    return img_dict


def rasterize_vector(vector_path, by_field, dtype='Int32'):
    output_root = r'G:/GeoData/temp'
    raster_path = os.path.join(output_root, f'{os.path.splitext(os.path.basename(vector_path))[0]}_'
                                            f'{datetime.now().strftime("%Y%m%dT%H%M%S")}.tif')
    logging.info('Calling GDAL rasterize for {vector_path}')
    rasterize_command_10m = (fr'"C:\Program Files\QGIS 3.28.8\bin\gdal_rasterize.exe"'
                             f' -a {by_field} -tr 10 10 -a_nodata -9999 -te 753980.000000000 5158970.000000000 778410.000000000 5187320.000000000 '
                             f' -ot {dtype} -of GTiff {vector_path} {raster_path}')

    os.system(rasterize_command_10m)
    return raster_path

def open_as_raster(vector_path, by_field='category', nodata=-9999, sleep=0, **kwargs):
    raster_path = rasterize_vector(vector_path, by_field, **kwargs)
    time.sleep(sleep)
    with rasterio.open(raster_path) as src:
        img = src.read(1)
    return np.where(img == nodata, np.nan, img)
    # img = rasterio.open().read(1)

def plot_from_dict(to_plot, nrows=None, ncols=None, xticks=None, yticks=None, xticksl='', yticksl='',ylabel='', xlabel='', title = None, extent = None, **kwargs):
    """

    :param to_plot: list of images to be passed to ax.imshow
    :param nrows: optional nubor of rows in plot
    :param ncols: optional, number of columns in plot
    :param extent: (xmin, xmax, ymin, ymax)
    :return:
    """

    if not nrows:
        nrows = int(np.sqrt(len(to_plot))/1.4)
        ncols = int(np.ceil(len(to_plot)/nrows))

    fig, axs = plt.subplots(nrows,ncols)
    for i, (key, img) in enumerate(to_plot.items()):
        if extent:
            img = img[extent[0]:extent[1], extent[2]:extent[3], :]
        axs[int(i/ncols)][i%ncols].imshow(img, **kwargs)
        if xticks is not None:
            axs[int(i/ncols)][i%ncols].set_xticks(xticks, labels = xticksl)
            axs[int(i/ncols)][i%ncols].set_yticks(yticks, labels = yticksl)
        else:
            axs[int(i/ncols)][i%ncols].axis('off')
        axs[int(i / ncols)][i % ncols].set_ylabel(ylabel)
        axs[int(i / ncols)][i % ncols].set_xlabel(xlabel)
        if title:
            axs[int(i/ncols)][i%ncols].text(1.05, 0.5, title[i], transform=axs[int(i/ncols)][i%ncols].transAxes, va='center', rotation='vertical')
        else:
            axs[int(i/ncols)][i%ncols].set_title(key, pad=0)

    for i in range(nrows):
        for j in range(ncols):
            if (len(axs[i, j].images)==0):
                axs[i, j].axis('off')

    plt.subplots_adjust(hspace=.1, wspace=0)
    plt.show()
    return fig


def str_to_date(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """

    # Create a dictionary with unique dates as keys and their corresponding
    # parsed datetime objects as values
    dates = {date: pd.to_datetime(date,
                              format="%Y-%m-%d") for date in s.unique()}

    # Map the original dates to their parsed values using the lookup dictionary
    return s.map(dates).dt.date




