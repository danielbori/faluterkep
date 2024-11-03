import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from .sentinel_functions import *
import pandas as pd
import rasterio
import numpy as np
from rasterio.enums import Resampling
import logging

class SentinelProduct:
    """
    For managing a single sentinel product
    """

    def __init__(self, product_root_path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root_path = product_root_path
        self.logger.info(f"Initializing product from {self.root_path}")
        self.xml_path = os.path.join(product_root_path,  'MTD_MSIL2A.xml')
        self.xml_root = ET.parse(self.xml_path).getroot()
        self.img_dict, self.prod_inf, self.prod_org = self.get_images_dict()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger = logging.getLogger(f"{self.__class__.__name__} - {self.date}")
        self.shape_10m = rasterio.open(self.img_dict['B02_10m']).shape
        try:
            self.prod_characteristics = etree_to_dict(self.xml_root[0][1])['Product_Image_Characteristics']
        except KeyError:
            self.prod_characteristics = etree_to_dict(self.xml_root[0][1])['L2A_Product_Image_Characteristics']

    @property
    def boa_quantification_value(self):
        try:
            boa_value = int(self.prod_characteristics['QUANTIFICATION_VALUES_LIST']['BOA_QUANTIFICATION_VALUE']['#text'])
        except KeyError:
            boa_value = int(self.prod_characteristics['L1C_L2A_Quantification_Values_List']['L2A_BOA_QUANTIFICATION_VALUE']['#text'])
        return boa_value


    @property
    def reflectance_conversion_list(self):
        return self.prod_characteristics['Reflectance_Conversion']['SOLAR_IRRADIANCE']



    @property
    def scene_classification_list(self):
        try:
            classification_dict =  {int(x['SCENE_CLASSIFICATION_INDEX']): x['SCENE_CLASSIFICATION_TEXT']
                    for x in self.prod_characteristics['Scene_Classification_List']['Scene_Classification_ID']}
        except KeyError:
            classification_dict =  {int(x['L2A_SCENE_CLASSIFICATION_INDEX']): x['L2A_SCENE_CLASSIFICATION_TEXT']
                    for x in self.prod_characteristics['L2A_Scene_Classification_List']['L2A_Scene_Classification_ID']}
        return classification_dict

    @property
    def boa_offset(self):
        if 'BOA_ADD_OFFSET_VALUES_LIST' in self.prod_characteristics.keys():
            return {int(x['@band_id']): int(x['#text']) for x in self.prod_characteristics['BOA_ADD_OFFSET_VALUES_LIST']['BOA_ADD_OFFSET']}
            # return pd.DataFrame(self.prod_characteristics['BOA_ADD_OFFSET_VALUES_LIST'])
        else:
            return None

    @property
    def bands_spectrum_df(self):
        spectral_info = self.prod_characteristics['Spectral_Information_List']
        # spectral_info = {s['@bandId']: s['@physicalBand'] for s in spectral_info['Spectral_Information']}

        spectral_info_df = pd.DataFrame(spectral_info['Spectral_Information'])
        spectral_info_df['spectral_values'] = spectral_info_df['Spectral_Response'].apply(
            lambda x: np.array(x['VALUES'].split(' '), dtype=np.float64))
        spectral_info_df['min_spectrum'] = spectral_info_df['Wavelength'].apply(lambda x: int(x['MIN']['#text']))
        spectral_info_df['central_spectrum'] = spectral_info_df['Wavelength'].apply(
            lambda x: float(x['CENTRAL']['#text']))
        # spectral_info_df['physical_band'] = spectral_info_df['@physicalBand'].apply(lambda x: 'B' + x[1:].lstrip('0', ))
        # spectral_info_df['boa_offset'] = x['#text'] for x in self.boa_offset
        return spectral_info_df

    @property
    def date(self):
        return str(self.datetime.date())
    @property
    def datetime(self):
        return datetime.fromisoformat(self.prod_inf['PRODUCT_START_TIME'])
    @property
    def raster_meta_10m(self):
        with rasterio.open(self.img_dict['B02_10m']) as dataset:
            meta = dataset.meta
        return meta


    def open_scl(self, convert_to_10 = False):
        if convert_to_10:
            img = self.open_upscale(self.img_dict['SCL_20m'], upscale_factor=2)
            # BAD WORK, SHOULD FIX - works for my dataset, but could shift rasters
            img = img[:,:,:-1]
            img = np.pad(img, ((0, 0), (0, 1), (0, 0)), mode='constant')[0,:,:]
        else:
            img = rasterio.open(self.img_dict['SCL_20m']).read(1)
        return img

    def open_tci(self, resolution=10):
        img = rasterio.open(self.img_dict[f'TCI_{int(resolution)}m']).read()
        return img

    def get_images_dict(self):
        """
        :return: dictionary of image file paths, prod_inf, prod_org
        """

        # Read from a file
        # tree = ET.parse(r"G:\Geodata\sentinel_cropped\2018_unzipped\S2A_MSIL2A_20180103T095401_N9999_R079_T33TYM_20221023T202305\MTD_MSIL2A.xml")

        try:
            prod_inf = etree_to_dict(self.xml_root[0][0])['Product_Info']
        except KeyError:
            prod_inf = etree_to_dict(self.xml_root[0][0])['L2A_Product_Info']

        try:
            prod_org = prod_inf['Product_Organisation']['Granule_List']
        except KeyError:
            prod_org = prod_inf['L2A_Product_Organisation']['Granule_List']
        try:
            image_list = prod_org['Granule']['IMAGE_FILE']
        except TypeError:
            try:
                xss = [x['Granule']['IMAGE_FILE_2A'] for x in prod_org]
                image_list = [x for xs in xss for x in xs ]
            except KeyError:
                xss = [x['Granule']['IMAGE_FILE'] for x in prod_org]
                image_list = [x for xs in xss for x in xs ]
        except KeyError:
            image_list = prod_org['Granule']['IMAGE_FILE_2A']
        img_dict = {f"{x.split('_')[-2]}_{x.split('_')[-1]}": os.path.join(self.root_path, x + '.tif') for x in image_list}
        return (img_dict, prod_inf, prod_org)

    def band_id_from_key(self, band_key):
        physical_band = f"B{band_key.split('_')[0][1:].lstrip('0')}"
        band_id = int(self.bands_spectrum_df[self.bands_spectrum_df['@physicalBand'] == physical_band]['@bandId'].iloc[0])
        return band_id


    def open_upscale(self, path, upscale_factor = 2, mask_geom = None):
        with rasterio.open(path) as dataset:
            # resample data to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * upscale_factor),
                    int(dataset.width * upscale_factor)
                ),
                resampling=Resampling.bilinear
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]),
                (dataset.height / data.shape[-2])
            )
            return data

    def get_surface_reflectance(self, band_key):
        self.logger.info(f'Getting surface reflectance for {band_key}')
        if '10m' in band_key:
            img = rasterio.open(self.img_dict[band_key]).read(1)
        elif '20m' in band_key:
            img = self.open_upscale(self.img_dict[band_key], upscale_factor=2)
            # BAD WORK, SHOULD FIX - works for my dataset, but could shift rasters
            img = img[:,:,:-1]
            img = np.pad(img, ((0, 0), (0, 1), (0, 0)), mode='constant')[0,:,:]
        elif '60m' in band_key:
            img = self.open_upscale(self.img_dict[band_key], upscale_factor=6)
            # BAD WORK, SHOULD FIX - works for my dataset, but could shift rasters
            img = img[:,4:,2:]
            img = np.pad(img, ((0, 0), (0, 7), (0, 3)), mode='constant')[0,:,:]
        else:
            self.logger.error(f"{band_key} is not valid for surface_reflectance")
            return 'error'
        if self.boa_offset:
            offset = self.boa_offset[self.band_id_from_key(band_key)]
            return (np.add(img,offset))/self.boa_quantification_value
        else:
            return img/self.boa_quantification_value


    def get_bands(self, band_subset = ['B02_10m', "B03_10m", "B04_10m", 'B08_10m']):
        arrays = {}
        for physicalBand in band_subset:
            new_band = self.get_surface_reflectance(physicalBand)
            # if len(arrays) > 0:
            #     if arrays[0].shape != new_band.shape:
            #         raise ValueError(f"New Array shape for {physicalBand} - {new_band.shape} not matching previous shape: {arrays[0].shape}")
            arrays[physicalBand] = new_band
        stacked_arr = np.stack(list(arrays.values()), axis=2)
        return {'bands': list(arrays.keys()), 'data': stacked_arr}

    def plot_bands(self):
        plot_bands(self.root_path)

    def save_raster(self, data, path):
        meta = self.raster_meta_10m
        meta.update(dtype=rasterio.float32)
        # meta
        with rasterio.open(path, 'w', **meta) as file:
            file.write(data)

    def get_memory_raster(self, data, **kwargs):
        metadata = self.get_meta(**kwargs)
        # Create a MemoryFile
        mem = rasterio.io.MemoryFile()
        # Create a new raster dataset in memory
        with mem.open(**metadata) as dataset:
            # Write data to the raster dataset
            dataset.write(data, 1)
        return mem

    def get_meta(self, resolution = 10, dtype = rasterio.float32):
        if resolution == 20:
            with rasterio.open(self.img_dict['B02_20m']) as dataset:
                meta = dataset.meta
                meta.update(dtype = dtype)
        elif resolution == 10:
            with rasterio.open(self.img_dict['B02_10m']) as dataset:
                meta = dataset.meta
                meta.update(dtype = dtype)
        elif resolution == 60:
            with rasterio.open(self.img_dict['B09_60m']) as dataset:
                meta = dataset.meta
                meta.update(dtype=dtype)
        else:
            self.logger.error(f"Could not get metadata for resolution {resolution}")
            raise ValueError("get_meta Could not get metadata for resolution {resolution}")
        return meta

    def mask_memfile(self, memfile, geom, **kwargs):
        meta = self.get_meta(**kwargs)
        nodata = -9999
        if meta['dtype'] == rasterio.uint8:
            nodata = 255
        with memfile.open(**meta) as dataset:
            out_array, out_transform = rasterio.mask.mask(dataset, geom, nodata=nodata)
        return out_array, out_transform, nodata

    def get_masked(self, band_key, geom, **kwargs):
        if 'SCL' in band_key:
            memfile = self.get_memory_raster(self.open_scl(), resolution=20, dtype=rasterio.uint8)
            out_array, out_transform, nodata = self.mask_memfile(memfile, geom, dtype=rasterio.uint8, **kwargs)
            memfile.close()
            arr = np.where(out_array == nodata, np.nan, out_array)
        else:
            memfile = self.get_memory_raster(self.get_surface_reflectance(band_key))
            out_array, out_transform, nodata = self.mask_memfile(memfile, geom, **kwargs)
            memfile.close()
            arr = np.where(out_array == nodata, np.nan, out_array)

        return arr, out_transform

def plot_tci_from_dataframe(df, **kwargs):
    """
    :param df: should have root_path of sentinel products to plot
    :return: figure
    """
    tci_dict = {}
    for sen_path in df['root_path']:
        senprod = SentinelProduct(sen_path)
        tci_dict[senprod.date] = (np.transpose(senprod.open_tci(resolution=60), (1, 2, 0)))
    return plot_from_dict(tci_dict, **kwargs)

