import matplotlib.dates
matplotlib.use('qtagg')
from sentinel_core import SentinelProduct
from sentinel_functions import *
import logging
import time

init_logging(console_level=logging.INFO)

df_all = pd.read_csv(r'G:\Geodata\sentinel_cropped\metadata\metadata_all\meta_allyears.csv')
df_all['datetime'] = pd.to_datetime(df_all['datetime'])
meta_df = df_all[df_all[['CLOUD_HIGH_PROBA', 'CLOUD_MEDIUM_PROBA', 'THIN_CIRRUS', 'CLOUD_SHADOW']].sum(axis=1) < 0.32]

meta_df['datetime'] = pd.to_datetime(meta_df['datetime'])


for year in [2021,2022,2023]:
    meta_current = meta_df[meta_df['datetime'].dt.year==year]
    prod_list = []
    for path in meta_current['root_path']:
        prod_list.append(SentinelProduct(path))

    band_list = ['B04_10m', 'B08_10m']

    # prod = prod_list[22]
    monthly_ndvi = {}
    lastmonth = None
    ndvi_current = np.empty(prod_list[0].shape_10m)
    ndvi_alltimes = None

    with open(f'product_ndvi/{year}_product_dates.csv', 'a') as file:
        file.write(f"index, date\n")
        for i, p in enumerate(prod_list):
            file.write(f"{i}, {p.date}\n")

    for index, prod in enumerate(prod_list):
        start_time = time.time()
        bands = prod.get_bands(band_list)
        scl = prod.open_scl(convert_to_10=True)
        cldmask = np.isin(scl[:, :, np.newaxis], [0, 3, 8, 9, 10])
        # Replace cloudy pixels with np.nan
        nocld_data = np.where(cldmask == 1, np.nan, bands['data'])
        month = datetime.fromisoformat(prod.date).month
        red = nocld_data[:,:,0]
        nir = nocld_data[:,:,1]
        ndvi = (nir-red)/(nir+red)
        ndvi = ndvi[np.newaxis, :, :]
        if ndvi_alltimes is None:
            ndvi_alltimes = ndvi
        else:
            ndvi_alltimes = np.concatenate((ndvi, ndvi_alltimes), axis=0)
        if not lastmonth:
            ndvi_current = ndvi
            lastmonth = month
        else:
            if lastmonth == month:
                ndvi_current = np.concatenate((ndvi, ndvi_current), axis=0 )
            else:
                monthly_ndvi[lastmonth] = np.nanmean(ndvi_current, axis=0)
                ndvi_current = ndvi
                lastmonth = month
        print(f"{prod.date} calculated in {time.time() - start_time} secs")
    monthly_ndvi[lastmonth] = np.nanmean(ndvi_current, axis=0)

    all_months = np.stack(list(monthly_ndvi.values()))

    meta = prod.raster_meta_10m
    meta.update(dtype=rasterio.float32, count=ndvi_alltimes.shape[0])
    # meta
    with rasterio.open(f'product_ndvi/raster_all_{year}.tif', 'w', **meta) as file:
        file.write(ndvi_alltimes)
    meta.update(count=all_months.shape[0])
    with rasterio.open(f'product_ndvi/raster_monthly_{year}.tif', 'w', **meta) as file:
        file.write(all_months)

months = ['Jan', 'Febr', 'Márc', 'Ápr', 'Máj', 'Jún', 'Júl', 'Aug', 'Szept', 'Okt', 'Nov', 'Dec']
fig = plot_from_dict(monthly_ndvi, nrows=3, ncols=4, vmin=0, vmax=1, title=months, cmap='RdYlGn')

fig.axes[0].colorbar()

all_months = np.stack(list(monthly_ndvi.values()))
all_months_std = np.nanstd(all_months, axis=0)

for i in range(12):
    prod.save_raster(all_months[i, :, :][np.newaxis, :,:], rf'G:\Geodata\sentinel_ndvi\2022\ndvi_2022_{i+1}.tif')

prod.save_raster(np.nanmean(all_months, axis=0)[np.newaxis, :,:], rf'G:\Geodata\sentinel_ndvi\2022\ndvi_2022_nanmean.tif')
prod.save_raster(np.nanstd(all_months[3:9, :,:], axis=0)[np.newaxis, :,:], r'G:/GeoData/sentinel_ndvi_2023/monthly_std_apr-okt.tif')

ndvi_alltimes_argmax = np.empty(ndvi_alltimes.shape)
ndvi_alltimes_std = np.nanstd(ndvi_alltimes.astype(np.float32), axis=0)

ndvi_nanmean = np.nanmean(ndvi_alltimes, axis=0)
ndvi_nanargmax = np.nanargmax(ndvi_alltimes.astype(np.float32)[~np.isnan(ndvi_alltimes[0, :,:])], axis=0)

prod.save_raster(ndvi_alltimes_std[np.newaxis, :,:], r'G:/GeoData/ndvi_std_all_2022.tif')

