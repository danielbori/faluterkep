import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates
matplotlib.use('qtagg')
from sentinel_core import SentinelProduct, plot_tci_from_dataframe
from .sentinel_functions import *
import logging
import time
import geopandas as gpd
import seaborn as sns
from sklearnex.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score

init_logging(console_level=logging.WARNING)


meta_df = pd.read_csv(r"G:\Geodata\sentinel_cropped\2023\2023_meta_classification.csv")
vector_path = r'G:/Geodata/vector/shp/haszn_mintak_33n_2023_ekezetes_3.shp'
classes = open_as_raster(vector_path, sleep=1)
classes_gpd = gpd.read_file(vector_path)
# classes_gpd.sort_values('category')
df_nocld = meta_df[
    meta_df[['CLOUD_HIGH_PROBA', 'CLOUD_MEDIUM_PROBA', 'THIN_CIRRUS', 'CLOUD_SHADOW']].sum(axis=1) < 0.16]

prod_list = []
for path in df_nocld['root_path']:
    prod_list.append(SentinelProduct(path))

band_list = ['B01_20m',
             'B02_10m',
             'B03_10m',
             'B04_10m',
             'B05_20m',
             'B06_20m',
             'B07_20m',
             'B08_10m',
             'B8A_20m',
             'B09_60m',
             'B11_20m',
             'B12_20m']

accuracy_scores = []
# roc_curves = []
confusion_matrices = []
class_count = len(classes_gpd['category'].unique())
all_predictions_proba = np.ndarray((len(prod_list), prod_list[0].shape_10m[0], prod_list[0].shape_10m[1], class_count), dtype=np.float32)
all_predictions = np.ndarray((len(prod_list), prod_list[0].shape_10m[0], prod_list[0].shape_10m[1]), dtype=np.uint8)
# df_coll = pd.DataFrame()
index, prod = 1, prod_list[1]
for index, prod in enumerate(prod_list):
    start_time = time.time()
    bands = prod.get_bands(band_list)
    bands['class'] = classes
    scl = prod.open_scl(convert_to_10=True)
    cldmask = np.isin(scl[:,:,np.newaxis], [0,3,8,9,10])
    # Replace cloudy pixels with np.nan
    nocld_data = np.where(cldmask == 1, np.nan, bands['data'])
    classes_data = np.where(cldmask[:,:,0] == 1, np.nan, bands['class'])
    training_data = np.where(np.isnan(bands['class'][:,:,np.newaxis]), np.nan, nocld_data)

    training_data = training_data.reshape(-1, training_data.shape[-1])
    classes_data = classes_data.reshape(-1)

    training_data = training_data[~np.isnan(training_data[:, 1]), :]
    classes_data = classes_data[~np.isnan(classes_data)]

    X_train, X_test, y_train, y_test = train_test_split(training_data, classes_data, test_size=1/3, random_state=42 )

    rf_classifier = RandomForestClassifier()
    # Fitting the Random Forest Classifier
    rf_classifier.fit(X_train, y_train)
    # Make prediction
    test_result = rf_classifier.predict(X_test)
    # More metrics here
    accuracy_scores.append({prod.date: accuracy_score(y_test, test_result)})
    confusion_matrices.append({prod.date: confusion_matrix(y_test, test_result)})

    result_prob_all = np.empty((prod.shape_10m[0], prod.shape_10m[1], class_count))
    nocld_data = nocld_data.reshape(-1, nocld_data.shape[2])

    prediction = np.empty(prod.shape_10m).reshape(-1)
    prediction[~np.isnan(nocld_data[:,1])] = rf_classifier.predict(nocld_data[~np.isnan(nocld_data[:, 1])])
    prediction = prediction.reshape(prod.shape_10m)
    prod.save_raster(prediction[np.newaxis, :,:], rf'G:\Geodata\classification\2023_per_product\{prod.date}_prediction_result.tif')
    all_predictions[index] = prediction

    all_predictions[index] = prediction.reshape(prod.shape_10m)
    proba_prediction = rf_classifier.predict_proba(nocld_data[~np.isnan(nocld_data[:, 1])])
    for j in range(class_count):
        result_prob = np.empty((prod.shape_10m[0] * prod.shape_10m[1])) * np.nan
        result_prob[~np.isnan(nocld_data[:,1])] = proba_prediction[:,j]
        result_prob = result_prob.reshape(2835, 2443)
        result_prob_all[:,:,j] = result_prob
    all_predictions_proba[index] = result_prob_all
    print(f"Prediction on {prod.date} done in {time.time()-start_time:.2f} secs")
