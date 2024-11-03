# -*- coding: utf-8 -*-
"""
Created on Sat May 20

@author: Bori Dániel
"""

#%%Imports

import os
import re
import subprocess
import logging

for year in [2021]:
    print(f"Starting cropping {year}")
    source_folder = rf"K:\sentinel_nagyszekely\{year}\unzipped"
    result_folder =r"I:\sentinel\cropped"

    upper_left_x, upper_left_y = 595070, 160180
    lower_right_x, lower_right_y = 618000, 131450

    # Crop data
    result_path = os.path.join(result_folder, str(year))
    if not os.path.isdir(result_path):
        os.mkdirs(result_path)

    window = (upper_left_x, upper_left_y,
              lower_right_x, lower_right_y)
    for root, folder, files in os.walk(source_folder):
        if os.path.isdir(os.path.join(result_path, os.path.split(root)[1])):
            print(folder)
            continue
        for file in files:
            if re.match(r'.*[.]jp2$', file) and not re.match(r'(MSK_DETFOO|MSK_QUALIT)_...[.]jp2$', file):
                cropped_file = os.path.join(result_path, os.path.splitext(os.path.relpath(os.path.join(root, file), source_folder))[0] + '.tif')
                if not os.path.isdir(os.path.split(cropped_file)[0]):
                    os.makedirs(os.path.split(cropped_file)[0])

                # GDAL_Translate using osGeoShell
                osgeoshell = r"C:\Program Files\QGIS 3.28.8\OSGeo4W.bat"
                gdalTranslate = r"C:\ProgramData\anaconda3\envs\geo\Library\bin\gdal_translate.exe"
                transcmd = r' -projwin ' + '595070 161000 618000 131450' + ' -projwin_srs "EPSG:23700" '
                if not os.path.isfile(cropped_file):
                    subprocess.call(osgeoshell + " gdal_translate.exe" + transcmd + os.path.join(root, file) + ' ' + cropped_file)
                else:
                    logging.warning(f"{cropped_file} already exists, not cropping...")
            else:
                new_path = os.path.join(result_path, os.path.relpath(os.path.join(root, file), source_folder))
                if not os.path.isdir(os.path.split(new_path)[0]):
                    os.makedirs(os.path.split(new_path)[0])
                if os.path.isfile(new_path):
                    logging.warning(f"{new_path} File exists, not copying")
                    continue
                os.system(f'copy {os.path.join(root, file)} {new_path}')

