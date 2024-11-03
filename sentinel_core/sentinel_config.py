# -*- coding: utf-8 -*-
"""
Created on Sat May 20 14:47:37 2023

@author: Bori Dániel
"""

class Config:

    source_folder = r"path_to_sentinel_data"
    result_folder = r"path_to_sentinel_data\sentinel_cropped"
    start_date = '2016-01-01'
    end_date = '2019-01-01'
    max_cloud_cover = 80
    vector_path = r""

    upper_left_x, upper_left_y = 595070, 160180
    lower_right_x, lower_right_y = 618000,131450
    tile_id = '33TYM'