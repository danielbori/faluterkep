import os

def add_exif_gps_tags(groupname):
    root = QgsProject.instance().layerTreeRoot()
    image_folder = r'C:\Users\user\archiv_legifotok\1977'
    group = root.findGroup(groupname)
    list = group.findLayers()
    source_crs = QgsCoordinateReferenceSystem(23700)
    dest_crs = QgsCoordinateReferenceSystem(4326)
    tr = QgsCoordinateTransform(source_crs, dest_crs, QgsProject.instance())
    for lyr in list:
        path_r = os.path.join(image_folder,  f"{lyr.name()}.tif")
        center_point = lyr.layer().dataProvider().extent().center()
        center_point = QgsPoint(center_point)
        center_point.transform(tr)
        exiftool_call = f'exiftool "{os.path.join(image_folder, lyr.name())}.TIF"\
         -GPSLatitude={center_point.y()} -GPSLatitudeRef=N \
 -GPSLongitude={center_point.x()}  -GPSLongitudeRef=E \
 -GPSAltitude=1880 -GPSAltitudeRef=0'
        with open(os.path.join(image_folder, 'add_gps.bat'), 'a+') as file:
            file.write(exiftool_call)
add_exif_gps_tags('1977')