import os
import time
from qgis.core import QgsProject, QgsRasterLayer

# Raszter rétegek exportálása
def export_raster_layers(groupname, output_folder):
    root = QgsProject.instance().layerTreeRoot()
    group = root.findGroup(groupname)
    for child in group.findLayers():
        layer=child.layer()
        print(f"Exporting {layer.name()}")
        start = time.time()
        if isinstance(layer, QgsRasterLayer):
            file_name = os.path.join(output_folder,
                                     f"{layer.name().replace(' ', '_')}.tif")
            file_writer = QgsRasterFileWriter(file_name)
            pipe = QgsRasterPipe()
            provider = layer.dataProvider()
            if not pipe.set(provider.clone()):
                print ("Cannot set pipe provider")
                continue

            file_writer.writeRaster(
                pipe,
                provider.xSize(),
                provider.ySize(),
                provider.extent(),
                provider.crs())
            print(f"{layer.name()} export kész -- {time.time()-start:.2f} mp.")
        else:
            print(f"Layer is a {type(layer)}, not a raster")
            
if __name__ == "__main__":
    group_name = "réteg_csoport_neve" 
    output_folder = r"C:\geoadat\raszter_mappa"
    export_raster_layers(group_name, output_folder)
