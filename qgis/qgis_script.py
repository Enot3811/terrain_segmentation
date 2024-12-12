from pathlib import Path

from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsRectangle,
    QgsPointXY,
    QgsCoordinateTransform,
    QgsLayoutExporter,
    QgsPrintLayout,
    QgsLayoutItemMap,
    QgsUnitTypes,
    QgsLayoutSize,
    QgsVectorLayer,
    QgsSingleSymbolRenderer,
    QgsFillSymbol,
    QgsLineSymbol,
    QgsWkbTypes,
    QgsMarkerSymbol
)


def export_satellite_map_fragment(
    layer_name: str,
    point1: tuple[float, float],
    point2: tuple[float, float],
    scale: int,
    output_path: str,
    dpi: int = 300,
    lat_lon: bool = False,
    verbose: bool = False
):
    """Выгрузить регион ограниченный двумя точками с заданным масштабом."""
    # Get the layer by name
    layer = QgsProject.instance().mapLayersByName(layer_name)[0]
    
    # Create QgsPointXY objects from the input coordinates
    p1 = QgsPointXY(point1[0], point1[1])
    p2 = QgsPointXY(point2[0], point2[1])

    # Create a rectangle from the two points
    rect = QgsRectangle(p1, p2)
    
    # Transform the rectangle to the layer's CRS
    if lat_lon:
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        transform = QgsCoordinateTransform(
            wgs84, layer.crs(), QgsProject.instance())
        transformed_rect = transform.transformBoundingBox(rect)
    else:
        transformed_rect = rect
    
    # Calculate the size of the map item based on the extent and scale
    width_mm = transformed_rect.width() / scale * 1000
    height_mm = transformed_rect.height() / scale * 1000
    
    # Create a new print layout
    project = QgsProject.instance()
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()

    # Set the layout size
    layout.pageCollection().pages()[0].setPageSize(
        QgsLayoutSize(width_mm, height_mm, QgsUnitTypes.LayoutMillimeters))
    
    # Add a map item to the layout
    map_item = QgsLayoutItemMap(layout)
    map_item.setRect(
        0, 0, width_mm, height_mm)  # Set the size and position of the map item
    
    # Set the map extent to our transformed rectangle
    map_item.setExtent(transformed_rect)
    
    # Set the map scale
    map_item.setScale(scale)

    # Set only the specific layer to be rendered
    map_item.setLayers([layer])
    
    # Add the map item to the layout
    layout.addLayoutItem(map_item)
    
    # Export the layout as an image
    exporter = QgsLayoutExporter(layout)
    image_settings = exporter.ImageExportSettings()
    image_settings.dpi = dpi  # Set the export resolution
    exporter.exportToImage(output_path, image_settings)
    
    if verbose:
        print(f"Map fragment exported to {output_path}")
        print(f'Image size: {width_mm * dpi / 25.4:.0f}x'
              f'{height_mm * dpi / 25.4:.0f} pixels')
        

def create_styled_layer(layer_name, color):
    # Get the vector layer by name
    layer = QgsProject.instance().mapLayersByName(layer_name)[0]
    
    # Create a new memory layer to store features
    styled_layer = QgsVectorLayer(
        layer.source(), f"Styled {layer_name}", layer.providerType())
    
    # Style the layer based on its geometry type
    if styled_layer.geometryType() == QgsWkbTypes.LineGeometry:
        symbol = QgsLineSymbol.createSimple({'width': '0.5'})
        symbol.setColor(QColor(*color))
    elif styled_layer.geometryType() == QgsWkbTypes.PolygonGeometry:
        symbol = QgsFillSymbol.createSimple({'outline_style': 'no'})
        symbol.setColor(QColor(*color))
    elif styled_layer.geometryType() == QgsWkbTypes.PointGeometry:
        symbol = QgsMarkerSymbol.createSimple({'size': '1'})
        symbol.setColor(QColor(*color))
    else:
        print(f"Unsupported geometry type for layer: {layer_name}")
        return None
    
    renderer = QgsSingleSymbolRenderer(symbol)
    styled_layer.setRenderer(renderer)
    
    return styled_layer


def export_vector_map_fragment(
    layer_config: dict,
    point1: tuple[float, float],
    point2: tuple[float, float],
    scale: int,
    output_path: str,
    dpi: int = 300,
    lat_lon: bool = False,
    verbose: bool = False
):
    """
    Выгрузить векторный регион ограниченный двумя точками с заданным масштабом.
    """
    # Create styled layers
    styled_layers = [create_styled_layer(layer_name, color)
                     for layer_name, color in layer_config.items()]
    styled_layers = [layer for layer in styled_layers
                     if layer is not None]  # Remove any None layers

    if not styled_layers:
        print("No valid layers to export")
        return
    
    # Create QgsPointXY objects from the input coordinates
    p1 = QgsPointXY(point1[0], point1[1])
    p2 = QgsPointXY(point2[0], point2[1])

    # Create a rectangle from the two points
    rect = QgsRectangle(p1, p2)
    
    # Transform the rectangle to the layers' CRS
    # (assuming all layers have the same CRS)
    if lat_lon:
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        transform = QgsCoordinateTransform(
            wgs84, styled_layers[0].crs(), QgsProject.instance())
        transformed_rect = transform.transformBoundingBox(rect)
    else:
        transformed_rect = rect
    
    # Calculate the size of the map item based on the extent and scale
    width_mm = transformed_rect.width() / scale * 1000
    height_mm = transformed_rect.height() / scale * 1000
    
    # Create a new print layout
    project = QgsProject.instance()
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()

    # Set the layout size
    layout.pageCollection().pages()[0].setPageSize(
        QgsLayoutSize(width_mm, height_mm, QgsUnitTypes.LayoutMillimeters))
    
    # Add a map item to the layout
    map_item = QgsLayoutItemMap(layout)
    map_item.setRect(
        0, 0, width_mm, height_mm)  # Set the size and position of the map item
    
    # Set the map extent to our transformed rectangle
    map_item.setExtent(transformed_rect)

    # Set the map scale
    map_item.setScale(scale)
    
    # Add layers to the map item
    map_item.setLayers(styled_layers)
    
    # Add the map item to the layout
    layout.addLayoutItem(map_item)
    
    # Export the layout as an image
    exporter = QgsLayoutExporter(layout)
    image_settings = exporter.ImageExportSettings()
    image_settings.dpi = dpi
    image_settings.backgroundColor = QColor(255, 255, 255)  # Black background
    exporter.exportToImage(output_path, image_settings)
    
    if verbose:
        print(f"Map fragment exported to {output_path}")
        print(f'Image size: {width_mm * dpi / 25.4:.0f}x'
              f'{height_mm * dpi / 25.4:.0f} pixels')


# First region
# start_point = (4298995,7318038)
# end_point = (4470188,7493475)
# n_y_regions = 82
# n_x_regions = 84
# x_side_size = 2048
# y_side_size = 2048
# layer_config = {
#     "road": [255, 0, 0],
#     "parking": [255, 0, 0],
#     "trees": [0, 255, 0],
#     'water': [0, 0, 255],
#     'waterway': [0, 0, 255],
# }

# Region for demonstration
start_point = (4118776, 6224502)
end_point = (4163160, 6257337)
y_side_size = 3240
x_side_size = 5760
n_x_regions = (end_point[0] - start_point[0]) // x_side_size
n_y_regions = 4

out_dir = Path('exported_regions')
layer_config = {
    "demonstration_roads": [255, 0, 0],
    "demonstration_trees": [0, 255, 0],
}
lat_lon = False
dpi = 300
verbose = True
m_in_px = 3
scale = round(dpi * m_in_px * 1000 / 25.4)
# layer_name = 'Bing_satellite'
layer_name = 'ESRI_satellite'

export_osm = True
export_satellite = True

# Export OSM
if export_osm:
    osm_dir = out_dir / 'osm'
    if osm_dir.exists():
        input(f'osm_dir уже существует {osm_dir}')
    else:
        osm_dir.mkdir(parents=True)
    for i in range(n_x_regions):  # iterate along x (from left to right)
        for j in range(n_y_regions):  # iterate along y (from down to up)
            point1 = (start_point[0] + x_side_size * i,
                      start_point[1] + y_side_size * j)
            point2 = (start_point[0] + x_side_size * (i + 1),
                      start_point[1] + y_side_size * (j + 1))
            img_pth = str(osm_dir /
                          f'osm_{point1[0]}_{point1[1]}_'
                          f'{point2[0]}_{point2[1]}.png')
            export_vector_map_fragment(
                layer_config, point1, point2, scale, img_pth,
                dpi=dpi, lat_lon=lat_lon, verbose=verbose)

# Export satellite
if export_satellite:
    satellite_dir = out_dir / layer_name
    if satellite_dir.exists():
        input(f'satellite_dir уже существует {str(satellite_dir)}')
    else:
        satellite_dir.mkdir(parents=True)
    for i in range(n_x_regions):  # iterate along x (from left to right)
        for j in range(n_y_regions):  # iterate along y (from down to up)
            point1 = (start_point[0] + x_side_size * i,
                      start_point[1] + y_side_size * j)
            point2 = (start_point[0] + x_side_size * (i + 1),
                      start_point[1] + y_side_size * (j + 1))
            img_pth = str(satellite_dir /
                          f'{layer_name}_{point1[0]}_{point1[1]}_'
                          f'{point2[0]}_{point2[1]}.png')
            export_satellite_map_fragment(
                layer_name, point1, point2, scale, img_pth,
                dpi=dpi, lat_lon=lat_lon, verbose=verbose)
