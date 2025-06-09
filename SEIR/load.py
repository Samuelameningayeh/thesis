import os
import pandas as pd
import geopandas as gpd
from typing import List, Optional

def merge_shapefiles_in_folder(
    folder_path: str,
    output_path: Optional[str] = None,
    target_crs: Optional[str] = None,
    merge_method: str = "concat",
    common_columns_only: bool = False
) -> gpd.GeoDataFrame:
    """
    Find and merge all shapefiles in a folder.
    
    Parameters:
        folder_path (str): Path to folder containing shapefiles
        output_path (str, optional): Path to save merged shapefile. If None, won't save.
        target_crs (str, optional): CRS to use for all files (e.g., 'EPSG:4326'). If None, uses first file's CRS.
        merge_method (str): How to merge - 'concat', 'union', or 'intersection'
        common_columns_only (bool): If True, only keeps columns present in all files
    
    Returns:
        geopandas.GeoDataFrame: Merged GeoDataFrame
    
    Raises:
        ValueError: If no shapefiles found or invalid merge method
    """
    # Find all .shp files in folder
    shp_files = [f for f in os.listdir(folder_path) if f.endswith('.shp')]
    if not shp_files:
        raise ValueError(f"No shapefiles found in {folder_path}")
    
    print(f"Found {len(shp_files)} shapefiles to merge")
    
    # Load all shapefiles
    gdfs = []
    for shp_file in shp_files:
        file_path = os.path.join(folder_path, shp_file)
        gdf = gpd.read_file(file_path)
        gdfs.append(gdf)
        print(f"Loaded {shp_file} with {len(gdf)} features")
    
    # Determine target CRS if not specified
    if target_crs is None:
        target_crs = gdfs[0].crs
        print(f"Using CRS from first file: {target_crs}")
    
    # Standardize CRS and validate geometries
    for i, gdf in enumerate(gdfs):
        if gdf.crs != target_crs:
            gdfs[i] = gdf.to_crs(target_crs)
        gdfs[i].geometry = gdfs[i].geometry.make_valid()
    
    # Handle different merge methods
    if merge_method == "concat":
        if common_columns_only:
            common_cols = list(set.intersection(*[set(gdf.columns) for gdf in gdfs]))
            gdfs = [gdf[common_cols] for gdf in gdfs]
        
        merged = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True),
            crs=target_crs
        )
    
    elif merge_method in ["union", "intersection"]:
        from functools import reduce
        merged = reduce(
            lambda x, y: gpd.overlay(x, y, how=merge_method),
            gdfs
        )
        merged.crs = target_crs
    
    else:
        raise ValueError("merge_method must be 'concat', 'union', or 'intersection'")
    
    # Remove any null geometries that may have been created
    merged = merged[~merged.geometry.is_empty]
    
    # Save if output path specified
    if output_path:
        merged.to_file(output_path)
        print(f"Merged shapefile saved to {output_path}")
    
    return merged