import os
import pandas as pd
import geopandas as gpd

def load_data(
        path: str,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        shapefile: bool = False,
):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
        path (str): Path to the CSV file.
        delimiter (str): Delimiter used in the CSV file. Default is ','. 
        encoding (str): Encoding of the CSV file. Default is 'utf-8'.
        shapefile (bool): If True, load as a shapefile using geopandas. Default is False.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        print(f"Loading data from {path}...Please wait!")
        if shapefile:
            # Load shapefile using geopandas
            df = gpd.read_file(path)
            print(f"Done loading data from {path}!!!\n There are {len(df)} rows of observations.\n There are {len(df.columns)} columns")
            return df
        else:
            # Load CSV file using pandas
            df = pd.read_csv(path, delimiter=delimiter, encoding=encoding)
            print(f"Done loading data from {path}!!!\n There are {len(df)} rows of observations.\n There are {len(df.columns)} columns")
            return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None