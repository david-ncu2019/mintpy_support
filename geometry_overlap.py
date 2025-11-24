#!/usr/bin/env python3
"""
MintPy program to find common bounding box between ascending and descending 
geometry files and export overlapped pixels to a new HDF file.

Author: Your Name
Date: 2025
"""

import os
import sys
import numpy as np
import h5py
from mintpy.utils import readfile, writefile, utils as ut
from mintpy.objects.coord import coordinate


def read_geometry_metadata(geom_file):
    """
    Read geometry file metadata and create coordinate object.
    
    Parameters:
        geom_file (str): Path to geometry file
        
    Returns:
        tuple: (metadata dict, coordinate object)
    """
    print(f"Reading metadata from: {geom_file}")
    
    # Read file attributes (metadata)
    metadata = readfile.read_attribute(geom_file)
    
    # Create coordinate object for spatial transformations
    coord_obj = coordinate(metadata, lookup_file=geom_file)
    coord_obj.open()
    
    return metadata, coord_obj


def get_geographic_bounds(metadata, coord_obj):
    """
    Calculate geographic bounds (lat/lon) from geometry file.
    
    Parameters:
        metadata (dict): File metadata
        coord_obj: MintPy coordinate object
        
    Returns:
        tuple: (south, north, west, east) in decimal degrees
    """
    # Get image dimensions
    length = int(metadata['LENGTH'])
    width = int(metadata['WIDTH'])
    
    # Define pixel box covering entire image (x0, y0, x1, y1)
    pixel_box = (0, 0, width, length)
    
    # Convert to geographic coordinates
    if coord_obj.geocoded:
        # For geocoded data, use direct calculation
        lat0 = float(metadata['Y_FIRST'])
        lon0 = float(metadata['X_FIRST'])
        lat_step = float(metadata['Y_STEP'])
        lon_step = float(metadata['X_STEP'])
        
        south = lat0 + lat_step * length
        north = lat0
        west = lon0
        east = lon0 + lon_step * width
    else:
        # For radar coordinates, use coordinate transformation
        geo_box = coord_obj.box_pixel2geo(pixel_box)
        if geo_box is not None:
            west, north, east, south = geo_box
        else:
            raise ValueError("Could not convert radar coordinates to geographic coordinates")
    
    return south, north, west, east


def calculate_common_bounding_box(asc_bounds, desc_bounds):
    """
    Calculate the intersection (common area) of two geographic bounding boxes.
    
    Parameters:
        asc_bounds (tuple): (south, north, west, east) for ascending pass
        desc_bounds (tuple): (south, north, west, east) for descending pass
        
    Returns:
        tuple: (south, north, west, east) of common area, or None if no overlap
    """
    asc_s, asc_n, asc_w, asc_e = asc_bounds
    desc_s, desc_n, desc_w, desc_e = desc_bounds
    
    # Calculate intersection bounds
    # Common area is the intersection of both bounding boxes
    common_south = max(asc_s, desc_s)  # Maximum of southern boundaries
    common_north = min(asc_n, desc_n)  # Minimum of northern boundaries
    common_west = max(asc_w, desc_w)   # Maximum of western boundaries
    common_east = min(asc_e, desc_e)   # Minimum of eastern boundaries
    
    # Check if there is actual overlap
    if common_south >= common_north or common_west >= common_east:
        print("WARNING: No geographic overlap found between files!")
        return None
    
    return common_south, common_north, common_west, common_east


def convert_common_bounds_to_pixel_boxes(common_bounds, asc_coord, desc_coord):
    """
    Convert common geographic bounds back to pixel coordinates for each file.
    
    Parameters:
        common_bounds (tuple): (south, north, west, east) of overlap area
        asc_coord: Coordinate object for ascending file
        desc_coord: Coordinate object for descending file
        
    Returns:
        tuple: (asc_pixel_box, desc_pixel_box) each as (x0, y0, x1, y1)
    """
    south, north, west, east = common_bounds
    geo_box = (west, north, east, south)  # Convert to (W, N, E, S) format
    
    # Convert to pixel coordinates for each file
    if asc_coord.geocoded:
        asc_pixel_box = asc_coord.box_geo2pixel(geo_box)
    else:
        asc_pixel_box = asc_coord.bbox_geo2radar(geo_box)
        
    if desc_coord.geocoded:
        desc_pixel_box = desc_coord.box_geo2pixel(geo_box)
    else:
        desc_pixel_box = desc_coord.bbox_geo2radar(geo_box)
    
    # Ensure pixel boxes are within data coverage
    asc_pixel_box = asc_coord.check_box_within_data_coverage(asc_pixel_box)
    desc_pixel_box = desc_coord.check_box_within_data_coverage(desc_pixel_box)
    
    return asc_pixel_box, desc_pixel_box


def read_overlapped_data(geom_file, pixel_box, dataset_names=None):
    """
    Read data from geometry file within specified pixel box.
    
    Parameters:
        geom_file (str): Path to geometry file
        pixel_box (tuple): (x0, y0, x1, y1) pixel coordinates
        dataset_names (list): List of dataset names to read, None for all
        
    Returns:
        dict: Dictionary with dataset names as keys and numpy arrays as values
    """
    print(f"Reading data from {geom_file} with box {pixel_box}")
    
    if dataset_names is None:
        # Read available datasets from file
        with h5py.File(geom_file, 'r') as f:
            dataset_names = list(f.keys())
            # Filter for commonly used geometry datasets
            common_datasets = ['height', 'incidenceAngle', 'azimuthAngle', 
                             'slantRangeDistance', 'latitude', 'longitude']
            dataset_names = [name for name in dataset_names if name in common_datasets]
    
    data_dict = {}
    for dataset in dataset_names:
        try:
            data, _ = readfile.read(geom_file, datasetName=dataset, box=pixel_box)
            data_dict[dataset] = data
            print(f"  Successfully read dataset: {dataset} with shape {data.shape}")
        except Exception as e:
            print(f"  Warning: Could not read dataset {dataset}: {e}")
    
    return data_dict


def create_output_hdf_file(output_file, asc_data, desc_data, asc_metadata, desc_metadata, 
                          common_bounds, asc_pixel_box, desc_pixel_box):
    """
    Create new HDF file with overlapped data from both ascending and descending files.
    
    Parameters:
        output_file (str): Path for output HDF file
        asc_data (dict): Data dictionary from ascending file
        desc_data (dict): Data dictionary from descending file
        asc_metadata (dict): Metadata from ascending file
        desc_metadata (dict): Metadata from descending file
        common_bounds (tuple): Geographic bounds of common area
        asc_pixel_box (tuple): Pixel box used for ascending file
        desc_pixel_box (tuple): Pixel box used for descending file
    """
    print(f"Creating output file: {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Get dimensions from the data
        if asc_data:
            sample_data = next(iter(asc_data.values()))
            length, width = sample_data.shape
        else:
            raise ValueError("No data available to determine output dimensions")
        
        # Write datasets at root level with prefixes for MintPy compatibility
        # This allows MintPy to recognize the file while preserving asc/desc distinction
        for dataset_name, data in asc_data.items():
            # Store ascending data with "asc_" prefix
            f.create_dataset(f'asc_{dataset_name}', data=data, compression='gzip')
            
        for dataset_name, data in desc_data.items():
            # Store descending data with "desc_" prefix  
            f.create_dataset(f'desc_{dataset_name}', data=data, compression='gzip')
        
        # For MintPy compatibility, also create standard geometry datasets
        # Use ascending data as primary datasets (could also use average or other combination)
        primary_datasets = ['height', 'incidenceAngle', 'azimuthAngle', 'slantRangeDistance']
        
        for dataset_name in primary_datasets:
            if dataset_name in asc_data:
                f.create_dataset(dataset_name, data=asc_data[dataset_name], compression='gzip')
                print(f"  Created primary dataset: {dataset_name}")
        
        # Calculate coordinate information for the overlapped area
        south, north, west, east = common_bounds
        
        # Essential MintPy attributes at root level
        f.attrs['WIDTH'] = str(width)
        f.attrs['LENGTH'] = str(length)
        f.attrs['FILE_TYPE'] = 'geometry'
        
        # Add coordinate system information
        if 'Y_FIRST' in asc_metadata:
            # For geocoded data, calculate the coordinate parameters for overlapped area
            y_step = float(asc_metadata.get('Y_STEP', 0))
            x_step = float(asc_metadata.get('X_STEP', 0))
            
            f.attrs['Y_FIRST'] = str(north)
            f.attrs['X_FIRST'] = str(west)  
            f.attrs['Y_STEP'] = str(y_step)
            f.attrs['X_STEP'] = str(x_step)
            f.attrs['UNIT'] = 'degrees'
        
        # Copy other important metadata from source files
        important_attrs = ['EARTH_RADIUS', 'HEIGHT', 'PROCESSOR', 'WAVELENGTH', 
                          'ANTENNA_SIDE', 'ORBIT_DIRECTION', 'PLATFORM', 'EPSG']
        
        for attr in important_attrs:
            if attr in asc_metadata:
                f.attrs[attr] = str(asc_metadata[attr])
            elif attr in desc_metadata:
                f.attrs[attr] = str(desc_metadata[attr])
        
        # Additional overlap-specific metadata
        f.attrs['DESCRIPTION'] = 'Overlapped pixels from ascending and descending geometry files'
        f.attrs['COMMON_BOUNDS_SOUTH'] = str(south)
        f.attrs['COMMON_BOUNDS_NORTH'] = str(north)
        f.attrs['COMMON_BOUNDS_WEST'] = str(west)
        f.attrs['COMMON_BOUNDS_EAST'] = str(east)
        f.attrs['ASC_PIXEL_BOX'] = str(asc_pixel_box)
        f.attrs['DESC_PIXEL_BOX'] = str(desc_pixel_box)
        
        # Create metadata groups for detailed original metadata
        asc_meta_group = f.create_group('metadata_ascending')
        desc_meta_group = f.create_group('metadata_descending')
        
        # Add original metadata to metadata groups
        for key, value in asc_metadata.items():
            try:
                asc_meta_group.attrs[key] = str(value)
            except (TypeError, ValueError):
                asc_meta_group.attrs[key] = str(value)
                
        for key, value in desc_metadata.items():
            try:
                desc_meta_group.attrs[key] = str(value)
            except (TypeError, ValueError):
                desc_meta_group.attrs[key] = str(value)
    
    print(f"Successfully created MintPy-compatible output file with overlapped data")
    print(f"Available datasets:")
    print(f"  - Primary datasets: {primary_datasets}")
    print(f"  - Ascending datasets: asc_height, asc_incidenceAngle, asc_azimuthAngle, asc_slantRangeDistance")
    print(f"  - Descending datasets: desc_height, desc_incidenceAngle, desc_azimuthAngle, desc_slantRangeDistance")


def main(asc_geom_file, desc_geom_file, output_file, dataset_names=None):
    """
    Main function to find common bounding box and export overlapped pixels.
    
    Parameters:
        asc_geom_file (str): Path to ascending geometry file
        desc_geom_file (str): Path to descending geometry file
        output_file (str): Path for output HDF file
        dataset_names (list): List of datasets to extract, None for all available
    """
    print("=== MintPy Common Bounding Box and Overlap Export ===")
    
    # Step 1: Check input files exist
    for file_path in [asc_geom_file, desc_geom_file]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Step 2: Read metadata and create coordinate objects
    print("\nStep 1: Reading file metadata...")
    asc_metadata, asc_coord = read_geometry_metadata(asc_geom_file)
    desc_metadata, desc_coord = read_geometry_metadata(desc_geom_file)
    
    # Step 3: Calculate geographic bounds for each file
    print("\nStep 2: Calculating geographic bounds...")
    asc_bounds = get_geographic_bounds(asc_metadata, asc_coord)
    desc_bounds = get_geographic_bounds(desc_metadata, desc_coord)
    
    print(f"Ascending bounds (S, N, W, E): {asc_bounds}")
    print(f"Descending bounds (S, N, W, E): {desc_bounds}")
    
    # Step 4: Find common bounding box
    print("\nStep 3: Finding common bounding box...")
    common_bounds = calculate_common_bounding_box(asc_bounds, desc_bounds)
    
    if common_bounds is None:
        raise ValueError("No overlap found between ascending and descending files!")
    
    print(f"Common bounds (S, N, W, E): {common_bounds}")
    
    # Step 5: Convert common bounds to pixel coordinates
    print("\nStep 4: Converting to pixel coordinates...")
    asc_pixel_box, desc_pixel_box = convert_common_bounds_to_pixel_boxes(
        common_bounds, asc_coord, desc_coord)
    
    print(f"Ascending pixel box (x0, y0, x1, y1): {asc_pixel_box}")
    print(f"Descending pixel box (x0, y0, x1, y1): {desc_pixel_box}")
    
    # Step 6: Read overlapped data
    print("\nStep 5: Reading overlapped data...")
    asc_data = read_overlapped_data(asc_geom_file, asc_pixel_box, dataset_names)
    desc_data = read_overlapped_data(desc_geom_file, desc_pixel_box, dataset_names)
    
    # Step 7: Create output HDF file
    print("\nStep 6: Creating output HDF file...")
    create_output_hdf_file(output_file, asc_data, desc_data, asc_metadata, desc_metadata,
                          common_bounds, asc_pixel_box, desc_pixel_box)
    
    print(f"\n=== Processing completed successfully! ===")
    print(f"Output file: {output_file}")
    print(f"Common area size: {len(asc_data)} datasets")


if __name__ == '__main__':
    """
    Example usage:
    python mintpy_overlap_export.py asc_geometry.h5 desc_geometry.h5 overlap_output.h5
    """
    
    if len(sys.argv) < 4:
        print("Usage: python script.py <asc_geometry_file> <desc_geometry_file> <output_file>")
        print("Example: python script.py inputs/geometryRadarAsc.h5 inputs/geometryRadarDesc.h5 overlap_data.h5")
        sys.exit(1)
    
    asc_file = sys.argv[1]
    desc_file = sys.argv[2] 
    output_file = sys.argv[3]
    
    # Optional: specify which datasets to extract
    # dataset_names = ['height', 'incidenceAngle', 'azimuthAngle', 'slantRangeDistance']
    dataset_names = None  # Extract all available datasets
    
    try:
        main(asc_file, desc_file, output_file, dataset_names)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)