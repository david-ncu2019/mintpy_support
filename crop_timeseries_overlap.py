#!/usr/bin/env python3
"""
MintPy program to find overlapping area between ascending and descending timeseries files
and crop both files to their common spatial coverage.

This program leverages MintPy's built-in spatial overlap and subsetting functions to ensure
compatibility and consistency with the MintPy processing framework.

Author: Based on MintPy built-in functions
Date: 2025
"""

import os
import sys
import numpy as np
from mintpy.utils import readfile, utils as ut, attribute as attr
from mintpy.subset import get_box_overlap_index, subset_file
from mintpy.objects import timeseries


def read_timeseries_spatial_info(ts_file):
    """
    Read spatial information from timeseries file using MintPy built-in functions.
    
    Parameters:
        ts_file (str): Path to timeseries file
        
    Returns:
        tuple: (metadata dict, coordinate object, pixel_box, geo_box)
    """
    print(f"Reading spatial information from: {os.path.basename(ts_file)}")
    
    # Read metadata using MintPy built-in function
    metadata = readfile.read_attribute(ts_file)
    
    # Create coordinate object using MintPy built-in utility
    coord_obj = ut.coordinate(metadata)
    coord_obj.open()
    
    # Get full coverage pixel box
    width = int(metadata['WIDTH'])
    length = int(metadata['LENGTH'])
    pixel_box = (0, 0, width, length)
    
    # Convert to geographic box using MintPy built-in function
    geo_box = coord_obj.box_pixel2geo(pixel_box)
    
    print(f"  Dimensions: {width} x {length} (width x length)")
    print(f"  Pixel box: {pixel_box}")
    print(f"  Geographic box: {geo_box}")
    
    return metadata, coord_obj, pixel_box, geo_box


def calculate_spatial_overlap(asc_info, desc_info):
    """
    Calculate spatial overlap using MintPy built-in overlap functions.
    
    Parameters:
        asc_info (tuple): (metadata, coord_obj, pixel_box, geo_box) for ascending
        desc_info (tuple): (metadata, coord_obj, pixel_box, geo_box) for descending
        
    Returns:
        tuple: (common_geo_box, asc_crop_pix_box, desc_crop_pix_box)
    """
    asc_meta, asc_coord, asc_pix_box, asc_geo_box = asc_info
    desc_meta, desc_coord, desc_pix_box, desc_geo_box = desc_info
    
    print("\nCalculating spatial overlap...")
    
    # Check if both files are in geocoded coordinates
    if asc_geo_box is None or desc_geo_box is None:
        raise ValueError("Both timeseries files must be in geocoded coordinates for overlap calculation")
    
    # Calculate common geographic bounds using coordinate intersection
    # geo_box format: (west, north, east, south)
    asc_west, asc_north, asc_east, asc_south = asc_geo_box
    desc_west, desc_north, desc_east, desc_south = desc_geo_box
    
    # Find intersection bounds
    common_west = max(asc_west, desc_west)
    common_east = min(asc_east, desc_east)
    common_north = min(asc_north, desc_north)
    common_south = max(asc_south, desc_south)
    
    # Check for valid overlap
    if common_west >= common_east or common_south >= common_north:
        raise ValueError("No spatial overlap found between ascending and descending files")
    
    common_geo_box = (common_west, common_north, common_east, common_south)
    
    print(f"Ascending geographic bounds: {asc_geo_box}")
    print(f"Descending geographic bounds: {desc_geo_box}")
    print(f"Common overlap bounds: {common_geo_box}")
    
    # Convert common geographic bounds to pixel boxes for each file
    asc_crop_pix_box = asc_coord.box_geo2pixel(common_geo_box)
    desc_crop_pix_box = desc_coord.box_geo2pixel(common_geo_box)
    
    # Ensure pixel boxes are within data coverage using MintPy built-in function
    asc_crop_pix_box = asc_coord.check_box_within_data_coverage(asc_crop_pix_box)
    desc_crop_pix_box = desc_coord.check_box_within_data_coverage(desc_crop_pix_box)
    
    print(f"Ascending crop pixel box: {asc_crop_pix_box}")
    print(f"Descending crop pixel box: {desc_crop_pix_box}")
    
    return common_geo_box, asc_crop_pix_box, desc_crop_pix_box


def crop_timeseries_to_overlap(ts_file, pixel_box, output_file):
    """
    Crop timeseries file to specified pixel box using MintPy built-in subset function.
    
    Parameters:
        ts_file (str): Path to input timeseries file
        pixel_box (tuple): Pixel box for cropping (x0, y0, x1, y1)
        output_file (str): Path for output cropped file
        
    Returns:
        str: Path to output file
    """
    print(f"\nCropping {os.path.basename(ts_file)} to overlap region...")
    
    # Create subset dictionary using pixel coordinates
    x0, y0, x1, y1 = pixel_box
    subset_dict = {
        'subset_x': [x0, x1],
        'subset_y': [y0, y1],
        'subset_lat': None,
        'subset_lon': None,
        'fill_value': np.nan
    }
    
    # Use MintPy's built-in subset_file function
    try:
        result_file = subset_file(ts_file, subset_dict, out_file=output_file)
        print(f"Successfully cropped file: {os.path.basename(result_file)}")
        return result_file
    except Exception as e:
        print(f"Error cropping file: {e}")
        raise


def verify_cropped_files(asc_file, desc_file):
    """
    Verify that cropped files have identical spatial dimensions and coverage.
    
    Parameters:
        asc_file (str): Path to cropped ascending file
        desc_file (str): Path to cropped descending file
        
    Returns:
        bool: True if files are spatially aligned
    """
    print("\nVerifying cropped files...")
    
    # Read metadata from both files
    asc_meta = readfile.read_attribute(asc_file)
    desc_meta = readfile.read_attribute(desc_file)
    
    # Check dimensions
    asc_dims = (int(asc_meta['WIDTH']), int(asc_meta['LENGTH']))
    desc_dims = (int(desc_meta['WIDTH']), int(desc_meta['LENGTH']))
    
    # Check coordinate parameters
    coord_params = ['X_FIRST', 'Y_FIRST', 'X_STEP', 'Y_STEP']
    coord_match = True
    
    for param in coord_params:
        if param in asc_meta and param in desc_meta:
            asc_val = float(asc_meta[param])
            desc_val = float(desc_meta[param])
            if abs(asc_val - desc_val) > 1e-6:  # Allow for small numerical differences
                coord_match = False
                print(f"  WARNING: {param} mismatch - Asc: {asc_val}, Desc: {desc_val}")
    
    # Report verification results
    dims_match = asc_dims == desc_dims
    
    print(f"  Ascending dimensions: {asc_dims[0]} x {asc_dims[1]}")
    print(f"  Descending dimensions: {desc_dims[0]} x {desc_dims[1]}")
    print(f"  Dimensions match: {dims_match}")
    print(f"  Coordinates match: {coord_match}")
    
    if dims_match and coord_match:
        print("  ✓ Files are properly aligned for comparative analysis")
        return True
    else:
        print("  ✗ Files may have alignment issues")
        return False


def main(asc_ts_file, desc_ts_file, output_dir=None):
    """
    Main function to crop ascending and descending timeseries to their overlap area.
    
    Parameters:
        asc_ts_file (str): Path to ascending timeseries file
        desc_ts_file (str): Path to descending timeseries file
        output_dir (str): Directory for output files (optional)
        
    Returns:
        tuple: (cropped_asc_file, cropped_desc_file)
    """
    print("=== MintPy Timeseries Overlap Cropping Tool ===")
    
    # Validate input files
    for ts_file in [asc_ts_file, desc_ts_file]:
        if not os.path.isfile(ts_file):
            raise FileNotFoundError(f"Timeseries file not found: {ts_file}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(asc_ts_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    asc_basename = os.path.splitext(os.path.basename(asc_ts_file))[0]
    desc_basename = os.path.splitext(os.path.basename(desc_ts_file))[0]
    
    cropped_asc_file = os.path.join(output_dir, f"cropped_{asc_basename}.h5")
    cropped_desc_file = os.path.join(output_dir, f"cropped_{desc_basename}.h5")
    
    print(f"Input ascending file: {asc_ts_file}")
    print(f"Input descending file: {desc_ts_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Read spatial information from both files
        asc_info = read_timeseries_spatial_info(asc_ts_file)
        desc_info = read_timeseries_spatial_info(desc_ts_file)
        
        # Calculate spatial overlap
        common_geo_box, asc_crop_box, desc_crop_box = calculate_spatial_overlap(asc_info, desc_info)
        
        # Calculate overlap area
        overlap_width = asc_crop_box[2] - asc_crop_box[0]
        overlap_length = asc_crop_box[3] - asc_crop_box[1]
        overlap_area_pixels = overlap_width * overlap_length
        
        print(f"Overlap area dimensions: {overlap_width} x {overlap_length} pixels")
        print(f"Total overlap area: {overlap_area_pixels:,} pixels")
        
        # Crop both files to overlap region
        result_asc = crop_timeseries_to_overlap(asc_ts_file, asc_crop_box, cropped_asc_file)
        result_desc = crop_timeseries_to_overlap(desc_ts_file, desc_crop_box, cropped_desc_file)
        
        # Verify results
        alignment_success = verify_cropped_files(result_asc, result_desc)
        
        print(f"\n=== Processing completed successfully ===")
        print(f"Cropped ascending file: {result_asc}")
        print(f"Cropped descending file: {result_desc}")
        
        if alignment_success:
            print("The cropped files are ready for comparative analysis")
        else:
            print("Warning: Manual verification of spatial alignment may be needed")
        
        return result_asc, result_desc
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == '__main__':
    """
    Command line usage:
    python crop_timeseries_overlap.py ascending_timeseries.h5 descending_timeseries.h5 [output_dir]
    """
    
    if len(sys.argv) < 3:
        print("Usage: python crop_timeseries_overlap.py <ascending_timeseries.h5> <descending_timeseries.h5> [output_dir]")
        print("Example: python crop_timeseries_overlap.py asc_timeseries.h5 desc_timeseries.h5 ./cropped_results/")
        sys.exit(1)
    
    asc_file = sys.argv[1]
    desc_file = sys.argv[2]
    output_directory = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        cropped_asc, cropped_desc = main(asc_file, desc_file, output_directory)
        print(f"\nOutput files created successfully:")
        print(f"  {cropped_asc}")
        print(f"  {cropped_desc}")
        
    except Exception as error:
        print(f"Processing failed: {error}")
        sys.exit(1)