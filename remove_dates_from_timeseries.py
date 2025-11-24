#!/usr/bin/env python3
"""
Remove specific dates from MintPy timeseries HDF5 files.

This program removes specified dates from a timeseries file by reconstructing 
the file without the unwanted dates, preserving all metadata and structure.

Usage: python remove_timeseries_dates.py input_timeseries.h5 YYYYMMDD [YYYYMMDD2 ...]
Example: python remove_timeseries_dates.py vertical_timeseries.h5 20250801 20250815
"""

import os
import sys
import numpy as np
import h5py
from mintpy.utils import readfile, writefile
from mintpy.objects import timeseries


def read_timeseries_info(ts_file):
    """
    Read basic information from timeseries file.
    
    Parameters:
        ts_file (str): Path to timeseries HDF5 file
        
    Returns:
        tuple: (date_list, metadata, data_shape)
    """
    print(f"Reading timeseries information from: {ts_file}")
    
    # Read metadata
    metadata = readfile.read_attribute(ts_file)
    
    # Read date list
    with h5py.File(ts_file, 'r') as f:
        dates = f['date'][:]
        if hasattr(dates[0], 'decode'):
            date_list = [date.decode('utf-8') for date in dates]
        else:
            date_list = [str(date) for date in dates]
        
        # Get data shape
        data_shape = f['timeseries'].shape
    
    print(f"  Number of dates: {len(date_list)}")
    print(f"  Date range: {date_list[0]} to {date_list[-1]}")
    print(f"  Data shape: {data_shape}")
    
    return date_list, metadata, data_shape


def validate_dates_to_remove(date_list, dates_to_remove):
    """
    Validate that specified dates exist in the timeseries.
    
    Parameters:
        date_list (list): List of all dates in timeseries
        dates_to_remove (list): List of dates to remove
        
    Returns:
        list: Validated list of dates that exist and can be removed
    """
    valid_dates = []
    invalid_dates = []
    
    for date in dates_to_remove:
        if date in date_list:
            valid_dates.append(date)
        else:
            invalid_dates.append(date)
    
    if invalid_dates:
        print(f"WARNING: The following dates do not exist in the file: {invalid_dates}")
        print(f"Available dates: {date_list}")
    
    if not valid_dates:
        raise ValueError("No valid dates found to remove!")
    
    print(f"Dates to be removed: {valid_dates}")
    return valid_dates


def create_date_mask(date_list, dates_to_remove):
    """
    Create boolean mask for dates to keep.
    
    Parameters:
        date_list (list): Complete list of dates
        dates_to_remove (list): Dates to exclude
        
    Returns:
        numpy.ndarray: Boolean mask where True means keep the date
    """
    keep_mask = np.ones(len(date_list), dtype=bool)
    
    for date in dates_to_remove:
        if date in date_list:
            idx = date_list.index(date)
            keep_mask[idx] = False
    
    return keep_mask


def remove_dates_from_timeseries(input_file, dates_to_remove, output_file=None):
    """
    Remove specific dates from timeseries file and save result.
    
    Parameters:
        input_file (str): Path to input timeseries file
        dates_to_remove (list): List of dates to remove (YYYYMMDD format)
        output_file (str): Path for output file (optional)
        
    Returns:
        str: Path to output file
    """
    # Set output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_filtered.h5"
    
    # Read timeseries information
    date_list, metadata, data_shape = read_timeseries_info(input_file)
    
    # Validate dates to remove
    valid_dates_to_remove = validate_dates_to_remove(date_list, dates_to_remove)
    
    # Create mask for dates to keep
    keep_mask = create_date_mask(date_list, valid_dates_to_remove)
    
    # Calculate new dimensions
    new_date_list = [date for i, date in enumerate(date_list) if keep_mask[i]]
    num_dates_removed = len(date_list) - len(new_date_list)
    
    print(f"\nProcessing:")
    print(f"  Original number of dates: {len(date_list)}")
    print(f"  Dates to remove: {num_dates_removed}")
    print(f"  Remaining dates: {len(new_date_list)}")
    
    # Read and filter the timeseries data
    print(f"\nReading timeseries data...")
    with h5py.File(input_file, 'r') as f:
        # Read all timeseries data
        timeseries_data = f['timeseries'][:]
        
        # Apply temporal mask to keep only desired dates
        filtered_data = timeseries_data[keep_mask, :, :]
        
        print(f"  Original data shape: {timeseries_data.shape}")
        print(f"  Filtered data shape: {filtered_data.shape}")
    
    # Update metadata
    updated_metadata = metadata.copy()
    updated_metadata['START_DATE'] = new_date_list[0]
    updated_metadata['END_DATE'] = new_date_list[-1]
    
    # Write filtered timeseries to new file
    print(f"\nWriting filtered timeseries to: {output_file}")
    ts_obj = timeseries(output_file)
    ts_obj.write2hdf5(
        data=filtered_data,
        dates=new_date_list,
        metadata=updated_metadata,
        refFile=input_file
    )
    
    print(f"Successfully created filtered timeseries file!")
    print(f"Removed {num_dates_removed} dates from the timeseries")
    
    return output_file


def main():
    """Main function for command line execution."""
    if len(sys.argv) < 3:
        print("Usage: python remove_timeseries_dates.py <input_timeseries.h5> <date1> [date2] ...")
        print("Example: python remove_timeseries_dates.py vertical_timeseries.h5 20250801 20250815")
        print("\nNote: Dates should be in YYYYMMDD format")
        sys.exit(1)
    
    input_file = sys.argv[1]
    dates_to_remove = sys.argv[2:]
    
    # Validate input file
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Validate date format
    for date in dates_to_remove:
        if len(date) != 8 or not date.isdigit():
            raise ValueError(f"Invalid date format: {date}. Expected YYYYMMDD format.")
    
    print("=== MintPy Timeseries Date Removal Tool ===")
    print(f"Input file: {input_file}")
    print(f"Dates to remove: {dates_to_remove}")
    
    try:
        output_file = remove_dates_from_timeseries(input_file, dates_to_remove)
        print(f"\n=== Processing completed successfully! ===")
        print(f"Output file: {output_file}")
        
        # Display verification information
        print(f"\nVerification:")
        print(f"To verify the results, you can use:")
        print(f"  info.py {output_file} --date")
        print(f"  view.py {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()