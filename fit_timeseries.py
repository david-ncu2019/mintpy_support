#!/usr/bin/env python3
"""
Generate temporally-interpolated timeseries from sparse SAR observations using MintPy's temporal fitting functions.

This script performs model-based temporal interpolation by:
1. Fitting polynomial and periodic models to sparse SAR acquisition dates
2. Reconstructing displacement values at specified temporal intervals (daily or monthly)
3. Saving results to a new HDF5 file for further analysis

Sampling Options:
- Daily: Complete daily interpolation for detailed temporal analysis
- Monthly: End-of-month values for efficient storage and seasonal analysis

Author: Generated for MintPy temporal analysis workflow
"""

import os
import sys
import time
import datetime
import numpy as np
import h5py
from pathlib import Path

# Import MintPy modules
try:
    from mintpy.utils import readfile, writefile, ptime
    from mintpy.utils import time_func
    from mintpy.objects import timeseries, cluster
except ImportError as e:
    print(f"Error importing MintPy modules: {e}")
    print("Please ensure MintPy is properly installed and in your Python path.")
    sys.exit(1)


def create_daily_date_list(start_date, end_date):
    """
    Generate daily date list between start and end dates.
    
    Parameters:
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format
        
    Returns:
        list: Daily dates in YYYYMMDD format
    """
    start_dt = datetime.datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y%m%d')
    
    daily_dates = []
    current_dt = start_dt
    
    while current_dt <= end_dt:
        daily_dates.append(current_dt.strftime('%Y%m%d'))
        current_dt += datetime.timedelta(days=1)
    
    return daily_dates


def create_monthly_date_list(start_date, end_date):
    """
    Generate end-of-month date list between start and end dates.
    
    Parameters:
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format
        
    Returns:
        list: End-of-month dates in YYYYMMDD format
    """
    from calendar import monthrange
    
    start_dt = datetime.datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y%m%d')
    
    monthly_dates = []
    current_year = start_dt.year
    current_month = start_dt.month
    
    while True:
        # Get the last day of the current month
        last_day = monthrange(current_year, current_month)[1]
        end_of_month = datetime.datetime(current_year, current_month, last_day)
        
        # Only include if within our date range
        if end_of_month >= start_dt:
            if end_of_month <= end_dt:
                monthly_dates.append(end_of_month.strftime('%Y%m%d'))
            else:
                # Add the end date if it's not already an end-of-month date
                if end_dt.day != last_day or end_dt.month != current_month or end_dt.year != current_year:
                    monthly_dates.append(end_dt.strftime('%Y%m%d'))
                break
        
        # Move to next month
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
        
        # Safety check to prevent infinite loop
        if current_year > end_dt.year + 1:
            break
    
    return monthly_dates


def create_temporal_date_list(start_date, end_date, sampling_mode='daily'):
    """
    Generate temporal date list based on sampling mode.
    
    Parameters:
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format
        sampling_mode (str): 'daily' or 'monthly'
        
    Returns:
        list: Temporal dates in YYYYMMDD format
    """
    if sampling_mode == 'daily':
        return create_daily_date_list(start_date, end_date)
    elif sampling_mode == 'monthly':
        return create_monthly_date_list(start_date, end_date)
    else:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}. Use 'daily' or 'monthly'.")


def setup_temporal_model(poly_degree=1, periods=[1.0, 0.5]):
    """
    Configure temporal deformation model.
    
    Parameters:
        poly_degree (int): Polynomial degree (0=offset, 1=velocity, 2=acceleration)
        periods (list): Periodic components in years (1.0=annual, 0.5=semi-annual)
        
    Returns:
        dict: Model configuration dictionary
    """
    model = {
        'polynomial': poly_degree,
        'periodic': periods,
        'stepDate': [],
        'polyline': [],
        'exp': {},
        'log': {}
    }
    return model


def fit_timeseries_model(ts_file, model, max_memory=4.0, ref_date=None, ref_point=None):
    """
    Fit temporal model to sparse timeseries data.
    
    Parameters:
        ts_file (str): Path to input timeseries HDF5 file
        model (dict): Temporal model configuration
        max_memory (float): Maximum memory usage in GB
        ref_date (str): Reference date in YYYYMMDD format
        ref_point (tuple): Reference point coordinates (y, x)
        
    Returns:
        tuple: (model_parameters, metadata, original_dates, design_matrix)
    """
    print("Loading timeseries data and fitting temporal model...")
    
    # Initialize timeseries object
    ts_obj = timeseries(ts_file)
    ts_obj.open(print_msg=True)
    
    # Get basic information
    date_list = ts_obj.get_date_list()
    atr = ts_obj.metadata
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    num_date = len(date_list)
    seconds = float(atr.get('CENTER_LINE_UTC', 0))
    
    print(f"Timeseries dimensions: {num_date} dates, {length} x {width} pixels")
    print(f"Date range: {date_list[0]} to {date_list[-1]}")
    
    # Set reference date if not provided
    if ref_date is None:
        if 'REF_DATE' in atr:
            ref_date = atr['REF_DATE']
        else:
            ref_date = date_list[0]
            print(f"Warning: Using first date as reference: {ref_date}")
    
    # Get number of model parameters
    num_param = time_func.get_num_param(model)
    print(f"Temporal model parameters: {num_param}")
    
    # Create design matrix for original dates
    G_orig = time_func.get_design_matrix4time_func(
        date_list=date_list,
        model=model,
        ref_date=ref_date,
        seconds=seconds
    )
    
    # Calculate memory requirements and split into blocks if necessary
    memory_per_pixel = (num_date + num_param * 2) * 4 / (1024**3)  # GB
    total_memory = memory_per_pixel * length * width
    
    if total_memory > max_memory:
        num_blocks = int(np.ceil(total_memory / max_memory))
        print(f"Memory requirement ({total_memory:.1f} GB) exceeds limit ({max_memory} GB)")
        print(f"Processing in {num_blocks} blocks")
    else:
        num_blocks = 1
    
    # Split processing area into blocks
    box_list, num_blocks = cluster.split_box2sub_boxes(
        box=(0, 0, width, length),
        num_split=num_blocks,
        dimension='y',
        print_msg=True if num_blocks > 1 else False
    )
    
    # Initialize output arrays
    model_params = np.full((num_param, length, width), np.nan, dtype=np.float32)
    
    # Process each block
    for i, box in enumerate(box_list):
        if num_blocks > 1:
            print(f"\nProcessing block {i+1}/{num_blocks}")
        
        box_width = box[2] - box[0]
        box_length = box[3] - box[1]
        num_pixels = box_length * box_width
        
        # Read timeseries data for this block
        ts_data = readfile.read(ts_file, box=box)[0]
        
        # Apply spatial referencing if specified
        if ref_point is not None:
            ref_y, ref_x = ref_point
            if box[1] <= ref_y < box[3] and box[0] <= ref_x < box[2]:
                # Reference point is in this block
                local_ref_y = ref_y - box[1]
                local_ref_x = ref_x - box[0]
                ref_values = ts_data[:, local_ref_y, local_ref_x]
                ts_data -= ref_values.reshape(-1, 1, 1)
        
        # Apply temporal referencing
        if ref_date in date_list:
            ref_idx = date_list.index(ref_date)
            ts_data -= ts_data[ref_idx:ref_idx+1, :, :]
        
        # Reshape for processing
        ts_data = ts_data.reshape(num_date, -1)
        
        # Create mask for valid pixels
        ts_mean = np.nanmean(ts_data, axis=0)
        valid_mask = ~np.isnan(ts_mean)
        valid_pixels = np.sum(valid_mask)
        
        if valid_pixels == 0:
            print(f"Warning: No valid pixels in block {i+1}")
            continue
        
        print(f"Fitting model for {valid_pixels}/{num_pixels} valid pixels...")
        
        # Fit temporal model using MintPy's function
        try:
            G, m, residuals = time_func.estimate_time_func(
                model=model,
                date_list=date_list,
                dis_ts=ts_data[:, valid_mask],
                seconds=seconds
            )
            
            # Store results
            params_block = np.full((num_param, num_pixels), np.nan, dtype=np.float32)
            params_block[:, valid_mask] = m
            params_block = params_block.reshape(num_param, box_length, box_width)
            
            model_params[:, box[1]:box[3], box[0]:box[2]] = params_block
            
        except Exception as e:
            print(f"Error fitting model in block {i+1}: {e}")
            continue
    
    return model_params, atr, date_list, G_orig


def reconstruct_temporal_timeseries_blockwise(model_params, model, temporal_dates, atr, ref_date, output_file, sampling_mode='daily', max_memory=8.0):
    """
    Reconstruct temporally-interpolated timeseries using fitted model parameters with block-wise processing.
    
    Parameters:
        model_params (np.array): Fitted model parameters [num_param x length x width]
        model (dict): Temporal model configuration
        temporal_dates (list): Temporal date list in YYYYMMDD format
        atr (dict): Metadata from original timeseries
        ref_date (str): Reference date
        output_file (str): Output file path for incremental writing
        sampling_mode (str): 'daily' or 'monthly' sampling mode
        max_memory (float): Maximum memory usage in GB
        
    Returns:
        None: Results are written directly to file
    """
    print(f"\nReconstructing {sampling_mode} timeseries for {len(temporal_dates)} time points...")
    
    num_param, length, width = model_params.shape
    num_temporal = len(temporal_dates)
    seconds = float(atr.get('CENTER_LINE_UTC', 0))

    # Ensure reference date exists in temporal dates list
    if ref_date not in temporal_dates:
        print(f"Warning: Original reference date {ref_date} not in {sampling_mode} dates.")
        ref_date = temporal_dates[0]
        print(f"Using {ref_date} as reference date instead.")
    
    # Create design matrix for temporal dates
    G_temporal = time_func.get_design_matrix4time_func(
        date_list=temporal_dates,
        model=model,
        ref_date=ref_date,
        seconds=seconds
    )
    
    print(f"Temporal design matrix shape: {G_temporal.shape}")
    
    # Calculate memory requirements and determine block size
    memory_per_block = num_temporal * width * 4 / (1024**3)  # GB per row block
    rows_per_block = max(1, int(max_memory / memory_per_block))
    
    # Further limit block size if still too large
    max_rows_per_block = min(rows_per_block, 500)  # Cap at 500 rows per block for safety
    num_blocks = int(np.ceil(length / max_rows_per_block))
    
    print(f"Processing {length} rows in {num_blocks} blocks of ~{max_rows_per_block} rows each")
    print(f"Estimated memory per block: {num_temporal * max_rows_per_block * width * 4 / (1024**3):.2f} GB")
    
    # Create the output file first
    setup_output_file(output_file, temporal_dates, atr, model, sampling_mode)
    
    # Process in spatial blocks
    with h5py.File(output_file, 'r+') as f:
        timeseries_dset = f['timeseries']
        
        for block_idx in range(num_blocks):
            start_row = block_idx * max_rows_per_block
            end_row = min((block_idx + 1) * max_rows_per_block, length)
            block_height = end_row - start_row
            
            print(f"Processing spatial block {block_idx + 1}/{num_blocks}: rows {start_row}-{end_row-1}")
            
            # Extract model parameters for this block
            block_params = model_params[:, start_row:end_row, :]  # [num_param x block_height x width]
            
            # Initialize output for this block
            temporal_block = np.full((num_temporal, block_height, width), np.nan, dtype=np.float32)
            
            # Process row by row within the block
            for local_row in range(block_height):
                global_row = start_row + local_row
                
                # Extract parameters for this row
                row_params = block_params[:, local_row, :].T  # [width x num_param]
                
                # Find valid pixels in this row
                valid_pixels = ~np.isnan(row_params).any(axis=1)
                
                if np.any(valid_pixels):
                    # Reconstruct timeseries for valid pixels: G_temporal @ params
                    valid_params = row_params[valid_pixels, :]  # [valid_pixels x num_param]
                    temporal_values = G_temporal @ valid_params.T  # [num_temporal x valid_pixels]
                    
                    # Store results in block array
                    temporal_block[:, local_row, valid_pixels] = temporal_values
                
                # Progress reporting
                if (global_row + 1) % 200 == 0 or global_row == length - 1:
                    progress = (global_row + 1) / length * 100
                    print(f"  Progress: {progress:.1f}% ({global_row + 1}/{length} rows)")
            
            # Write this block to file
            timeseries_dset[:, start_row:end_row, :] = temporal_block
            print(f"  Block {block_idx + 1} written to file")
            
            # Clear memory
            del temporal_block, block_params
    
    print(f"{sampling_mode.capitalize()} timeseries reconstruction completed successfully!")


def setup_output_file(output_file, temporal_dates, atr, model, sampling_mode='daily'):
    """
    Create and initialize the output HDF5 file structure.
    
    Parameters:
        output_file (str): Output file path
        temporal_dates (list): Temporal date list
        atr (dict): Metadata dictionary
        model (dict): Model configuration for metadata
        sampling_mode (str): 'daily' or 'monthly' sampling mode
    """
    print(f"Setting up output file: {output_file}")
    
    # Prepare metadata
    atr_out = dict(atr)
    atr_out['FILE_TYPE'] = 'timeseries'
    atr_out['START_DATE'] = temporal_dates[0]
    atr_out['END_DATE'] = temporal_dates[-1]
    atr_out['DATE12'] = f"{temporal_dates[0]}_{temporal_dates[-1]}"

    # Update reference date to match temporal dates
    if 'REF_DATE' in atr and atr['REF_DATE'] not in temporal_dates:
        print(f"Warning: Original REF_DATE {atr['REF_DATE']} not in {sampling_mode} dates.")
        atr_out['REF_DATE'] = temporal_dates[0]
        print(f"Updated REF_DATE to {temporal_dates[0]}")
    else:
        atr_out['REF_DATE'] = temporal_dates[0]  # Ensure consistency
    
    # Add model information to metadata
    atr_out['MODEL_POLYNOMIAL'] = str(model['polynomial'])
    atr_out['MODEL_PERIODIC'] = str(model['periodic'])
    atr_out['SAMPLING_MODE'] = sampling_mode
    atr_out['INTERPOLATION'] = f'model_based_{sampling_mode}'
    atr_out['DESCRIPTION'] = f'{sampling_mode.capitalize()} timeseries generated from temporal model fitting'
    
    # Prepare dataset structure
    num_dates = len(temporal_dates)
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    
    ds_name_dict = {
        'date': [f'S{len(temporal_dates[0])}', (num_dates,), np.array(temporal_dates, dtype=f'S{len(temporal_dates[0])}')],
        'timeseries': [np.float32, (num_dates, length, width), None]
    }
    
    # Create HDF5 file structure
    writefile.layout_hdf5(output_file, metadata=atr_out, ds_name_dict=ds_name_dict)


def save_daily_timeseries(daily_ts, daily_dates, atr, output_file, model):
    """
    Save daily timeseries to HDF5 file.
    
    Parameters:
        daily_ts (np.array): Daily timeseries data
        daily_dates (list): Daily date list
        atr (dict): Metadata dictionary
        output_file (str): Output file path
        model (dict): Model configuration for metadata
    """
    print(f"\nSaving daily timeseries to {output_file}...")
    
    # Prepare metadata
    atr_out = dict(atr)
    atr_out['FILE_TYPE'] = 'timeseries'
    atr_out['START_DATE'] = daily_dates[0]
    atr_out['END_DATE'] = daily_dates[-1]
    atr_out['DATE12'] = f"{daily_dates[0]}_{daily_dates[-1]}"
    
    # Add model information to metadata
    atr_out['MODEL_POLYNOMIAL'] = str(model['polynomial'])
    atr_out['MODEL_PERIODIC'] = str(model['periodic'])
    atr_out['INTERPOLATION'] = 'model_based_daily'
    atr_out['DESCRIPTION'] = 'Daily timeseries generated from temporal model fitting'
    
    # Prepare dataset structure
    num_dates = len(daily_dates)
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    
    ds_name_dict = {
        'date': [f'S{len(daily_dates[0])}', (num_dates,), np.array(daily_dates, dtype=f'S{len(daily_dates[0])}')],
        'timeseries': [np.float32, (num_dates, length, width), None]
    }
    
    # Create and write HDF5 file
    writefile.layout_hdf5(output_file, metadata=atr_out, ds_name_dict=ds_name_dict)
    
    # Write timeseries data
    with h5py.File(output_file, 'r+') as f:
        f['timeseries'][:] = daily_ts
    
    print(f"Successfully saved {num_dates} daily observations to {output_file}")


def finalize_output_file(output_file, temporal_dates, sampling_mode='daily'):
    """
    Finalize the output file after block-wise writing is complete.
    
    Parameters:
        output_file (str): Output file path
        temporal_dates (list): Temporal date list
        sampling_mode (str): 'daily' or 'monthly' sampling mode
    """
    print(f"\nFinalizing output file: {output_file}")
    num_dates = len(temporal_dates)
    
    # Verify file integrity
    with h5py.File(output_file, 'r') as f:
        if 'timeseries' in f:
            shape = f['timeseries'].shape
            print(f"Final file shape: {shape}")
            print(f"Successfully saved {num_dates} {sampling_mode} observations to {output_file}")
        else:
            print("Warning: timeseries dataset not found in output file")


def main():
    """
    Main execution function for temporally-interpolated timeseries generation.
    """
    # Configuration parameters
    input_file = 'desc_004B/timeseries_SET_ERA5_tropHgt.h5'
    
    # Temporal sampling configuration
    sampling_mode = 'monthly'  # Options: 'daily', 'monthly'
    polynomial_degree = 1      # Linear trend
    periodic_components = [40, 20, 10, 5, 1]  # Annual and semi-annual
    max_memory_gb = 8.0       # Increased memory limit for better performance
    
    # Generate output filename based on sampling mode
    if sampling_mode == 'daily':
        output_file = 'fitted_daily_timeseries.h5'
    elif sampling_mode == 'monthly':
        output_file = 'fitted_monthly_timeseries.h5'
    else:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
    
    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        print("Please ensure the timeseries.h5 file exists in the current directory.")
        return 1
    
    try:
        start_time = time.time()
        
        # Check if output file already exists
        if os.path.exists(output_file):
            response = input(f"Output file {output_file} already exists. Overwrite? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                print("Operation cancelled by user.")
                return 0
        
        # Step 1: Setup temporal model
        print("Setting up temporal deformation model...")
        model = setup_temporal_model(
            poly_degree=polynomial_degree, 
            periods=periodic_components
        )
        print(f"Model configuration: polynomial={polynomial_degree}, periodic={periodic_components}")
        print(f"Temporal sampling mode: {sampling_mode}")
        
        # Step 2: Fit model to sparse timeseries data
        model_params, metadata, original_dates, design_matrix = fit_timeseries_model(
            ts_file=input_file,
            model=model,
            max_memory=max_memory_gb
        )
        
        # Step 3: Generate temporal date list based on sampling mode
        start_date = original_dates[0]
        end_date = original_dates[-1]
        temporal_dates = create_temporal_date_list(start_date, end_date, sampling_mode)
        print(f"Generated {len(temporal_dates)} {sampling_mode} dates from {start_date} to {end_date}")
        
        # Step 4: Reconstruct temporal timeseries using block-wise processing
        reconstruct_temporal_timeseries_blockwise(
            model_params=model_params,
            model=model,
            temporal_dates=temporal_dates,
            atr=metadata,
            ref_date=metadata.get('REF_DATE', original_dates[0]),
            output_file=output_file,
            sampling_mode=sampling_mode,
            max_memory=max_memory_gb
        )
        
        # Step 5: Finalize output file
        finalize_output_file(output_file, temporal_dates, sampling_mode)
        
        # Summary
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"\nProcessing completed successfully!")
        print(f"Execution time: {int(minutes):02d}m {seconds:04.1f}s")
        print(f"Input: {len(original_dates)} sparse observations")
        print(f"Output: {len(temporal_dates)} {sampling_mode} interpolated values")
        print(f"Temporal upsampling factor: {len(temporal_dates)/len(original_dates):.1f}x")
        
        # Additional information based on sampling mode
        if sampling_mode == 'monthly':
            daily_equivalent = len(create_daily_date_list(start_date, end_date))
            compression_ratio = daily_equivalent / len(temporal_dates)
            print(f"File size reduction vs daily: {compression_ratio:.1f}x smaller")
            print(f"Suitable for: seasonal analysis, long-term trends, efficient storage")
        elif sampling_mode == 'daily':
            print(f"Suitable for: detailed temporal analysis, short-term variations, complete coverage")
        
        return 0
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())