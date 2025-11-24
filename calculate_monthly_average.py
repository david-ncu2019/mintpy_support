#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np
import datetime
from collections import defaultdict

# MintPy utility functions
from mintpy.utils import readfile, writefile
from mintpy.utils import ptime
from mintpy.objects import timeseries

def run_monthly_resampling(ts_file, out_file=None):
    """
    Resample timeseries data to monthly averages following MintPy patterns.
    
    Parameters:
    ts_file : str, input timeseries HDF5 file
    out_file : str, output resampled timeseries HDF5 file
    
    Returns:
    out_file : str, path to output file
    """
    
    # Step 1: Read input data and metadata
    print(f'reading timeseries data from file: {ts_file}')
    ts_obj = timeseries(ts_file)
    ts_data = ts_obj.read()
    meta = ts_obj.metadata.copy()
    
    # Step 2: Get date information
    date_list = ts_obj.get_date_list()
    dates_dt = ptime.date_list2vector(date_list)[0]
    
    # Step 3: Group dates by month
    monthly_bins = defaultdict(list)
    for i, date in enumerate(dates_dt):
        month_key = date.strftime("%Y-%m")
        monthly_bins[month_key].append(i)
    
    # Step 4: Calculate monthly averages
    sorted_months = sorted(monthly_bins.keys())
    num_months = len(sorted_months)
    length, width = ts_data.shape[1], ts_data.shape[2]
    
    print(f'resampling {len(date_list)} acquisitions to {num_months} monthly averages')
    print(f'data dimensions: {num_months} x {length} x {width}')
    
    # Create resampled data array
    resampled_data = np.zeros((num_months, length, width), dtype=np.float32)
    
    for i, month in enumerate(sorted_months):
        indices = monthly_bins[month]
        month_stack = ts_data[indices, :, :]
        resampled_data[i, :, :] = np.nanmean(month_stack, axis=0)
    
    # Step 5: Create new date list (using 15th of each month)
    monthly_dates = [datetime.datetime.strptime(f"{m}-15", "%Y-%m-%d") for m in sorted_months]
    monthly_dates_strings = [d.strftime("%Y%m%d") for d in monthly_dates]
    
    # Step 6: Update metadata
    meta = meta.copy()  # Ensure we have a copy
    meta['START_DATE'] = monthly_dates_strings[0]
    meta['END_DATE'] = monthly_dates_strings[-1]
    meta['REF_DATE'] = monthly_dates_strings[0]  # Set to first monthly date
    meta['num_file'] = str(num_months)
    
    # Remove old date info that might conflict
    if 'date' in meta:
        del meta['date']
    
    # Step 7: Prepare data dictionary for writing (following MintPy pattern)
    # This is the key part - MintPy expects a dictionary with 'timeseries' and 'date' keys
    data_dict = {
        'timeseries': resampled_data,
        'date': np.array(monthly_dates_strings, dtype='S8')  # Convert to bytes for HDF5
    }
    
    # Step 8: Write output file
    print(f'writing monthly resampled timeseries to file: {out_file}')
    writefile.write(data_dict, out_file=out_file, metadata=meta)
    
    return out_file

# Main execution
if __name__ == "__main__":
    # Set input and output files
    ts_file = "asc_004B/timeseries_SET_ERA5_tropHgt_ramp_demErr.h5"
    resampled_ts_file = "monthly_avg_asc_timeseries.h5"
    
    # Run the resampling
    run_monthly_resampling(ts_file, resampled_ts_file)