#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from collections import defaultdict

# MintPy utility functions
from mintpy.utils import readfile, writefile, utils as ut
from mintpy.utils import ptime
from mintpy.objects import timeseries
from mintpy.asc_desc2horz_vert import asc_desc2horz_vert


def get_overlap_lalo(atr_list):
    """Find overlap area in lat/lon of geocoded files based on their metadata."""
    S, N, W, E = None, None, None, None
    for i, atr in enumerate(atr_list):
        Si, Ni, Wi, Ei = ut.four_corners(atr)
        if i == 0:
            S, N, W, E = Si, Ni, Wi, Ei
        else:
            S = max(Si, S)
            N = min(Ni, N)
            W = max(Wi, W)
            E = min(Ei, E)
    return S, N, W, E


def get_all_unique_dates(date_list1, date_list2):
    """Get all unique dates from both date lists (union)."""
    return sorted(list(set(date_list1) | set(date_list2)))


def get_common_dates(date_list1, date_list2):
    """Find common dates between two date lists (intersection)."""
    return sorted(list(set(date_list1) & set(date_list2)))


def direct_los_to_vertical(los_data, inc_angle):
    """Direct conversion from LOS to vertical using: dU = dLOS / cos(incidence_angle)"""
    inc_rad = np.deg2rad(inc_angle)
    vert_data = los_data / np.cos(inc_rad)
    return vert_data


def run_horz_vert_decomposition(asc_file, desc_file, geom_asc_file=None, geom_desc_file=None, 
                               horz_az_angle=-90.0, out_dir=None):
    """
    Decompose ascending and descending LOS timeseries into horizontal and vertical components.
    Following the exact approach from MintPy's asc_desc2horz_vert.py
    """
    
    if out_dir is None:
        out_dir = os.path.dirname(asc_file)
    
    print('='*60)
    print('ASCENDING/DESCENDING TO HORIZONTAL/VERTICAL DECOMPOSITION')
    print('='*60)
    
    # Step 1: Read metadata and check file compatibility - following MintPy pattern
    print('\n1. Reading file metadata and calculating overlap...')
    atr_list = [readfile.read_attribute(fname) for fname in [asc_file, desc_file]]
    
    # Check coordinate systems
    for atr in atr_list:
        if any(x not in atr.keys() for x in ['Y_FIRST', 'X_FIRST']):
            raise ValueError('Input files are not geocoded.')
    
    print(f'Ascending file: {asc_file}')
    print(f'Descending file: {desc_file}')
    
    # Step 2: Calculate overlapping area - exact MintPy approach
    S, N, W, E = get_overlap_lalo(atr_list)
    lat_step = float(atr_list[0]['Y_STEP'])
    lon_step = float(atr_list[0]['X_STEP'])
    length = int(round((S - N) / lat_step))
    width = int(round((E - W) / lon_step))
    
    print(f'Overlapping area in SNWE: {(S, N, W, E)}')
    print(f'Overlap dimensions: {length} x {width}')
    
    # Step 3: Read timeseries dates
    print('\n2. Reading timeseries dates...')
    asc_obj = timeseries(asc_file)
    desc_obj = timeseries(desc_file)
    
    asc_dates = asc_obj.get_date_list()
    desc_dates = desc_obj.get_date_list()
    
    print(f'Ascending: {len(asc_dates)} dates from {asc_dates[0]} to {asc_dates[-1]}')
    print(f'Descending: {len(desc_dates)} dates from {desc_dates[0]} to {desc_dates[-1]}')
    
    # Find ALL unique dates from both datasets
    all_unique_dates = get_all_unique_dates(asc_dates, desc_dates)
    common_dates = get_common_dates(asc_dates, desc_dates)
    
    print(f'All unique dates: {len(all_unique_dates)}')
    print(f'Common dates: {len(common_dates)}')
    print(f'Ascending only: {len(set(asc_dates) - set(desc_dates))}')
    print(f'Descending only: {len(set(desc_dates) - set(asc_dates))}')
    
    if len(common_dates) == 0:
        print('\nNo common dates found. Performing direct LOS-to-vertical conversion only.')
        return direct_conversion_workflow(asc_file, desc_file, out_dir)
    
    # Step 4: Process ALL unique dates with mixed decomposition approach
    print(f'\n3. Processing all {len(all_unique_dates)} unique dates...')
    
    num_dates = len(all_unique_dates)
    
    # Initialize output arrays for all dates
    dhorz_all = np.zeros((num_dates, length, width), dtype=np.float32)
    dvert_all = np.zeros((num_dates, length, width), dtype=np.float32)
    
    # Get geometry information first
    print('\n4. Reading geometry information...')
    
    if geom_asc_file and geom_desc_file:
        # Read 2D geometry (not implemented for this case)
        print('2D geometry files not supported in mixed date processing')
        raise NotImplementedError('2D geometry files not supported for mixed date processing')
    else:
        # Calculate constant angles from metadata
        asc_inc_angle = ut.incidence_angle(atr_list[0], dimension=0, print_msg=False)
        asc_az_angle = ut.heading2azimuth_angle(float(atr_list[0]['HEADING']))
        desc_inc_angle = ut.incidence_angle(atr_list[1], dimension=0, print_msg=False)
        desc_az_angle = ut.heading2azimuth_angle(float(atr_list[1]['HEADING']))
        
        print(f'Ascending - Inc: {asc_inc_angle:.1f}°, Az: {asc_az_angle:.1f}°')
        print(f'Descending - Inc: {desc_inc_angle:.1f}°, Az: {desc_az_angle:.1f}°')
    
    # Prepare spatial cropping boxes
    asc_coord = ut.coordinate(atr_list[0])
    desc_coord = ut.coordinate(atr_list[1])
    asc_y0, asc_x0 = asc_coord.lalo2yx(N, W)
    desc_y0, desc_x0 = desc_coord.lalo2yx(N, W)
    asc_box = (asc_x0, asc_y0, asc_x0 + width, asc_y0 + length)
    desc_box = (desc_x0, desc_y0, desc_x0 + width, desc_y0 + length)
    
    print(f'Ascending box: {asc_box}')
    print(f'Descending box: {desc_box}')
    
    # Read all data once
    print('\nReading all timeseries data...')
    asc_data_full = readfile.read(asc_file, box=asc_box)[0]
    desc_data_full = readfile.read(desc_file, box=desc_box)[0]
    
    # Step 5: Process each date individually
    print(f'\n5. Processing {num_dates} dates with mixed decomposition approach...')
    
    for date_idx, date in enumerate(all_unique_dates):
        print(f'Processing date {date_idx + 1}/{num_dates}: {date}')
        
        # Check availability in each track
        has_asc = date in asc_dates
        has_desc = date in desc_dates
        
        if has_asc and has_desc:
            # Both tracks available - full decomposition
            asc_idx = asc_dates.index(date)
            desc_idx = desc_dates.index(date)
            
            # Extract LOS data for this date
            dlos_date = np.zeros((2, length, width), dtype=np.float32)
            dlos_date[0, :, :] = asc_data_full[asc_idx, :, :]
            dlos_date[1, :, :] = desc_data_full[desc_idx, :, :]
            
            # Setup geometry for decomposition
            los_inc_angle = np.array([asc_inc_angle, desc_inc_angle], dtype=np.float32)
            los_az_angle = np.array([asc_az_angle, desc_az_angle], dtype=np.float32)
            
            # Perform full asc/desc decomposition
            dhorz_date, dvert_date = asc_desc2horz_vert(dlos_date, los_inc_angle, los_az_angle, horz_az_angle)
            
            dhorz_all[date_idx, :, :] = dhorz_date
            dvert_all[date_idx, :, :] = dvert_date
            
            print(f'  -> Full decomposition (asc + desc)')
            
        elif has_asc:
            # Only ascending available - direct conversion
            asc_idx = asc_dates.index(date)
            los_data = asc_data_full[asc_idx, :, :]
            
            # Direct LOS to vertical conversion
            vert_data = direct_los_to_vertical(los_data, asc_inc_angle)
            
            dhorz_all[date_idx, :, :] = 0  # Assume no horizontal displacement
            dvert_all[date_idx, :, :] = vert_data
            
            print(f'  -> Direct conversion (asc only)')
            
        elif has_desc:
            # Only descending available - direct conversion
            desc_idx = desc_dates.index(date)
            los_data = desc_data_full[desc_idx, :, :]
            
            # Direct LOS to vertical conversion
            vert_data = direct_los_to_vertical(los_data, desc_inc_angle)
            
            dhorz_all[date_idx, :, :] = 0  # Assume no horizontal displacement
            dvert_all[date_idx, :, :] = vert_data
            
            print(f'  -> Direct conversion (desc only)')
            
        else:
            # This should not happen
            print(f'  -> ERROR: Date {date} not found in either track!')
            dhorz_all[date_idx, :, :] = np.nan
            dvert_all[date_idx, :, :] = np.nan
    
    print(f'\nProcessing complete:')
    print(f'Horizontal timeseries shape: {dhorz_all.shape}')
    print(f'Vertical timeseries shape: {dvert_all.shape}')
    
    # Step 7: Prepare output metadata and write timeseries files
    print('\n6. Writing output timeseries files...')
    
    # Create output metadata based on overlap area
    out_atr = atr_list[0].copy()
    out_atr['FILE_TYPE'] = 'timeseries'
    out_atr['UNIT'] = 'm'
    out_atr['LENGTH'] = str(length)
    out_atr['WIDTH'] = str(width)
    out_atr['Y_FIRST'] = str(N)
    out_atr['X_FIRST'] = str(W)
    out_atr['Y_STEP'] = str(lat_step)
    out_atr['X_STEP'] = str(lon_step)
    
    # Update date range and reference information for ALL dates
    out_atr['START_DATE'] = all_unique_dates[0]
    out_atr['END_DATE'] = all_unique_dates[-1]
    out_atr['REF_DATE'] = all_unique_dates[0]
    out_atr['num_file'] = str(num_dates)
    
    # Update reference point coordinates
    ref_lat, ref_lon = float(out_atr['REF_LAT']), float(out_atr['REF_LON'])
    coord_out = ut.coordinate(out_atr)
    ref_y, ref_x = coord_out.lalo2yx(ref_lat, ref_lon)
    out_atr['REF_Y'] = str(int(ref_y))
    out_atr['REF_X'] = str(int(ref_x))
    
    # Prepare data dictionaries for timeseries output
    data_dict_horz = {
        'timeseries': dhorz_all,
        'date': np.array(all_unique_dates, dtype='S8')
    }
    data_dict_vert = {
        'timeseries': dvert_all,
        'date': np.array(all_unique_dates, dtype='S8')
    }
    
    # Write output timeseries files
    horz_file = os.path.join(out_dir, 'horizontal_timeseries.h5')
    vert_file = os.path.join(out_dir, 'vertical_timeseries.h5')
    
    writefile.write(data_dict_horz, out_file=horz_file, metadata=out_atr)
    writefile.write(data_dict_vert, out_file=vert_file, metadata=out_atr)
    
    print(f'Horizontal timeseries written to: {horz_file}')
    print(f'  - {num_dates} dates from {all_unique_dates[0]} to {all_unique_dates[-1]}')
    print(f'  - Dimensions: {num_dates} x {length} x {width}')
    print(f'  - Common dates (full decomposition): {len(common_dates)}')
    print(f'  - Single-track dates (direct conversion): {num_dates - len(common_dates)}')
    print(f'Vertical timeseries written to: {vert_file}')
    print(f'  - {num_dates} dates from {all_unique_dates[0]} to {all_unique_dates[-1]}')
    print(f'  - Dimensions: {num_dates} x {length} x {width}')
    print(f'  - Common dates (full decomposition): {len(common_dates)}')
    print(f'  - Single-track dates (direct conversion): {num_dates - len(common_dates)}')
    
    return [horz_file, vert_file]


def direct_conversion_workflow(asc_file, desc_file, out_dir):
    """Direct conversion workflow when no common dates exist."""
    
    print('\n--- DIRECT CONVERSION WORKFLOW ---')
    output_files = []
    
    for i, (los_file, orbit) in enumerate([(asc_file, 'ascending'), (desc_file, 'descending')]):
        print(f'\n{i+1}. Processing {orbit} track...')
        
        # Read timeseries
        ts_obj = timeseries(los_file)
        los_data = ts_obj.read()
        atr = ts_obj.metadata
        dates = ts_obj.get_date_list()
        
        # Get incidence angle
        inc_angle = ut.incidence_angle(atr, dimension=0, print_msg=False)
        print(f'Incidence angle: {inc_angle:.1f}°')
        
        # Convert LOS to vertical
        print('Converting LOS to vertical displacement...')
        vert_data = direct_los_to_vertical(los_data, inc_angle)
        
        # Prepare output metadata
        out_atr = atr.copy()
        out_atr['FILE_TYPE'] = 'timeseries'
        out_atr['UNIT'] = 'm'
        
        # Prepare data dictionary
        data_dict = {
            'timeseries': vert_data,
            'date': np.array(dates, dtype='S8')
        }
        
        # Write output
        out_file = os.path.join(out_dir, f'vertical_{orbit}.h5')
        writefile.write(data_dict, out_file=out_file, metadata=out_atr)
        
        print(f'Vertical component written to: {out_file}')
        output_files.append(out_file)
    
    return output_files


def main(asc_file, desc_file, geom_asc_file=None, geom_desc_file=None, 
         horz_az_angle=-90.0, out_dir=None):
    """Main function to run horizontal/vertical decomposition."""
    
    return run_horz_vert_decomposition(
        asc_file, desc_file, geom_asc_file, geom_desc_file, 
        horz_az_angle, out_dir
    )


if __name__ == "__main__":
    # Example usage
    asc_file = "monthly_fitted/asc_fitted_monthly_msk.h5"
    desc_file = "monthly_fitted/desc_fitted_monthly_msk.h5"
    out_dir = "./monthly_fitted/"  # Current directory
    
    # Run decomposition
    output_files = main(asc_file=asc_file, desc_file=desc_file, out_dir=out_dir)
    
    print('\n' + '='*60)
    print('DECOMPOSITION COMPLETE')
    print('='*60)