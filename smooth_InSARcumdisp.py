import numpy as np
import pandas as pd
import GPy
import xarray as xr
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def convert_time_to_numbers(displacement_data):
    """Convert dates to numbers that GP can understand."""
    start_date = pd.to_datetime(displacement_data.time.values[0])
    
    time_numbers = np.array([
        (pd.to_datetime(t) - start_date).days 
        for t in displacement_data.time.values
    ])[:, None]
    
    max_days = int(time_numbers.max())
    daily_grid = np.arange(0, max_days + 1)[:, None]
    
    return time_numbers, daily_grid

def fit_single_pixel_gp(time_obs, displacement_values, time_pred):
    """Fit GP to one pixel's time series."""
    # Skip if insufficient valid data
    valid_data = ~np.isnan(displacement_values)
    if np.sum(valid_data) < 5:
        return np.full(len(time_pred), np.nan), np.full(len(time_pred), np.nan)
    
    # Prepare data
    clean_times = time_obs[valid_data]
    if clean_times.ndim == 1:
        clean_times = clean_times[:, None]
    
    clean_displacements = displacement_values[valid_data]
    if clean_displacements.ndim == 1:
        clean_displacements = clean_displacements[:, None]
    
    if time_pred.ndim == 1:
        time_pred = time_pred[:, None]
    
    # Fit GP
    kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=30.0)
    model = GPy.models.GPRegression(clean_times, clean_displacements, kernel)
    model.optimize(messages=False, max_iters=50)
    
    pred_mean, pred_var = model.predict(time_pred)
    pred_std = np.sqrt(pred_var)
    
    return pred_mean.flatten(), pred_std.flatten()

def process_spatial_chunk(chunk_data, time_obs, time_pred, chunk_id):
    """Process one spatial chunk."""
    n_times, n_lats, n_lons = chunk_data.shape
    total_pixels = n_lats * n_lons
    
    pred_means = np.zeros((len(time_pred), n_lats, n_lons))
    pred_stds = np.zeros((len(time_pred), n_lats, n_lons))
    
    with tqdm(total=total_pixels, desc=f"Chunk {chunk_id}", leave=False) as pbar:
        for i in range(n_lats):
            for j in range(n_lons):
                pixel_ts = chunk_data[:, i, j]
                mean, std = fit_single_pixel_gp(time_obs, pixel_ts, time_pred)
                pred_means[:, i, j] = mean
                pred_stds[:, i, j] = std
                pbar.update(1)
    
    return pred_means, pred_stds

def create_output_folder(input_filename):
    """Create output folder based on input filename."""
    input_path = Path(input_filename)
    folder_name = f"{input_path.stem}_gp_chunks"
    output_dir = Path(folder_name)
    output_dir.mkdir(exist_ok=True)
    return output_dir

def save_chunk_result(pred_means, pred_stds, lat_coords, lon_coords, time_coords, 
                     output_dir, chunk_id, lat_start, lat_end, lon_start, lon_end):
    """Save individual chunk result to file."""
    chunk_dataset = xr.Dataset({
        'displacement': (['time', 'latitude', 'longitude'], pred_means),
        'uncertainty': (['time', 'latitude', 'longitude'], pred_stds)
    }, coords={
        'time': time_coords,
        'latitude': lat_coords,
        'longitude': lon_coords
    }, attrs={
        'chunk_info': f"lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]",
        'method': 'Independent pixel GP interpolation'
    })
    
    chunk_file = output_dir / f"chunk_{chunk_id:04d}_lat{lat_start}-{lat_end}_lon{lon_start}-{lon_end}.nc"
    chunk_dataset.to_netcdf(chunk_file)
    return chunk_file

def run_incremental_gp_interpolation(displacement_data, input_filename, chunk_size=(50, 50, 50)):
    """Main function: Process chunks incrementally and save to disk."""
    
    print("Step 1: Converting time to numbers...")
    time_obs, time_pred = convert_time_to_numbers(displacement_data)
    
    print("Step 2: Creating output directory...")
    output_dir = create_output_folder(input_filename)
    print(f"Output directory: {output_dir}")
    
    print("Step 3: Calculating chunk boundaries...")
    n_times, n_lats, n_lons = displacement_data.shape
    
    # Calculate chunk boundaries
    lat_chunks = []
    for i in range(0, n_lats, chunk_size[1]):
        lat_chunks.append((i, min(i + chunk_size[1], n_lats)))
    
    lon_chunks = []
    for j in range(0, n_lons, chunk_size[2]):
        lon_chunks.append((j, min(j + chunk_size[2], n_lons)))
    
    total_chunks = len(lat_chunks) * len(lon_chunks)
    print(f"Total chunks to process: {total_chunks}")
    
    print("Step 4: Processing chunks incrementally...")
    
    # Create daily date range for output
    start_date = pd.to_datetime(displacement_data.time.values[0])
    daily_dates = pd.date_range(start=start_date, periods=len(time_pred), freq='D')
    
    chunk_id = 0
    chunk_files = []
    
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar_main:
        for lat_start, lat_end in lat_chunks:
            for lon_start, lon_end in lon_chunks:
                # Extract chunk
                chunk = displacement_data.isel(
                    latitude=slice(lat_start, lat_end),
                    longitude=slice(lon_start, lon_end)
                )
                
                # Check if chunk has any valid data
                if np.all(np.isnan(chunk.values)):
                    pbar_main.set_postfix_str(f"Skipping empty chunk {chunk_id}")
                    pbar_main.update(1)
                    chunk_id += 1
                    continue
                
                # Process chunk
                pred_means, pred_stds = process_spatial_chunk(
                    chunk.values, time_obs, time_pred, chunk_id
                )
                
                # Save chunk result
                chunk_file = save_chunk_result(
                    pred_means, pred_stds,
                    chunk.latitude.values, chunk.longitude.values, daily_dates,
                    output_dir, chunk_id, lat_start, lat_end, lon_start, lon_end
                )
                chunk_files.append(chunk_file)
                
                pbar_main.set_postfix_str(f"Saved chunk {chunk_id}")
                pbar_main.update(1)
                chunk_id += 1
    
    # Save chunk index file
    chunk_index = {
        'total_chunks': len(chunk_files),
        'chunk_files': [str(f) for f in chunk_files],
        'original_shape': displacement_data.shape,
        'chunk_size': chunk_size,
        'processing_complete': True
    }
    
    index_file = output_dir / "chunk_index.txt"
    with open(index_file, 'w') as f:
        for key, value in chunk_index.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Step 5: Processing complete!")
    print(f"Processed {len(chunk_files)} chunks")
    print(f"Results saved in: {output_dir}")
    print(f"Chunk index: {index_file}")
    
    return output_dir, chunk_files

def merge_chunks_if_needed(output_dir, chunk_files, max_size_gb=20):
    """Optional: Merge chunks back into single file if small enough."""
    
    # Estimate merged file size
    sample_chunk = xr.open_dataset(chunk_files[0])
    chunk_size_mb = sample_chunk.nbytes / (1024**2)
    estimated_size_gb = (chunk_size_mb * len(chunk_files)) / 1024
    sample_chunk.close()
    
    if estimated_size_gb > max_size_gb:
        print(f"Merged file would be ~{estimated_size_gb:.1f}GB (>{max_size_gb}GB limit)")
        print("Keeping chunks separate for memory efficiency")
        return None
    
    print(f"Merging {len(chunk_files)} chunks into single file...")
    merged_file = output_dir / "merged_gp_interpolated.nc"
    
    # Open all chunks and concatenate
    chunks = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        chunks.append(xr.open_dataset(chunk_file))
    
    # Merge chunks (this requires implementing spatial reconstruction)
    print("Note: Spatial merging requires additional implementation")
    print("Chunks saved individually for now")
    
    # Close chunks
    for chunk in chunks:
        chunk.close()
    
    return None

# Main workflow
if __name__ == "__main__":
    print("Loading displacement data...")
    input_file = "cropped_desc_S-E-trop-ramp-dem_msk_epsg3826.nc"
    
    # Load data
    input_data = xr.open_dataset(input_file)
    displacement_data = input_data["displacement"]
    
    print(f"Data shape: {displacement_data.shape}")
    print(f"Memory requirement for full result: {np.prod(displacement_data.shape) * 8 / (1024**3):.1f} GB")
    
    # Run incremental processing
    output_dir, chunk_files = run_incremental_gp_interpolation(
        displacement_data, 
        input_file,
        chunk_size=(50, 50, 50)  # Adjust based on your system
    )
    
    # Optional: attempt merge if result would be manageable
    merge_chunks_if_needed(output_dir, chunk_files, max_size_gb=20)
    
    print("Processing complete!")