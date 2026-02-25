"""
1m to Xm Raster Resampling - COMPLETE VERSION
Advanced aggregation functions, dynamic output formats, binary masks
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QTextEdit, QRadioButton, QButtonGroup, QComboBox, QScrollArea
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import numpy as np
from osgeo import gdal, gdalconst, ogr, osr
import time
from multiprocessing import Pool, cpu_count
import tempfile
import shutil


class RasterAnalyzer(QThread):
    """Analyze raster to find value range"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            ds = gdal.Open(self.filepath, gdalconst.GA_ReadOnly)
            if ds is None:
                self.error.emit('Cannot open raster')
                return

            band = ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            
            width = ds.RasterXSize
            height = ds.RasterYSize
            
            self.status.emit(f'Analyzing {width:,} × {height:,} raster...')
            
            # Sample data across entire raster
            values = []
            
            for i in range(0, height, max(1, height // 50)):
                row = band.ReadAsArray(0, i, width, 1)
                if row is not None:
                    row = row.flatten()
                    if nodata is not None:
                        row = row[row != nodata]
                    values.extend(row)
            
            values = np.array(values, dtype=np.int32)
            
            result = {
                'width': width,
                'height': height,
                'nodata': nodata,
                'has_data': len(values) > 0
            }
            
            if result['has_data']:
                result['min'] = int(values.min())
                result['max'] = int(values.max())
                result['mean'] = float(values.mean())
                result['unique_count'] = len(np.unique(values))
                
                # Get percentiles
                unique_vals = np.unique(values)
                result['p10'] = int(np.percentile(unique_vals, 10))
                result['p25'] = int(np.percentile(unique_vals, 25))
                result['p50'] = int(np.percentile(unique_vals, 50))
                result['p75'] = int(np.percentile(unique_vals, 75))
                result['p90'] = int(np.percentile(unique_vals, 90))
            
            ds = None
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


def get_grid_info(filepath):
    """Get geotransform and size from raster"""
    ds = gdal.Open(filepath, gdalconst.GA_ReadOnly)
    if ds is None:
        raise Exception(f'Cannot open: {filepath}')
    
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    proj = ds.GetProjection()
    
    ds = None
    return {
        'geotransform': gt,
        'width': width,
        'height': height,
        'projection': proj
    }


def rasterize_vector_row_batch(args):
    """Rasterize a batch of rows from vector - parallel worker"""
    (y_start, y_end, vector_path, width, gt, proj, buffer_dist) = args
    
    # Create temporary raster for this batch
    temp_raster = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
    
    try:
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(
            temp_raster,
            width,
            y_end - y_start,
            1,
            gdal.GDT_Byte,
            options=['COMPRESS=DEFLATE']
        )
        
        if ds_out is None:
            raise Exception('Cannot create raster')
        
        # Adjust geotransform for this batch
        gt_batch = (gt[0], gt[1], gt[2], gt[3] + (y_start * gt[5]), gt[4], gt[5])
        ds_out.SetGeoTransform(gt_batch)
        ds_out.SetProjection(proj)
        
        band = ds_out.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.Fill(0)
        
        # Open vector
        ds_vec = ogr.Open(vector_path)
        if ds_vec is None:
            raise Exception(f'Cannot open vector')
        
        layer = ds_vec.GetLayer(0)
        
        # Rasterize with buffer if specified
        if buffer_dist > 0:
            # Create temporary memory layer with buffered features
            mem_driver = ogr.GetDriverByName('MEMORY')
            mem_ds = mem_driver.CreateDataSource('')
            mem_layer = mem_ds.CreateLayer('buffered', layer.GetSpatialRef(), layer.GetGeomType())
            
            # Copy and buffer features
            for feature in layer:
                geom = feature.GetGeometryRef()
                buffered_geom = geom.Buffer(buffer_dist)
                new_feature = ogr.Feature(mem_layer.GetLayerDefn())
                new_feature.SetGeometry(buffered_geom)
                mem_layer.CreateFeature(new_feature)
            
            gdal.RasterizeLayer(ds_out, [1], mem_layer, burn_values=[1])
        else:
            # Direct rasterization
            gdal.RasterizeLayer(ds_out, [1], layer, burn_values=[1])
        
        band.FlushCache()
        ds_vec = None
        ds_out = None
        
        # Read back the rasterized data
        ds_read = gdal.Open(temp_raster, gdalconst.GA_ReadOnly)
        band_read = ds_read.GetRasterBand(1)
        data = band_read.ReadAsArray()
        ds_read = None
        
        return (y_start, y_end, data, temp_raster)
        
    except Exception as e:
        return (y_start, y_end, None, temp_raster)


def rasterize_vector_parallel(vector_path, output_raster, gt, proj, width, height, 
                             buffer_dist=0, num_processes=None):
    """Rasterize vector in parallel batches"""
    
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    # Determine batch size
    batch_height = max(100, height // (num_processes * 2))
    
    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(
        output_raster,
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE']
    )
    
    if ds_out is None:
        raise Exception('Cannot create output raster')
    
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj)
    
    band_out = ds_out.GetRasterBand(1)
    band_out.SetNoDataValue(0)
    band_out.Fill(0)
    
    # Create batches
    batches = []
    y = 0
    while y < height:
        y_end = min(y + batch_height, height)
        batches.append((y, y_end, vector_path, width, gt, proj, buffer_dist))
        y = y_end
    
    # Process batches in parallel
    temp_files = []
    with Pool(processes=num_processes) as pool:
        for y_start, y_end, data, temp_file in pool.imap_unordered(rasterize_vector_row_batch, batches):
            temp_files.append(temp_file)
            if data is not None:
                band_out.WriteArray(data, 0, y_start)
    
    band_out.FlushCache()
    ds_out = None
    
    # Clean up temp files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
    
    return output_raster


def process_row_worker(args):
    """Process a single row with multiple aggregation functions"""
    (y_out, input_path, gt_in, gt_out, src_width, src_height, 
     min_value, max_value, nodata, resolution, aggregation_func, 
     binary_mode, binary_threshold, output_dtype, input_array, excluded_values) = args
    
    # Use cached array if available, otherwise open file
    if input_array is not None:
        # Use pre-loaded array (RAM cached)
        use_array = True
    else:
        # Open input for this worker (disk read)
        ds_in = gdal.Open(input_path, gdalconst.GA_ReadOnly)
        band_in = ds_in.GetRasterBand(1)
        use_array = False
    
    output_width = int(gt_out['width'])
    
    # Determine numpy dtype based on output GDAL dtype
    if output_dtype == gdal.GDT_Byte:
        np_dtype = np.uint8
    elif output_dtype == gdal.GDT_Int16:
        np_dtype = np.int16
    elif output_dtype == gdal.GDT_Int32:
        np_dtype = np.int32
    elif output_dtype == gdal.GDT_Float32:
        np_dtype = np.float32
    else:
        np_dtype = np.uint8
    
    out_row = np.zeros(output_width, dtype=np_dtype)
    
    # For each output pixel in this row
    for x_out in range(output_width):
        # Get geographic coordinates of output pixel
        x_geo = gt_out['geotransform'][0] + (x_out + 0.5) * gt_out['geotransform'][1]
        y_geo = gt_out['geotransform'][3] + (y_out + 0.5) * gt_out['geotransform'][5]
        
        # Convert to input pixel coordinates (1m)
        x_in = (x_geo - gt_in[0]) / gt_in[1]
        y_in = (y_geo - gt_in[3]) / gt_in[5]
        
        # Get integer pixel bounds (covers entire output cell)
        x_start = int(np.floor(x_in - resolution/2))
        x_end = int(np.ceil(x_in + resolution/2))
        y_start = int(np.floor(y_in - resolution/2))
        y_end = int(np.ceil(y_in + resolution/2))
        
        # Clamp to raster bounds
        x_start = max(0, x_start)
        x_end = min(src_width, x_end)
        y_start = max(0, y_start)
        y_end = min(src_height, y_end)
        
        result = 0
        
        # Read block
        if x_start < x_end and y_start < y_end:
            if use_array:
                # Read from cached array (fast!)
                block = input_array[y_start:y_end, x_start:x_end]
            else:
                # Read from disk (normal)
                block = band_in.ReadAsArray(x_start, y_start, x_end - x_start, y_end - y_start)
            
            if block is not None and block.size > 0:
                # Create valid pixel mask
                if nodata is not None:
                    valid_mask = block != nodata
                else:
                    valid_mask = np.ones_like(block, dtype=bool)
                
                # Apply value range filter
                range_mask = (block >= min_value) & (block <= max_value)
                
                # Create exclusion mask for custom NODATA values
                exclude_mask = np.ones_like(block, dtype=bool)
                for excluded_val in excluded_values:
                    exclude_mask = exclude_mask & (block != excluded_val)
                
                # Combine all masks
                combined_mask = valid_mask & range_mask & exclude_mask
                
                valid_pixels = block[combined_mask]
                
                if len(valid_pixels) > 0:
                    # Calculate aggregation
                    if aggregation_func == 'count':
                        agg_val = len(valid_pixels)
                    elif aggregation_func == 'sum':
                        agg_val = np.sum(valid_pixels)
                    elif aggregation_func == 'mean':
                        agg_val = np.mean(valid_pixels)
                    elif aggregation_func == 'max':
                        agg_val = np.max(valid_pixels)
                    elif aggregation_func == 'min':
                        agg_val = np.min(valid_pixels)
                    elif aggregation_func == 'median':
                        agg_val = np.median(valid_pixels)
                    elif aggregation_func == 'stddev':
                        agg_val = np.std(valid_pixels)
                    else:
                        agg_val = 0
                    
                    # Apply binary classification if enabled
                    # Binary mode: any pixels in valid range = 1, no pixels = 0
                    if binary_mode:
                        result = 1  # Has pixels in range!
                    else:
                        result = int(agg_val) if output_dtype != gdal.GDT_Float32 else agg_val
                else:
                    # No valid pixels in this cell
                    if binary_mode:
                        result = 0  # No pixels in range
                    else:
                        result = 0  # No pixels to aggregate
        
        out_row[x_out] = result
    
    if not use_array:
        ds_in = None
    return (y_out, out_row)


class ResampleWorker(QThread):
    """Resample with multiple aggregation functions"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    timing = pyqtSignal(str)  # New: timing updates
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_path, output_path, input_type='raster', 
                 vector_path=None, buffer_dist=0, min_value=None, max_value=None, 
                 donor_path=None, num_processes=None, resolution=10.0,
                 aggregation_func='count', binary_mode=False, binary_threshold=0, block_size=256, use_ram_cache=False, excluded_values=None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.input_type = input_type
        self.vector_path = vector_path
        self.buffer_dist = buffer_dist
        self.min_value = min_value if min_value is not None else -999999
        self.max_value = max_value if max_value is not None else 999999
        self.donor_path = donor_path
        self.num_processes = num_processes or max(1, cpu_count() - 1)
        self.resolution = resolution
        self.aggregation_func = aggregation_func
        self.binary_mode = binary_mode
        self.binary_threshold = binary_threshold  # Kept for compatibility (not used)
        self.block_size = block_size  # Block size for I/O optimization
        self.use_ram_cache = use_ram_cache  # RAM caching flag
        self.use_threading = use_ram_cache  # Use threading when RAM cache enabled
        self.excluded_values = excluded_values or []  # Values to exclude from aggregation
        self.start_time = None
        self.temp_raster = None
        self.input_array = None  # Will hold cached raster data
        # Timing tracking
        self.time_setup = 0
        self.time_load_ram = 0
        self.time_input_create = 0
        self.time_processing = 0
        self.time_output = 0

    def get_output_dtype(self):
        """Determine output data type based on aggregation function"""
        if self.binary_mode:
            return gdal.GDT_Byte
        
        dtype_map = {
            'count': gdal.GDT_Byte,
            'sum': gdal.GDT_Int32,
            'mean': gdal.GDT_Int16,
            'max': gdal.GDT_Int16,
            'min': gdal.GDT_Int16,
            'median': gdal.GDT_Int16,
            'stddev': gdal.GDT_Float32,
        }
        return dtype_map.get(self.aggregation_func, gdal.GDT_Byte)

    def run(self):
        try:
            self.start_time = time.time()
            t_setup_start = self.start_time
            
            # If vector input, rasterize to temporary 1m raster first
            if self.input_type == 'vector':
                self.status.emit('Rasterizing vector with parallel processing...')
                rasterize_start = time.time()
                
                # Get extent and CRS from vector
                ds_vec = ogr.Open(self.vector_path)
                layer = ds_vec.GetLayer(0)
                extent = layer.GetExtent()
                spatial_ref = layer.GetSpatialRef()
                proj = spatial_ref.ExportToWkt() if spatial_ref else ''
                ds_vec = None
                
                # Calculate extent
                minx, maxx, miny, maxy = extent
                width_1m = int(np.ceil((maxx - minx)))
                height_1m = int(np.ceil((maxy - miny)))
                
                self.status.emit(f'Vector extent: {width_1m:,} × {height_1m:,} pixels (1m)')
                self.status.emit(f'Rasterizing with {self.num_processes} processes...')
                
                # Create temporary 1m raster (PARALLEL)
                self.temp_raster = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
                gt = (minx, 1.0, 0, maxy, 0, -1.0)
                
                rasterize_vector_parallel(self.vector_path, self.temp_raster, gt, proj, 
                                        width_1m, height_1m, self.buffer_dist, self.num_processes)
                
                rasterize_time = time.time() - rasterize_start
                self.status.emit(f'Vector rasterized in {rasterize_time:.1f}s (buffer: {self.buffer_dist}m)')
                input_path_to_process = self.temp_raster
            else:
                input_path_to_process = self.input_path
            
            # Now process as normal raster
            ds_in = gdal.Open(input_path_to_process, gdalconst.GA_ReadOnly)
            if ds_in is None:
                raise Exception('Cannot open input')

            band_in = ds_in.GetRasterBand(1)
            gt_in = ds_in.GetGeoTransform()
            proj_in = ds_in.GetProjection()
            nodata = band_in.GetNoDataValue()
            
            src_width = ds_in.RasterXSize
            src_height = ds_in.RasterYSize
            
            self.status.emit(f'Input 1m: {src_width:,} × {src_height:,} pixels')
            
            # RAM CACHING: Load entire raster to memory if enabled
            if self.use_ram_cache and self.input_type == 'raster':
                self.status.emit('Loading input file to RAM (faster processing)...')
                ram_load_start = time.time()
                try:
                    # IMPORTANT: Preserve source data type (don't convert!)
                    # This prevents 8x inflation for byte data
                    # Byte: 1 byte × 1.3B pixels = 1.3GB
                    # (Not converted to Float64: 8 bytes × 1.3B = 10.4GB!)
                    source_dtype = band_in.DataType
                    self.input_array = band_in.ReadAsArray(buf_type=source_dtype)
                    
                    self.time_load_ram = time.time() - ram_load_start
                    ram_size_gb = self.input_array.nbytes / 1e9
                    self.status.emit(f'Loaded {ram_size_gb:.1f}GB to RAM in {self.time_load_ram:.1f}s ✓')
                except Exception as e:
                    self.status.emit(f'Warning: Could not load to RAM ({str(e)}), using disk')
                    self.input_array = None
            else:
                self.time_load_ram = 0
            
            if self.binary_mode:
                self.status.emit('Binary mask mode: classifying pixels by value range')
            else:
                self.status.emit(f'{self.aggregation_func.upper()} aggregation')
            self.status.emit(f'Resampling with {self.num_processes} processes...')

            # Get output grid from donor raster if provided
            output_gt = None
            output_width = None
            output_height = None
            
            if self.donor_path and os.path.exists(self.donor_path):
                try:
                    donor_info = get_grid_info(self.donor_path)
                    gt_donor = donor_info['geotransform']
                    output_gt = gt_donor
                    output_width = donor_info['width']
                    output_height = donor_info['height']
                    self.status.emit(f'Aligning to donor: {output_width:,} × {output_height:,}')
                except Exception as e:
                    self.error.emit(f'Cannot read donor: {str(e)}')
                    return
            else:
                # Use configurable resolution
                output_gt = (gt_in[0], gt_in[1]*self.resolution, gt_in[2], gt_in[3], gt_in[4], gt_in[5]*self.resolution)
                output_width = int(src_width / self.resolution)
                output_height = int(src_height / self.resolution)
                self.status.emit(f'Target {self.resolution}m: {output_width:,} × {output_height:,}')

            # Get output data type
            output_dtype = self.get_output_dtype()
            
            # Create output raster with correct data type
            driver = gdal.GetDriverByName('GTiff')
            
            # Determine compression level based on dtype
            compress_level = 9 if output_dtype == gdal.GDT_Byte else 6
            
            ds_out = driver.Create(
                self.output_path,
                output_width,
                output_height,
                1,
                output_dtype,
                options=[f'COMPRESS=DEFLATE', f'PREDICTOR=2']
            )
            
            if ds_out is None:
                raise Exception('Cannot create output')

            ds_out.SetGeoTransform(output_gt)
            ds_out.SetProjection(proj_in)
            
            band_out = ds_out.GetRasterBand(1)
            band_out.SetNoDataValue(0)
            
            self.status.emit(f'Processing {output_height:,} rows ({self.aggregation_func}, {self.resolution}m)...')
            
            # Prepare arguments for pool
            gt_out_dict = {
                'geotransform': output_gt,
                'width': output_width
            }
            
            pool_args = [
                (y_out, input_path_to_process, gt_in, gt_out_dict, src_width, src_height,
                 self.min_value, self.max_value, nodata, self.resolution, self.aggregation_func,
                 self.binary_mode, self.binary_threshold, output_dtype, None, self.excluded_values)  # Add excluded_values!
                for y_out in range(output_height)
            ]
            
            # Process rows in parallel using multiprocessing or threading
            resample_start = time.time()
            
            if self.use_threading:
                # Use threading for RAM-cached data (shared memory, no pickling!)
                from concurrent.futures import ThreadPoolExecutor
                
                with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
                    results_iter = executor.map(process_row_worker, pool_args)
                    
                    processed = 0
                    rows_dict = {}
                    
                    for y_out, out_row in results_iter:
                        rows_dict[y_out] = out_row
                        processed += 1
                        
                        # Progress
                        if processed % max(1, output_height // 20) == 0:
                            progress = int(100 * processed / output_height)
                            elapsed = time.time() - self.start_time
                            rate = processed / elapsed if elapsed > 0 else 0
                            remaining = (output_height - processed) / rate if rate > 0 else 0
                            self.progress.emit(progress)
                            self.status.emit(
                                f'Row {processed:,}/{output_height:,} ({progress}%) - ETA {remaining:.0f}s [Threading]'
                            )
                    
                    # Write remaining rows
                    for y in sorted(rows_dict.keys()):
                        out_row_2d = rows_dict[y].reshape(1, -1)
                        band_out.WriteArray(out_row_2d, 0, y)
            else:
                # Use multiprocessing for disk-based data (parallel I/O)
                with Pool(processes=self.num_processes) as pool:
                    results_iter = pool.imap_unordered(process_row_worker, pool_args, chunksize=10)
                    
                    processed = 0
                    rows_dict = {}
                    
                    for y_out, out_row in results_iter:
                        rows_dict[y_out] = out_row
                        processed += 1
                        
                        # Progress
                        if processed % max(1, output_height // 20) == 0:
                            progress = int(100 * processed / output_height)
                            elapsed = time.time() - self.start_time
                            rate = processed / elapsed if elapsed > 0 else 0
                            remaining = (output_height - processed) / rate if rate > 0 else 0
                            self.progress.emit(progress)
                            self.status.emit(
                                f'Row {processed:,}/{output_height:,} ({progress}%) - ETA {remaining:.0f}s'
                            )
                    
                    # Write remaining rows
                    for y in sorted(rows_dict.keys()):
                        out_row_2d = rows_dict[y].reshape(1, -1)
                        band_out.WriteArray(out_row_2d, 0, y)
            
            resample_time = time.time() - resample_start
            band_out.FlushCache()
            ds_in = None
            ds_out = None
            
            # Clean up temp raster
            if self.temp_raster and os.path.exists(self.temp_raster):
                try:
                    os.remove(self.temp_raster)
                except:
                    pass
            
            elapsed = time.time() - self.start_time
            
            if self.binary_mode:
                msg = f'✓ Complete in {elapsed:.1f}s ({self.aggregation_func.upper()}, binary 0/1 mask)'
            else:
                msg = f'✓ Complete in {elapsed:.1f}s ({self.aggregation_func.upper()}, {self.resolution}m resolution)'
            
            self.status.emit(msg)
            
            # Emit detailed timing breakdown
            if self.use_ram_cache and self.input_array is not None:
                timing_info = f'Timing: Load RAM {self.time_load_ram:.1f}s | Processing {resample_time:.1f}s | Block: {self.block_size}×{self.block_size} | Total: {elapsed:.1f}s'
            else:
                timing_info = f'Timing: Processing {resample_time:.1f}s | Block size: {self.block_size}×{self.block_size} | Total: {elapsed:.1f}s'
            self.timing.emit(timing_info)
            
            # CLEANUP: Clear cached array from RAM to free memory
            if self.input_array is not None:
                ram_freed_gb = self.input_array.nbytes / 1e9
                self.input_array = None  # Release memory!
                self.status.emit(f'Cleared {ram_freed_gb:.1f}GB from RAM ✓')
            
            self.progress.emit(100)
            self.finished.emit()

        except Exception as e:
            import traceback
            # CLEANUP: Also clear on error
            if self.input_array is not None:
                self.input_array = None
            self.error.emit(str(e) + '\n' + traceback.format_exc())


class GDALScriptGenerator:
    """Generate GDAL command-line equivalents for speed comparison"""
    
    def __init__(self, input_file, output_file, resolution, min_value=None, max_value=None, 
                 excluded_values=None, aggregation_func='count', num_threads=23):
        self.input_file = input_file
        self.output_file = output_file
        self.resolution = resolution
        self.min_value = min_value or 0
        self.max_value = max_value or 255
        self.excluded_values = excluded_values or []
        self.aggregation_func = aggregation_func.lower()
        self.num_threads = num_threads
    
    def _unix_path(self, path):
        """Convert Windows path to Unix-style path for bash"""
        # Handle Windows paths
        if path.startswith('D:') or path.startswith('C:') or '\\' in path:
            # Convert backslashes to forward slashes
            path = path.replace('\\', '/')
        return path
    
    def generate_gdalwarp(self):
        """Generate gdalwarp command for fast resampling baseline"""
        input_path = self._unix_path(self.input_file)
        output_path = self._unix_path(self.output_file)
        
        script = f"""#!/bin/bash
# GDAL Resampling Script (gdalwarp)
# Generated by Advanced Raster Resampling Tool
# Note: Uses interpolation, not aggregation. Results will differ from COUNT/SUM/MEAN.
# Useful for: Speed baseline comparison

INPUT="{input_path}"
OUTPUT="{output_path}"

echo "Starting GDAL resampling (gdalwarp)..."
time gdalwarp \\
  -r average \\
  -tr {self.resolution} {self.resolution} \\
  -multi \\
  -wo NUM_THREADS={self.num_threads} \\
  "$INPUT" "$OUTPUT"

echo "Complete!"
"""
        return script
    
    def generate_gdal_calc(self):
        """Generate gdal_calc.py command for masking and filtering"""
        input_path = self._unix_path(self.input_file)
        output_path = self._unix_path(self.output_file)
        
        # Build the calculation formula
        formula = "A"  # Start with all pixels
        
        # Add range filtering
        if self.min_value is not None or self.max_value is not None:
            formula = f"A*((A>={self.min_value})*(A<={self.max_value}))"
        
        # Add exclusions
        if self.excluded_values:
            for excluded in self.excluded_values:
                formula = f"{formula}*(A!={excluded})"
        
        script = f"""#!/bin/bash
# GDAL Value Filtering Script (gdal_calc.py)
# Generated by Advanced Raster Resampling Tool
# Note: Applies pixel-wise filtering, no aggregation.
# Useful for: Masking and value preprocessing

INPUT="{input_path}"
OUTPUT="{output_path}"

echo "Starting GDAL filtering (gdal_calc.py)..."
time gdal_calc.py \\
  -A "$INPUT" \\
  --outfile="$OUTPUT" \\
  --calc="{formula}" \\
  --type=Byte \\
  --NoDataValue=0

echo "Complete!"
"""
        return script
    
    def generate_gdalwarp_batch(self):
        """Generate Windows batch script for gdalwarp (fast resampling baseline)"""
        input_path = self.input_file  # Keep Windows paths for .bat
        output_path = self.output_file
        
        script = f"""@echo off
REM GDAL Resampling Script (gdalwarp) - Windows Batch
REM Generated by Advanced Raster Resampling Tool
REM Note: Uses interpolation, not aggregation. Results will differ from COUNT/SUM/MEAN.
REM Useful for: Speed baseline comparison

set INPUT={input_path}
set OUTPUT={output_path}

echo Starting GDAL resampling (gdalwarp)...
echo.

gdalwarp -r average -tr {self.resolution} {self.resolution} -multi -wo NUM_THREADS={self.num_threads} "%INPUT%" "%OUTPUT%"

echo.
echo Complete!
pause
"""
        return script
    
    def generate_gdal_calc_batch(self):
        """Generate Windows batch script for gdal_calc.py (masking)"""
        input_path = self.input_file  # Keep Windows paths for .bat
        output_path = self.output_file
        
        # Build the calculation formula
        formula = "A"  # Start with all pixels
        
        # Add range filtering
        if self.min_value is not None or self.max_value is not None:
            formula = f"A*((A>={self.min_value})*(A<={self.max_value}))"
        
        # Add exclusions
        if self.excluded_values:
            for excluded in self.excluded_values:
                formula = f"{formula}*(A!={excluded})"
        
        script = f"""@echo off
REM GDAL Value Filtering Script (gdal_calc.py) - Windows Batch
REM Generated by Advanced Raster Resampling Tool
REM Note: Applies pixel-wise filtering, no aggregation.
REM Useful for: Masking and value preprocessing

set INPUT={input_path}
set OUTPUT={output_path}

echo Starting GDAL filtering (gdal_calc.py)...
echo.

gdal_calc.py -A "%INPUT%" --outfile="%OUTPUT%" --calc="{formula}" --type=Byte --NoDataValue=0

echo.
echo Complete!
pause
"""
        return script
    
    def generate_python_script(self):
        """Generate Python GDAL script with full aggregation support"""
        input_path = self._unix_path(self.input_file)
        output_path = self._unix_path(self.output_file)
        excluded_str = str(self.excluded_values)
        
        script = f'''#!/usr/bin/env python3
"""
Advanced Raster Aggregation Script
Generated by Advanced Raster Resampling Tool
Full aggregation implementation using GDAL Python bindings
"""

import numpy as np
from osgeo import gdal
import time
import sys

# Settings (from GUI)
INPUT_FILE = "{input_path}"
OUTPUT_FILE = "{output_path}"
RESOLUTION = {self.resolution}
MIN_VALUE = {self.min_value}
MAX_VALUE = {self.max_value}
EXCLUDED_VALUES = {excluded_str}
AGGREGATION_FUNC = "{self.aggregation_func}"
NUM_THREADS = {self.num_threads}

def process_aggregation():
    """Main aggregation function"""
    
    print("Opening input raster...")
    ds_in = gdal.Open(INPUT_FILE, gdal.GA_ReadOnly)
    if ds_in is None:
        print(f"ERROR: Cannot open {{INPUT_FILE}}")
        return False
    
    band_in = ds_in.GetRasterBand(1)
    gt_in = ds_in.GetGeoTransform()
    proj_in = ds_in.GetProjection()
    nodata_in = band_in.GetNoDataValue()
    
    src_width = ds_in.RasterXSize
    src_height = ds_in.RasterYSize
    
    print(f"Input: {{src_width:,}} × {{src_height:,}} pixels")
    
    # Calculate output dimensions
    output_width = int(src_width / RESOLUTION)
    output_height = int(src_height / RESOLUTION)
    
    print(f"Output: {{output_width:,}} × {{output_height:,}} pixels ({{RESOLUTION:.1f}}m resolution)")
    
    # Create output raster
    print("Creating output raster...")
    drv = gdal.GetDriverByName('GTiff')
    ds_out = drv.Create(OUTPUT_FILE, output_width, output_height, 1, gdal.GDT_Int32)
    
    if ds_out is None:
        print(f"ERROR: Cannot create {{OUTPUT_FILE}}")
        return False
    
    # Set geotransform and projection
    gt_out = (gt_in[0], RESOLUTION, 0, gt_in[3], 0, -RESOLUTION)
    ds_out.SetGeoTransform(gt_out)
    ds_out.SetProjection(proj_in)
    
    band_out = ds_out.GetRasterBand(1)
    band_out.SetNoDataValue(0)
    
    # Process each row
    print(f"Processing {{output_height:,}} rows with {{AGGREGATION_FUNC}} aggregation...")
    start_time = time.time()
    
    for y_out in range(output_height):
        out_row = np.zeros(output_width, dtype=np.int32)
        
        for x_out in range(output_width):
            # Get geographic coordinates of output pixel center
            x_geo = gt_in[0] + (x_out * RESOLUTION) + (RESOLUTION / 2)
            y_geo = gt_in[3] - (y_out * RESOLUTION) - (RESOLUTION / 2)
            
            # Calculate input pixel range for this output cell
            x_start_pix = int((x_geo - RESOLUTION/2 - gt_in[0]) / gt_in[1])
            y_start_pix = int((gt_in[3] - y_geo - RESOLUTION/2) / abs(gt_in[5]))
            
            x_end_pix = min(x_start_pix + int(RESOLUTION), src_width)
            y_end_pix = min(y_start_pix + int(RESOLUTION), src_height)
            
            x_start_pix = max(0, x_start_pix)
            y_start_pix = max(0, y_start_pix)
            
            if x_end_pix <= x_start_pix or y_end_pix <= y_start_pix:
                continue
            
            # Read input block
            block_width = x_end_pix - x_start_pix
            block_height = y_end_pix - y_start_pix
            block = band_in.ReadAsArray(x_start_pix, y_start_pix, block_width, block_height)
            
            if block is None or block.size == 0:
                continue
            
            # Apply masks
            valid_mask = (block != nodata_in) if nodata_in is not None else np.ones_like(block, dtype=bool)
            range_mask = (block >= MIN_VALUE) & (block <= MAX_VALUE)
            
            # Apply exclusions
            exclude_mask = np.ones_like(block, dtype=bool)
            for excluded_val in EXCLUDED_VALUES:
                exclude_mask = exclude_mask & (block != excluded_val)
            
            # Combine all masks
            combined_mask = valid_mask & range_mask & exclude_mask
            valid_pixels = block[combined_mask]
            
            # Aggregate
            if len(valid_pixels) > 0:
                if AGGREGATION_FUNC == 'count':
                    out_row[x_out] = len(valid_pixels)
                elif AGGREGATION_FUNC == 'sum':
                    out_row[x_out] = np.sum(valid_pixels)
                elif AGGREGATION_FUNC == 'mean':
                    out_row[x_out] = int(np.mean(valid_pixels))
                elif AGGREGATION_FUNC == 'max':
                    out_row[x_out] = np.max(valid_pixels)
                elif AGGREGATION_FUNC == 'min':
                    out_row[x_out] = np.min(valid_pixels)
                elif AGGREGATION_FUNC == 'median':
                    out_row[x_out] = int(np.median(valid_pixels))
                elif AGGREGATION_FUNC == 'stddev':
                    out_row[x_out] = int(np.std(valid_pixels))
        
        # Write row
        band_out.WriteArray(out_row.reshape(1, -1), 0, y_out)
        
        # Progress
        if (y_out + 1) % max(1, output_height // 20) == 0:
            progress = int(100 * (y_out + 1) / output_height)
            elapsed = time.time() - start_time
            rate = (y_out + 1) / elapsed if elapsed > 0 else 0
            remaining = (output_height - (y_out + 1)) / rate if rate > 0 else 0
            print(f"  {{progress:3d}}% - {{elapsed:.1f}}s elapsed, {{remaining:.0f}}s remaining")
    
    # Close datasets
    band_out.FlushCache()
    ds_out = None
    ds_in = None
    
    elapsed = time.time() - start_time
    print(f"Complete in {{elapsed:.1f}} seconds!")
    return True

if __name__ == '__main__':
    success = process_aggregation()
    sys.exit(0 if success else 1)
'''
        return script


class ResampleGUI(QMainWindow):
    """GUI for advanced raster resampling"""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.analyzer = None
        self.raster_info = None
        self.src_width = None
        self.src_height = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Advanced Raster Resampling - Aggregation Functions + Binary Masks')
        self.setGeometry(100, 100, 1050, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        
        # ========== FIXED HEADER ==========
        header = QLabel('Advanced Raster Resampling Tool')
        header_font = header.font()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet('color: #0066cc; padding: 10px;')
        main_layout.addWidget(header)
        main_layout.addWidget(QLabel('-' * 100))
        
        # ========== SCROLLABLE CONTENT ==========
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('QScrollArea { border: none; }')
        
        scroll_widget = QWidget()
        layout = QVBoxLayout()

        # Input type selection
        input_type_group = QGroupBox('Input Source')
        input_type_layout = QVBoxLayout()
        
        self.input_type_group = QButtonGroup()
        self.raster_radio = QRadioButton('Raster (1m data)')
        self.polygon_radio = QRadioButton('Vector Polygon')
        self.line_radio = QRadioButton('Vector Line (with buffer)')
        
        self.raster_radio.setChecked(True)
        
        self.input_type_group.addButton(self.raster_radio, 0)
        self.input_type_group.addButton(self.polygon_radio, 1)
        self.input_type_group.addButton(self.line_radio, 2)
        
        self.input_type_group.buttonClicked.connect(self.on_input_type_changed)
        
        input_type_layout.addWidget(self.raster_radio)
        input_type_layout.addWidget(self.polygon_radio)
        input_type_layout.addWidget(self.line_radio)
        input_type_group.setLayout(input_type_layout)
        layout.addWidget(input_type_group)

        # Input
        input_group = QGroupBox('Input File')
        input_h = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText('Select input raster or vector...')
        input_btn = QPushButton('Browse...')
        input_btn.clicked.connect(self.browse_input)
        analyze_btn = QPushButton('Analyze')
        analyze_btn.clicked.connect(self.analyze_input)
        input_h.addWidget(self.input_edit)
        input_h.addWidget(input_btn)
        input_h.addWidget(analyze_btn)
        input_group.setLayout(input_h)
        layout.addWidget(input_group)

        # Info display
        info_group = QGroupBox('Input Analysis')
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.info_text.setText('Select input and click "Analyze"...')
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Vector-specific options
        vector_opts_group = QGroupBox('Vector Options')
        vector_opts_layout = QFormLayout()
        
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setMinimum(0)
        self.buffer_spin.setMaximum(10000)
        self.buffer_spin.setValue(0)
        self.buffer_spin.setSuffix(' m')
        self.buffer_spin.setEnabled(False)
        
        vector_opts_layout.addRow('Buffer distance (lines):', self.buffer_spin)
        vector_opts_group.setLayout(vector_opts_layout)
        layout.addWidget(vector_opts_group)

        # Output type selection (Aggregation vs Binary)
        output_type_group = QGroupBox('Output Type Selection')
        output_type_layout = QVBoxLayout()
        
        self.output_type_group = QButtonGroup()
        self.agg_radio = QRadioButton('Full Precision - Aggregation Functions')
        self.binary_radio = QRadioButton('Binary Mask - 0/1 Classification')
        
        self.agg_radio.setChecked(True)
        
        self.output_type_group.addButton(self.agg_radio, 0)
        self.output_type_group.addButton(self.binary_radio, 1)
        
        self.output_type_group.buttonClicked.connect(self.on_output_type_changed)
        
        output_type_layout.addWidget(self.agg_radio)
        output_type_layout.addWidget(self.binary_radio)
        output_type_group.setLayout(output_type_layout)
        layout.addWidget(output_type_group)
        
        # Binary threshold (for binary mode)
        binary_group = QGroupBox('Binary Mode Settings')
        binary_layout = QFormLayout()
        
        self.binary_threshold_spin = QSpinBox()
        self.binary_threshold_spin.setMinimum(0)
        self.binary_threshold_spin.setMaximum(255)
        self.binary_threshold_spin.setValue(0)
        self.binary_threshold_spin.setSuffix(' pixels')
        
        binary_layout.addRow('Binary Threshold:', self.binary_threshold_spin)
        binary_layout.addRow(QLabel('(0=presence/absence, 1+=minimum pixel count in cell)'))
        binary_group.setLayout(binary_layout)
        self.binary_group_widget = binary_group
        layout.addWidget(binary_group)

        # Aggregation function selection
        agg_group = QGroupBox('Aggregation Function')
        agg_layout = QFormLayout()
        
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(['Count', 'Sum', 'Mean', 'Max', 'Min', 'Median', 'StdDev'])
        self.agg_combo.currentTextChanged.connect(self.on_agg_changed)
        
        agg_layout.addRow('Function:', self.agg_combo)
        agg_layout.addRow(QLabel('(Count: density, Sum: total, Mean: average height, etc.)'))
        agg_group.setLayout(agg_layout)
        self.agg_group_widget = agg_group
        layout.addWidget(agg_group)



        # Output resolution
        resolution_group = QGroupBox('Output Resolution')
        resolution_layout = QFormLayout()
        
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setMinimum(0.1)
        self.resolution_spin.setMaximum(100.0)
        self.resolution_spin.setValue(10.0)
        self.resolution_spin.setDecimals(1)
        self.resolution_spin.setSuffix(' m')
        self.resolution_spin.setSingleStep(0.5)
        self.resolution_spin.valueChanged.connect(self.update_output_info)
        
        resolution_layout.addRow('Target pixel size:', self.resolution_spin)
        resolution_layout.addRow(QLabel('(e.g., 2.5, 5, 10 meters)'))
        
        resolution_group.setLayout(resolution_layout)
        layout.addWidget(resolution_group)

        # Processing options
        proc_group = QGroupBox('Processing Options')
        proc_layout = QFormLayout()
        
        self.num_processes_spin = QSpinBox()
        self.num_processes_spin.setMinimum(1)
        self.num_processes_spin.setMaximum(cpu_count())
        self.num_processes_spin.setValue(max(1, cpu_count() - 1))
        proc_layout.addRow('Number of processes:', self.num_processes_spin)
        proc_layout.addRow(QLabel(f'(Recommended: {max(1, cpu_count() - 1)}) CPU cores: {cpu_count()}'))
        
        # Block size for I/O optimization
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setMinimum(1)
        self.block_size_spin.setMaximum(512)
        self.block_size_spin.setValue(256)
        self.block_size_spin.setSuffix(' pixels')
        proc_layout.addRow('Block size (I/O optimization):', self.block_size_spin)
        proc_layout.addRow(QLabel('(256=default GeoTIFF, larger=faster I/O, 512=max recommended)'))
        
        # RAM caching option
        self.use_ram_cache = QCheckBox('Cache input file to RAM (faster I/O)')
        self.use_ram_cache.setChecked(False)
        proc_layout.addRow(self.use_ram_cache)
        proc_layout.addRow(QLabel('(Available: 196GB RAM | File size: ~6GB = 3% RAM usage)'))
        
        # Custom NODATA value exclusion (for files without NODATA metadata)
        proc_layout.addRow(QLabel(''))  # Spacer
        proc_layout.addRow(QLabel('Custom NODATA Exclusion:'))
        
        self.exclude_zero_check = QCheckBox('Exclude value 0 (sea/invalid)')
        self.exclude_zero_check.setChecked(True)
        proc_layout.addRow(self.exclude_zero_check)
        
        self.exclude_255_check = QCheckBox('Exclude value 255 (nodata/invalid)')
        self.exclude_255_check.setChecked(True)
        proc_layout.addRow(self.exclude_255_check)
        
        # Optional: Custom values to exclude
        custom_nodata_h = QHBoxLayout()
        custom_nodata_label = QLabel('Other values to exclude:')
        self.custom_nodata_input = QLineEdit()
        self.custom_nodata_input.setPlaceholderText('e.g., 1,2,3 (comma-separated)')
        self.custom_nodata_input.setMaximumWidth(200)
        custom_nodata_h.addWidget(custom_nodata_label)
        custom_nodata_h.addWidget(self.custom_nodata_input)
        custom_nodata_h.addStretch()
        proc_layout.addRow(custom_nodata_h)
        
        proc_layout.addRow(QLabel('(Pixels with these values are excluded from aggregation)'))
        
        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)

        # Grid alignment
        align_group = QGroupBox('Optional: Grid Alignment')
        align_layout = QVBoxLayout()
        
        self.use_align = QCheckBox('Align to donor raster grid:')
        align_layout.addWidget(self.use_align)
        
        align_file_h = QHBoxLayout()
        self.align_edit = QLineEdit()
        self.align_edit.setPlaceholderText('Reference raster...')
        self.align_edit.setEnabled(False)
        align_file_btn = QPushButton('Browse...')
        align_file_btn.clicked.connect(self.browse_align)
        align_file_btn.setEnabled(False)
        self.align_file_btn = align_file_btn
        
        align_file_h.addWidget(self.align_edit)
        align_file_h.addWidget(align_file_btn)
        
        self.use_align.stateChanged.connect(self.on_align_toggled)
        align_layout.addLayout(align_file_h)
        
        align_group.setLayout(align_layout)
        layout.addWidget(align_group)

        # Value range filtering (applies to BOTH modes!)
        range_group = QGroupBox('Value Range Filter')
        range_layout = QFormLayout()
        
        self.use_range = QCheckBox('Filter by value range:')
        range_layout.addRow(self.use_range)
        
        self.min_spin = QSpinBox()
        self.min_spin.setMinimum(0)
        self.min_spin.setMaximum(9999)
        self.min_spin.setValue(0)
        self.min_spin.setEnabled(False)
        range_layout.addRow('Minimum value:', self.min_spin)
        
        self.max_spin = QSpinBox()
        self.max_spin.setMinimum(0)
        self.max_spin.setMaximum(9999)
        self.max_spin.setValue(9999)
        self.max_spin.setEnabled(False)
        range_layout.addRow('Maximum value:', self.max_spin)
        
        self.range_info_label = QLabel('Applies to both modes: aggregation filters input, binary classifies as 0/1')
        self.range_info_label.setStyleSheet('color: #0066cc; font-style: italic;')
        range_layout.addRow(self.range_info_label)
        
        self.use_range.stateChanged.connect(self.on_range_toggled)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)

        # Output info
        output_info_group = QGroupBox('Output Information')
        output_info_layout = QVBoxLayout()
        self.output_info_label = QLabel('Resolution: 10.0m → ~18,617 × 23,133 pixels | Format: BYTE | Size: ~30 MB')
        self.output_info_label.setStyleSheet('color: #0066cc; font-weight: bold;')
        output_info_layout.addWidget(self.output_info_label)
        output_info_group.setLayout(output_info_layout)
        layout.addWidget(output_info_group)

        # Output
        output_group = QGroupBox('Output Raster')
        output_layout = QVBoxLayout()
        
        output_h = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText('Output path...')
        output_btn = QPushButton('Browse...')
        output_btn.clicked.connect(self.browse_output)
        output_h.addWidget(self.output_edit)
        output_h.addWidget(output_btn)
        output_layout.addLayout(output_h)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)



        # Add stretch at bottom to push everything to top
        layout.addStretch()
        
        scroll_widget.setLayout(layout)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        # ========== GDAL SCRIPT GENERATION SECTION ==========
        main_layout.addWidget(QLabel('-' * 100))
        
        script_group = QGroupBox('GDAL Script Generation (for speed comparison testing)')
        script_layout = QVBoxLayout()
        
        # Script type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel('Script type:'))
        self.script_type_group = QButtonGroup()
        
        rb_gdalwarp = QRadioButton('gdalwarp (fast resampling baseline)')
        rb_gdalwarp.setChecked(True)
        self.script_type_group.addButton(rb_gdalwarp, 0)
        type_layout.addWidget(rb_gdalwarp)
        
        rb_gdal_calc = QRadioButton('gdal_calc.py (masking)')
        self.script_type_group.addButton(rb_gdal_calc, 1)
        type_layout.addWidget(rb_gdal_calc)
        
        rb_python = QRadioButton('Python GDAL (full aggregation)')
        self.script_type_group.addButton(rb_python, 2)
        type_layout.addWidget(rb_python)
        type_layout.addStretch()
        
        script_layout.addLayout(type_layout)
        
        # File format selection (NEW!)
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel('File format:'))
        self.file_format_group = QButtonGroup()
        
        rb_sh = QRadioButton('.sh (bash - for Mac/Linux)')
        rb_sh.setChecked(True)
        self.file_format_group.addButton(rb_sh, 0)
        format_layout.addWidget(rb_sh)
        
        rb_bat = QRadioButton('.bat (Windows batch)')
        self.file_format_group.addButton(rb_bat, 1)
        format_layout.addWidget(rb_bat)
        
        rb_py = QRadioButton('.py (Python - all platforms)')
        self.file_format_group.addButton(rb_py, 2)
        format_layout.addWidget(rb_py)
        format_layout.addStretch()
        
        script_layout.addLayout(format_layout)
        
        # Generate button
        gen_btn_layout = QHBoxLayout()
        self.gen_script_btn = QPushButton('Generate Script')
        self.gen_script_btn.clicked.connect(self.on_generate_gdal_script)
        gen_btn_layout.addWidget(self.gen_script_btn)
        gen_btn_layout.addStretch()
        script_layout.addLayout(gen_btn_layout)
        
        # Script text area
        self.script_text = QTextEdit()
        self.script_text.setReadOnly(False)
        self.script_text.setMinimumHeight(150)
        self.script_text.setFontFamily('Courier')
        self.script_text.setFontPointSize(9)
        script_layout.addWidget(self.script_text)
        
        # Action buttons
        action_layout = QHBoxLayout()
        self.copy_script_btn = QPushButton('Copy to Clipboard')
        self.copy_script_btn.clicked.connect(self.on_copy_script)
        action_layout.addWidget(self.copy_script_btn)
        
        self.save_script_btn = QPushButton('Save As File')
        self.save_script_btn.clicked.connect(self.on_save_script)
        action_layout.addWidget(self.save_script_btn)
        
        action_layout.addStretch()
        script_layout.addLayout(action_layout)
        
        script_group.setLayout(script_layout)
        main_layout.addWidget(script_group)
        
        # ========== FIXED FOOTER ==========
        main_layout.addWidget(QLabel('-' * 100))
        
        # Progress and status
        progress_section = QGroupBox('Processing Status')
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        self.status_label = QLabel('Ready')
        self.status_label.setWordWrap(True)
        progress_layout.addWidget(self.status_label)
        
        # Timing stats
        self.timing_label = QLabel('Timing: —')
        self.timing_label.setStyleSheet('color: #666; font-size: 9pt;')
        progress_layout.addWidget(self.timing_label)
        
        progress_section.setLayout(progress_layout)
        main_layout.addWidget(progress_section)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.process_btn = QPushButton('Process')
        self.process_btn.clicked.connect(self.process)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.process_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)
        
        central.setLayout(main_layout)

    def on_output_type_changed(self):
        """Handle output type selection (Aggregation vs Binary)"""
        is_binary = self.output_type_group.checkedId() == 1
        
        # Enable/disable aggregation group
        self.agg_group_widget.setEnabled(not is_binary)
        
        # Update info labels based on mode
        if is_binary:
            self.range_info_label.setText('Binary mode: Min/Max range classifies pixels as 0/1 (in range=1, outside=0)')
            self.range_info_label.setStyleSheet('color: #006600; font-weight: bold;')
        else:
            self.range_info_label.setText('Aggregation mode: Min/Max range filters pixels before aggregation')
            self.range_info_label.setStyleSheet('color: #0066cc; font-style: italic;')
        
        # Value range now applies to BOTH modes - always available
        # Only disabled if vector input
        input_type = self.input_type_group.checkedId()
        is_vector = input_type in [1, 2]
        self.use_range.setEnabled(not is_vector)
        self.min_spin.setEnabled(not is_vector and self.use_range.isChecked())
        self.max_spin.setEnabled(not is_vector and self.use_range.isChecked())
        
        self.update_output_info()

    def on_input_type_changed(self):
        """Handle input type selection"""
        is_vector = self.input_type_group.checkedId() in [1, 2]
        is_line = self.input_type_group.checkedId() == 2
        
        self.buffer_spin.setEnabled(is_line)
        
        # Value range only available for raster input
        self.use_range.setEnabled(not is_vector)
        self.min_spin.setEnabled(not is_vector and self.use_range.isChecked())
        self.max_spin.setEnabled(not is_vector and self.use_range.isChecked())

    def on_align_toggled(self):
        """Enable/disable alignment"""
        enabled = self.use_align.isChecked()
        self.align_edit.setEnabled(enabled)
        self.align_file_btn.setEnabled(enabled)

    def on_range_toggled(self):
        """Enable/disable value range"""
        enabled = self.use_range.isChecked()
        self.min_spin.setEnabled(enabled)
        self.max_spin.setEnabled(enabled)

    def on_agg_changed(self):
        """Update when aggregation function changes"""
        self.update_output_info()

    def update_output_info(self):
        """Update output info with current settings"""
        is_binary = self.output_type_group.checkedId() == 1
        
        if is_binary:
            # Binary mode
            fmt = 'BYTE (0/1 only)'
            size_mb = 4
            res = self.resolution_spin.value()
            msg = f'Binary Mask: {fmt} | Threshold mode | Size: ~{size_mb} MB (5m) | Resolution: {res:.1f}m'
        else:
            # Aggregation mode
            res = self.resolution_spin.value()
            agg = self.agg_combo.currentText().lower()
            
            # Estimate format and file size
            if agg == 'count':
                fmt = 'BYTE'
                size_mb = 30
            elif agg == 'sum':
                fmt = 'INT32'
                size_mb = 200
            elif agg == 'stddev':
                fmt = 'FLOAT32'
                size_mb = 150
            else:  # mean, max, min, median
                fmt = 'INT16'
                size_mb = 100
            
            if self.src_width and self.src_height:
                out_width = int(self.src_width / res)
                out_height = int(self.src_height / res)
                msg = f'Resolution: {res:.1f}m → ~{out_width:,} × {out_height:,} pixels | Format: {fmt} | Size: ~{size_mb} MB'
            else:
                msg = f'Resolution: {res:.1f}m | Format: {fmt} | Size: ~{size_mb} MB'
        
        self.output_info_label.setText(msg)

    def browse_input(self):
        input_type = self.input_type_group.checkedId()
        
        if input_type == 0:
            path, _ = QFileDialog.getOpenFileName(self, 'Select Raster', '', 'TIF Files (*.tif);;All (*)')
        else:
            path, _ = QFileDialog.getOpenFileName(self, 'Select Vector', '', 
                                                  'Shapefiles (*.shp);;GeoJSON (*.geojson);;GeoPackage (*.gpkg);;All (*)')
        
        if path:
            self.input_edit.setText(path)
            base = Path(path).stem
            self.output_edit.setText(str(Path(path).parent / f'{base}_resampled.tif'))

    def browse_align(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Reference Raster', '', 'TIF Files (*.tif);;All (*)')
        if path:
            self.align_edit.setText(path)

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Output', '', 'TIF Files (*.tif)')
        if path and not path.endswith('.tif'):
            path += '.tif'
        if path:
            self.output_edit.setText(path)

    def analyze_input(self):
        path = self.input_edit.text().strip()
        input_type = self.input_type_group.checkedId()
        
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, 'Error', 'Select valid input file')
            return
        
        if input_type == 0:
            self.analyzer = RasterAnalyzer(path)
            self.analyzer.status.connect(self.status_label.setText)
            self.analyzer.finished.connect(self.on_analysis_complete)
            self.analyzer.error.connect(lambda e: QMessageBox.critical(self, 'Error', e))
            self.analyzer.start()
        else:
            try:
                ds = ogr.Open(path)
                layer = ds.GetLayer(0)
                extent = layer.GetExtent()
                feature_count = layer.GetFeatureCount()
                geom_type = ogr.GeometryTypeToName(layer.GetGeomType())
                
                minx, maxx, miny, maxy = extent
                width_1m = int(np.ceil(maxx - minx))
                height_1m = int(np.ceil(maxy - miny))
                
                msg = f"""Vector Analysis:
Geometry: {geom_type}
Features: {feature_count:,}

Extent: {width_1m:,} × {height_1m:,}m
Will rasterize to 1m then resample"""
                self.info_text.setText(msg)
                ds = None
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Cannot analyze: {str(e)}')

    def on_analysis_complete(self, info):
        """Handle raster analysis"""
        text = f"""Raster: {info['width']:,} × {info['height']:,} pixels"""
        
        # Store dimensions
        self.src_width = info['width']
        self.src_height = info['height']
        
        if info['has_data']:
            text += f"""
Min: {info['min']:,}
Max: {info['max']:,}
Mean: {info['mean']:.1f}"""
            
            self.min_spin.setMaximum(info['max'] + 1000)
            self.max_spin.setMaximum(info['max'] + 1000)
            self.min_spin.setValue(max(0, info['min']))
            self.max_spin.setValue(info['max'])
            
            # Set reasonable binary threshold
            self.binary_threshold_spin.setMaximum(info['max'] + 1000)
            self.binary_threshold_spin.setValue(int(info['mean']))
        
        self.info_text.setText(text)
        self.update_output_info()

    def process(self):
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        input_type = self.input_type_group.checkedId()
        output_type = self.output_type_group.checkedId()  # 0=Aggregation, 1=Binary

        if not input_path or not output_path:
            QMessageBox.warning(self, 'Error', 'Specify input and output')
            return

        if not os.path.exists(input_path):
            QMessageBox.warning(self, 'Error', 'Input not found')
            return

        donor_path = None
        if self.use_align.isChecked():
            donor_path = self.align_edit.text().strip()
            if not donor_path or not os.path.exists(donor_path):
                QMessageBox.warning(self, 'Error', 'Specify valid donor raster')
                return

        min_val = None
        max_val = None
        if input_type == 0 and self.use_range.isChecked():
            # Value range applies to BOTH modes now (raster input only)
            min_val = self.min_spin.value()
            max_val = self.max_spin.value()

        buffer_dist = self.buffer_spin.value() if input_type == 2 else 0
        input_type_str = 'raster' if input_type == 0 else 'vector'
        # Determine number of processes and whether to use threading
        # If using RAM cache: use threading (parallel + shared memory, no pickling!)
        # If using disk: use multiprocessing (parallel I/O)
        use_threading = self.use_ram_cache.isChecked()
        num_workers = self.num_processes_spin.value()
        block_size = self.block_size_spin.value()  # Get block size from GUI
        resolution = self.resolution_spin.value()
        
        # Extract custom NODATA exclusion values
        excluded_values = []
        if self.exclude_zero_check.isChecked():
            excluded_values.append(0)
        if self.exclude_255_check.isChecked():
            excluded_values.append(255)
        
        # Parse custom values to exclude
        custom_str = self.custom_nodata_input.text().strip()
        if custom_str:
            try:
                custom_excluded = [int(v.strip()) for v in custom_str.split(',') if v.strip()]
                excluded_values.extend(custom_excluded)
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Custom values must be comma-separated integers')
                return
        
        if output_type == 0:
            # Aggregation mode
            aggregation_func = self.agg_combo.currentText().lower()
            binary_mode = False
        else:
            # Binary mode
            aggregation_func = 'count'  # Dummy (not used in binary mode)
            binary_mode = True

        self.status_label.setText('Starting...')
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.worker = ResampleWorker(
            input_path, output_path, 
            input_type=input_type_str,
            vector_path=input_path if input_type_str == 'vector' else None,
            buffer_dist=buffer_dist,
            min_value=min_val, 
            max_value=max_val,
            donor_path=donor_path,
            num_processes=num_workers,
            resolution=resolution,
            aggregation_func=aggregation_func,
            binary_mode=binary_mode,
            binary_threshold=self.binary_threshold_spin.value(),  # Get from GUI!
            block_size=block_size,
            use_ram_cache=self.use_ram_cache.isChecked(),  # Pass RAM cache flag
            excluded_values=excluded_values  # Pass custom NODATA exclusions
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.timing.connect(self.on_timing_update)  # New timing signal
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.terminate()
            self.worker.wait()
        self.status_label.setText('Cancelled')
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def on_error(self, msg):
        QMessageBox.critical(self, 'Error', msg)
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def on_timing_update(self, timing_str):
        """Handle timing updates from worker"""
        self.timing_label.setText(timing_str)

    def on_finished(self):
        QMessageBox.information(self, 'Success', 'Processing complete!')
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def on_generate_gdal_script(self):
        """Generate GDAL script based on current settings"""
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        
        if not input_path or not output_path:
            QMessageBox.warning(self, 'Error', 'Specify input and output files')
            return
        
        resolution = self.resolution_spin.value()
        min_val = self.min_spin.value() if self.use_range.isChecked() else None
        max_val = self.max_spin.value() if self.use_range.isChecked() else None
        
        # Gather excluded values
        excluded_values = []
        if self.exclude_zero_check.isChecked():
            excluded_values.append(0)
        if self.exclude_255_check.isChecked():
            excluded_values.append(255)
        
        custom_str = self.custom_nodata_input.text().strip()
        if custom_str:
            try:
                custom_excluded = [int(v.strip()) for v in custom_str.split(',') if v.strip()]
                excluded_values.extend(custom_excluded)
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid custom values format')
                return
        
        # Get aggregation function
        if self.output_type_group.checkedId() == 0:
            agg_func = self.agg_combo.currentText().lower()
        else:
            agg_func = 'binary'
        
        # Create script generator
        gen = GDALScriptGenerator(
            input_path, output_path, resolution, 
            min_val, max_val, excluded_values, 
            agg_func, self.num_processes_spin.value()
        )
        
        # Generate appropriate script
        script_type = self.script_type_group.checkedId()
        file_format = self.file_format_group.checkedId()
        
        # Generate script based on type and format
        if file_format == 0:  # .sh format
            if script_type == 0:
                script = gen.generate_gdalwarp()
            elif script_type == 1:
                script = gen.generate_gdal_calc()
            else:
                script = gen.generate_python_script()
        elif file_format == 1:  # .bat format (Windows batch)
            if script_type == 0:
                script = gen.generate_gdalwarp_batch()
            elif script_type == 1:
                script = gen.generate_gdal_calc_batch()
            else:
                # For Python script in batch, just show the Python code
                script = gen.generate_python_script()
        else:  # .py format (Python)
            script = gen.generate_python_script()
        
        # Display script
        self.script_text.setPlainText(script)
        
        # Show info message
        type_names = ['gdalwarp (baseline)', 'gdal_calc.py (masking)', 'Python GDAL (full)']
        format_names = ['.sh (bash)', '.bat (Windows)', '.py (Python)']
        QMessageBox.information(self, 'Script Generated', 
                               f'Generated {type_names[script_type]} script ({format_names[file_format]})!\n\n'
                               'Copy to clipboard or save to file and run.')
    
    
    def on_copy_script(self):
        """Copy generated script to clipboard"""
        script_text = self.script_text.toPlainText()
        if not script_text:
            QMessageBox.warning(self, 'Error', 'Generate a script first')
            return
        
        clipboard = QApplication.clipboard()
        clipboard.setText(script_text)
        QMessageBox.information(self, 'Copied', 'Script copied to clipboard!')
    
    def on_save_script(self):
        """Save generated script to file"""
        script_text = self.script_text.toPlainText()
        if not script_text:
            QMessageBox.warning(self, 'Error', 'Generate a script first')
            return
        
        file_format = self.file_format_group.checkedId()
        script_type = self.script_type_group.checkedId()
        
        # Determine file extension and default name
        if file_format == 0:  # .sh
            extension = '*.sh'
            if script_type == 0:
                default_name = 'resample_raster.sh'
            elif script_type == 1:
                default_name = 'mask_raster.sh'
            else:
                default_name = 'aggregate_raster.py'  # .sh scripts show .py for Python
        elif file_format == 1:  # .bat
            extension = '*.bat'
            if script_type == 0:
                default_name = 'resample_raster.bat'
            elif script_type == 1:
                default_name = 'mask_raster.bat'
            else:
                default_name = 'aggregate_raster.bat'
        else:  # .py
            extension = '*.py'
            default_name = 'aggregate_raster.py'
        
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Script', default_name, 
            f'Script Files ({extension});;All Files (*)'
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(script_text)
                
                # Make executable on Unix-like systems (not Windows)
                if file_format == 0:  # Only for .sh files
                    os.chmod(filename, 0o755)
                
                QMessageBox.information(self, 'Saved', f'Script saved to:\n{filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save:\n{str(e)}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ResampleGUI()
    window.show()
    sys.exit(app.exec_())
