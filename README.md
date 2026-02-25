# Advanced Raster Resampling Tool

A professional-grade GUI tool for resampling and aggregating large geospatial rasters with custom masking, value filtering, and cross-platform script generation for speed comparison testing.

**Perfect for environmental monitoring, LiDAR analysis, and GIS workflows.**

---

## Features

### Core Functionality
- **Advanced Aggregation Functions**: COUNT, SUM, MEAN, MAX, MIN, MEDIAN, STDDEV
- **Binary Classification**: Create binary masks from raster data
- **Custom NODATA Handling**: Exclude specific values (0, 255, or custom) from aggregation
- **Value Range Filtering**: Filter pixels by minimum/maximum values
- **Multi-threaded Processing**: Configurable parallel processing for faster execution
- **Grid Alignment**: Align output to reference raster or custom grid
- **Professional Output**: GeoTIFF format with proper georeferencing

### GDAL Script Generator (Speed Comparison Testing)
Generate command-line scripts in multiple formats for benchmarking:

- **gdalwarp Scripts** (.sh/.bat): Native GDAL resampling baseline (~5-15 seconds)
- **gdal_calc.py Scripts** (.sh/.bat): Value filtering and masking (~10-20 seconds)  
- **Python GDAL Scripts** (.py): Full aggregation implementation (~38-45 seconds)

Perfect for understanding performance tradeoffs between:
- Native C code (gdalwarp) vs Python implementation
- Interpolation vs aggregation algorithms
- When to use each approach for your use case

### Multi-Platform Support
- **Windows**: Generate .bat (batch) files + Python scripts
- **Mac/Linux**: Generate .sh (bash) files + Python scripts
- **All Platforms**: Generate portable Python scripts

---

## Installation

### Requirements
- Python 3.6+
- PyQt5
- GDAL/osgeo (with Python bindings)
- NumPy

### Setup

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   # Using conda (recommended for geospatial work)
   conda install -c conda-forge gdal numpy PyQt5
   
   # Using pip
   pip install gdal numpy PyQt5
   ```

3. **Verify GDAL installation** (for script generation features)
   ```bash
   gdalwarp --version
   gdal_calc.py --version
   ```

4. **Run the tool**
   ```bash
   python3 resample_COMPLETE_V1.py
   ```

---

## Quick Start

### Basic Usage (GUI)

1. **Load your raster**
   - Click "Browse..." next to Input
   - Select your raster file (.tif, .tiff)

2. **Set output path**
   - Click "Browse..." next to Output
   - Choose where to save results

3. **Configure analysis**
   - **Resolution**: Output pixel size (e.g., 10m)
   - **Aggregation**: COUNT, SUM, MEAN, MAX, MIN, MEDIAN, STDDEV
   - **Value Range**: Min/max filter (e.g., 0-60)
   - **Exclusions**: Check boxes for 0, 255, or custom values

4. **Process**
   - Click "Process"
   - Wait for completion
   - Output file created at specified location

### Speed Comparison Testing

1. **Configure your analysis** (as above)

2. **Generate scripts for comparison**
   - Scroll to "GDAL Script Generation" section
   - Select script type: gdalwarp, gdal_calc, or Python GDAL
   - Select file format: .sh (Linux/Mac), .bat (Windows), or .py (all)
   - Click "Generate Script"

3. **Run scripts and measure**
   ```bash
   # Linux/Mac
   bash resample_raster.sh        # Expected: 5-15 seconds
   bash mask_raster.sh            # Expected: 10-20 seconds
   python3 aggregate_raster.py    # Expected: 38-45 seconds
   
   # Windows
   resample_raster.bat            # Expected: 5-15 seconds
   mask_raster.bat                # Expected: 10-20 seconds
   python3 aggregate_raster.py    # Expected: 38-45 seconds
   ```

4. **Compare results**
   - Calculate speedup ratios
   - Document your findings
   - Share with colleagues

---

## Usage Examples

### Example 1: ERAMMP Survey Data Aggregation

```
Input: 1m resolution habitat classification raster (6GB)
Settings:
  - Resolution: 10m
  - Aggregation: COUNT
  - Value range: 1-60 (valid habitat classes)
  - Exclude: 0 (water), 255 (nodata)
  
Output: 10m habitat count raster
Processing time: ~40 seconds
```

### Example 2: Speed Comparison Research

```
Input: Welsh LiDAR DEM (1m, 8GB)
Settings:
  - Resolution: 10m
  - Aggregation: MEAN
  - Value range: -50 to 1500 (elevation in meters)

Test three approaches:
1. gdalwarp baseline (resampling): 8 seconds
2. gdal_calc masking: 14 seconds
3. Python aggregation: 41 seconds

Finding: gdalwarp is 5.1x faster but uses different algorithm
```

### Example 3: Binary Classification

```
Input: Vegetation height raster
Settings:
  - Output type: Binary
  - Threshold: 5m
  
Output: 1 = tall vegetation (≥5m), 0 = short/no vegetation
```

---

## GUI Walkthrough

### Main Window Layout

```
┌─────────────────────────────────────────────┐
│ Input/Output File Selection                 │
│  [Browse Input] [Browse Output]             │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│ Processing Options                          │
│  Resolution: ____  Num Processes: ____      │
│  Value Range: ___ to ___                    │
│  ☑ Exclude 0  ☑ Exclude 255  ☑ Custom: ___ │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│ Output Type                                 │
│  ◉ Aggregation (count, sum, mean, etc)     │
│  ◉ Binary Classification (threshold mask)   │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│ GDAL Script Generation (Speed Testing)       │
│  Script type: ◉ gdalwarp ◉ gdal_calc ◉ Python │
│  Format: ◉ .sh ◉ .bat ◉ .py                │
│  [Generate] [Copy] [Save]                   │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│ [Process] [Cancel]                          │
└─────────────────────────────────────────────┘
```

---

## Command-Line Script Generation

Scripts are generated with your current settings and can be:
- **Copied to clipboard** for immediate use
- **Saved as files** for batch processing
- **Shared** with colleagues for reproducibility
- **Automated** in workflows

### Generated Script Examples

**gdalwarp.bat (Windows Batch)**
```batch
@echo off
set INPUT=D:\data\raster_1m.tif
set OUTPUT=D:\data\raster_10m.tif
gdalwarp -r average -tr 10.0 10.0 -multi "%INPUT%" "%OUTPUT%"
pause
```

**gdal_calc.sh (Bash)**
```bash
#!/bin/bash
INPUT="/path/to/raster.tif"
OUTPUT="/path/to/output.tif"
gdal_calc.py -A "$INPUT" --outfile="$OUTPUT" \
  --calc="A*((A>=1)*(A<=60))*(A!=0)*(A!=255)" \
  --type=Byte --NoDataValue=0
```

**aggregate_raster.py (Python - Portable)**
```python
#!/usr/bin/env python3
# Standalone Python GDAL script
# Full aggregation with all masking/filtering
# Works on Windows, Mac, Linux
```

---

## Advanced Features

### Custom NODATA Exclusion

The tool allows excluding specific pixel values from aggregation:

```
Preset exclusions:
  ☑ Exclude 0 (common for nodata/water)
  ☑ Exclude 255 (common for nodata)
  
Custom exclusions:
  [Other values: 1, 2, 3] (comma-separated)
```

This prevents invalid/sentinel values from affecting your aggregation results.

### Multi-threaded Processing

- Configurable number of processing threads
- Default: 23 threads (optimized for Ryzen/Intel CPUs)
- Adjust based on your system specs
- Faster processing on multi-core systems

### Grid Alignment

Align output raster to:
- **Reference raster**: Use georeferencing from existing raster
- **Custom grid**: Define custom grid origin and spacing
- Ensures compatible multi-raster datasets

---

## Performance

### Typical Processing Times

| Data Size | Resolution | Function | Time |
|-----------|-----------|----------|------|
| 1 GB | 1m → 10m | COUNT | ~4 seconds |
| 6 GB | 1m → 10m | COUNT | ~40 seconds |
| 6 GB | 1m → 5m | MEAN | ~150 seconds |

### Speed Comparison (6GB ERAMMP Data)

| Approach | Time | Speed vs Python |
|----------|------|-----------------|
| gdalwarp (baseline) | ~8s | 5x faster |
| gdal_calc (masking) | ~15s | 2.7x faster |
| Python (aggregation) | ~40s | baseline |

*Note: Different algorithms produce different results*

---

## Outputs

### File Format
- **GeoTIFF** (.tif) - Professional GIS format
- **Int32 or Float32** - Automatic based on aggregation function
- **Proper Georeferencing** - Maintains coordinate system and extents
- **NODATA Value** - 0 by default

### Quality Assurance
- Automatic data type preservation
- Proper georeferencing maintained
- NODATA values handled correctly
- Comprehensive error checking

---

## Workflow Examples

### Batch Processing Multiple Rasters

```bash
# Linux/Mac: create process_all.sh
#!/bin/bash
for input_file in /data/input/*.tif; do
    filename=$(basename "$input_file" .tif)
    output_file="/data/output/${filename}_10m.tif"
    python3 aggregate_raster.py "$input_file" "$output_file"
done

# Run: bash process_all.sh
```

```batch
REM Windows: create process_all.bat
@echo off
for %%f in (D:\data\input\*.tif) do (
    echo Processing %%~nf
    call resample_raster.bat %%f
)

REM Run: process_all.bat
```

### Integration with QGIS

1. Generate aggregated raster using this tool
2. Load result in QGIS for visualization
3. Compare with other datasets
4. Export for further analysis

### Research Workflows

1. Process environmental monitoring data
2. Generate aggregated products at different resolutions
3. Compare results with published datasets
4. Document methods with generated scripts
5. Share reproducible workflows with colleagues

---

## Troubleshooting

### Issue: "GDAL not found"
**Solution**: Ensure GDAL is installed and in system PATH
```bash
# Test installation
gdalwarp --version
gdal_calc.py --version
```

### Issue: Script generation works but script won't run
**Solution**: Make sure GDAL tools are accessible from command line

### Issue: Memory errors with very large rasters
**Solution**: 
- Reduce number of threads
- Process in smaller tiles manually
- Increase system RAM if possible

### Issue: Output raster looks wrong
**Solution**:
- Check NODATA value handling (0, 255, custom)
- Verify value range filter is appropriate
- Test with smaller subset first
- Review aggregation function choice

### Issue: Slow performance
**Solution**:
- Increase number of threads
- Use faster aggregation function (COUNT faster than STDDEV)
- Store data on faster disk
- Close other applications

---

## Contributing

This tool was developed for environmental monitoring and GIS research. Contributions are welcome!

### Areas for Enhancement
- GPU acceleration for processing
- Support for multi-band rasters
- Additional aggregation functions
- Cloud-optimized GeoTIFF support
- REST API for remote processing

---

## Citation

If you use this tool in research, please cite:

```
Advanced Raster Resampling Tool (Version 8.0)
For environmental monitoring and GIS analysis
GitHub: [your-github-repo]
```

---

## License

[Specify your license - MIT, GPL, Apache 2.0, etc.]

---

## Author

Developed for geospatial data analysis and environmental monitoring workflows.

**Featuring:**
- GUI-based raster processing
- Custom value exclusion masking
- Cross-platform script generation
- Speed comparison capabilities

---

## Support & Documentation

### Included Documentation
- `GDAL_SCRIPT_GENERATOR_COMPLETE_V8.txt` - Detailed feature documentation
- `GDAL_SCRIPT_GENERATOR_QUICK_REFERENCE.txt` - Quick start guide
- `HOW_TO_RUN_SH_SCRIPTS.txt` - Script execution guide
- `WINDOWS_BAT_FILE_SUPPORT.txt` - Windows batch files guide
- `IMPLEMENTATION_COMPLETE.txt` - Technical implementation details

### Getting Help
1. Check documentation files
2. Review troubleshooting section
3. Test with sample data first
4. Check console output for error messages

---

## Acknowledgments

Built with:
- **PyQt5** - GUI framework
- **GDAL** - Geospatial data processing
- **NumPy** - Numerical computing
- **Python** - Core language

Perfect for:
- ERAMMP survey data analysis
- LiDAR processing and classification
- Environmental monitoring
- Habitat mapping
- Raster data standardization

---

## Version History

### v8.0 (Current)
- ✅ Windows .bat file support
- ✅ File format selector (sh/bat/py)
- ✅ Path conversion for cross-platform compatibility
- ✅ Three GDAL script types
- ✅ Copy to clipboard functionality
- ✅ Professional documentation

### v7.1
- ✅ Custom NODATA exclusion (0, 255, custom values)
- ✅ Value range filtering
- ✅ Threading bug fixes
- ✅ GUI improvements

### v7.0
- ✅ Multi-threaded processing
- ✅ All aggregation functions
- ✅ Binary classification
- ✅ Grid alignment support

---

## Ready to Use! 🚀

Download the tool and start processing your geospatial data today!

### Quick Links
- [Installation Guide](#installation)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Documentation Files](#support--documentation)

**Questions? Check the included documentation files or review the troubleshooting section.**

---

## Next Steps

1. Download and install
2. Review quick start guide
3. Test with sample data
4. Process your rasters
5. Generate comparison scripts
6. Document your workflows
7. Share with colleagues

**Enjoy! 🎯**
