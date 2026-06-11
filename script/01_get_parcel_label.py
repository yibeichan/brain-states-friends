#!/usr/bin/env python
"""
Extract and Process Parcellation Labels for Brain Atlases

This script extracts human-readable labels for brain parcellations from various
atlas formats and saves them in multiple formats for easy access during analysis.
It handles both Schaefer and atlas-4S parcellation schemes.

PURPOSE:
========
Brain parcellations divide the brain into regions identified by numeric IDs.
This script maps those IDs to meaningful anatomical/functional labels (e.g., 
"17Networks_LH_VisCent_ExStr_1" or "Left-Amygdala") for easier interpretation
of analysis results.

SUPPORTED PARCELLATIONS:
========================
1. Schaefer Parcellations:
   - Schaefer2018_400Parcels with Tian Subcortex
   - Format: Combined cortical (Schaefer) + subcortical (Tian) regions

2. Atlas-4S Parcellations:
   - atlas-4S456Parcels through atlas-4S1056Parcels
   - Cortical regions: Use 17-network labels for functional interpretation
   - Subcortical regions: Last 56 parcels use anatomical labels

INPUT FILES:
============
- Schaefer: .txt label files with format:
  label_name
  id r g b a
  
- Atlas-4S: .tsv files with columns:
  - index: parcel ID
  - label: anatomical name (used for subcortical)
  - label_17network: functional network (used for cortical)

OUTPUT FILES:
=============
For each parcellation, creates three files:
1. {parcellation}_labels.npy: NumPy array indexed by parcel ID
2. {parcellation}_labels.txt: Human-readable text format
3. {parcellation}_labels.csv: CSV format with columns [parcel_id, label_name]

OUTPUT STRUCTURE:
=================
/scratch/data/parcellation_labels/
├── Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4_labels.npy
├── Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4_labels.txt
├── Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4_labels.csv
├── atlas-4S456Parcels_labels.npy
├── atlas-4S456Parcels_labels.txt
├── atlas-4S456Parcels_labels.csv
└── ... (other parcellations)

USAGE:
======
# Process all parcellations
python 01_get_parcel_label.py

# Process specific parcellation
python 01_get_parcel_label.py --parcellation 456
python 01_get_parcel_label.py --parcellation Schaefer

ALGORITHM:
==========
1. Identify parcellation type from filename
2. Locate corresponding label file (.txt for Schaefer, .tsv for atlas-4S)
3. Parse labels according to format:
   - Schaefer: Extract from paired lines (name, then ID+colors)
   - Atlas-4S: Use label_17network for cortical, label for subcortical
4. Save in three formats for different use cases

Author: Yibei Chen (yc-nn branch)
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import argparse
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

def parse_schaefer_labels(label_file):
    """
    Parse Schaefer label file format.
    
    Parameters
    ----------
    label_file : str
        Path to Schaefer label .txt file
        
    Returns
    -------
    dict
        Dictionary mapping parcel IDs to label names
        
    Notes
    -----
    Expected format: alternating lines with label name, then "id r g b a"
    Example:
        17Networks_LH_VisCent_ExStr_1
        1 120 18 134 255
    """
    labels = {}
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    # Process pairs of lines
    i = 0
    while i < len(lines):
        # First line contains the label name
        label_name = lines[i].strip()
        
        # Second line contains the ID and color values
        if i + 1 < len(lines):
            parts = lines[i + 1].strip().split()
            if len(parts) >= 1:
                parcel_id = int(parts[0])
                labels[parcel_id] = label_name
        
        i += 2
    
    return labels

def parse_atlas_labels(csv_file):
    """
    Parse atlas TSV file format with intelligent label selection.
    
    Parameters
    ----------
    csv_file : str
        Path to atlas .tsv file
        
    Returns
    -------
    dict
        Dictionary mapping parcel IDs to label names
        
    Notes
    -----
    Label selection strategy:
    - For atlas-4S: 
        * Cortical parcels: Use 'label_17network' for functional network labels
        * Subcortical parcels (last 56): Use 'label' for anatomical names
    - For other atlases (HCP, Glasser, Gordon, Tian):
        * Use 'label' column if 'label_17network' doesn't exist
        * Otherwise use 'label_17network'
    
    The threshold for 4S atlases is automatically determined from the filename:
    e.g., atlas-4S1056Parcels -> 1000 cortical + 56 subcortical
    """
    # Read CSV file
    df = pd.read_csv(csv_file, sep='\t')
    
    # Check which columns are available
    has_17network = 'label_17network' in df.columns
    has_label = 'label' in df.columns
    
    # Extract atlas size from filename to determine if this is a 4S atlas
    # e.g., "atlas-4S1056Parcels_dseg.tsv" -> 1056
    filename = os.path.basename(csv_file)
    import re
    match = re.search(r'atlas-4S(\d+)Parcels', filename)
    is_4s_atlas = match is not None
    
    if is_4s_atlas and match:
        total_parcels = int(match.group(1))
        # Last 56 parcels are subcortical
        subcortical_threshold = total_parcels - 56
    else:
        # For non-4S atlases, no threshold needed
        subcortical_threshold = None
    
    labels = {}
    
    # Process each row
    for idx, row in df.iterrows():
        parcel_id = row['index']
        
        # Determine which label to use
        if is_4s_atlas and subcortical_threshold is not None:
            # For 4S atlases: use label_17network for cortical, label for subcortical
            if parcel_id > subcortical_threshold:
                label_name = row['label']
            else:
                label_name = row['label_17network'] if has_17network else row['label']
        else:
            # For other atlases: prefer label_17network if available, otherwise use label
            if has_17network:
                label_name = row['label_17network']
            elif has_label:
                label_name = row['label']
            else:
                raise KeyError("Neither 'label' nor 'label_17network' column found in TSV file")
        
        labels[parcel_id] = label_name
    
    return labels

def get_label_file_path(parc_path):
    """
    Determine the label file path based on parcellation filename.
    
    Parameters
    ----------
    parc_path : str
        Path to parcellation .dlabel.nii file
        
    Returns
    -------
    str or None
        Path to corresponding label file, or None if cannot be determined
        
    Notes
    -----
    Mapping logic:
    - Schaefer2018*.dlabel.nii -> Schaefer2018*_label.txt or .txt
    - atlas-*Parcels*.dlabel.nii -> atlas-*Parcels_dseg.tsv (for 4S atlases)
    - atlas-*.dlabel.nii -> atlas-*_dseg.tsv (for other atlases like HCP, Glasser, etc.)
    """
    parc_dir = os.path.dirname(parc_path)
    parc_basename = os.path.basename(parc_path)
    
    # For Schaefer parcellation
    if "Schaefer2018" in parc_basename:
        # Look for .txt file with similar name
        label_file = parc_path.replace('.dlabel.nii', '_label.txt')
        if os.path.exists(label_file):
            return label_file
        # Try without the .dlabel extension
        label_file = parc_path.replace('.dlabel.nii', '.txt')
        return label_file
    
    # For atlas parcellations (atlas-4S, atlas-HCP, atlas-Glasser, etc.)
    elif "atlas-" in parc_basename:
        # Extract the atlas name (e.g., "atlas-4S1056Parcels", "atlas-HCP", "atlas-Glasser")
        atlas_name = parc_basename.split('_space')[0]
        # Look for TSV file
        return os.path.join(parc_dir, f"{atlas_name}_dseg.tsv")
    
    return None

def get_parcel_sizes(parc_path):
    """
    Compute parcel sizes (number of vertices/voxels per parcel) from dlabel.nii file.
    
    Parameters
    ----------
    parc_path : str
        Path to parcellation .dlabel.nii file
        
    Returns
    -------
    dict
        Dictionary mapping parcel IDs to their sizes (number of vertices/voxels)
        
    Notes
    -----
    Uses nibabel to load the dlabel.nii file and count unique values.
    Background (value 0) is excluded from the count.
    """
    try:
        # Load the dlabel.nii file
        img = nib.load(parc_path)
        data = img.get_fdata()
        
        # Flatten and get unique values
        unique_values, counts = np.unique(data, return_counts=True)
        
        # Create parcel sizes dictionary (exclude background value 0)
        parcel_sizes = {}
        for value, count in zip(unique_values, counts):
            if value != 0:  # Skip background
                parcel_sizes[int(value)] = int(count)
        
        print(f"Computed parcel sizes: {len(parcel_sizes)} parcels")
        return parcel_sizes
        
    except Exception as e:
        print(f"ERROR: Could not compute parcel sizes from {parc_path}: {e}")
        return None

def process_parcellation_labels(parc_path, output_dir):
    """
    Process labels for a single parcellation and save in multiple formats.
    
    Parameters
    ----------
    parc_path : str
        Path to parcellation .dlabel.nii file
    output_dir : str
        Directory to save output label files
        
    Returns
    -------
    dict or None
        Dictionary of labels if successful, None if failed
        
    Side Effects
    ------------
    Creates four files in output_dir:
    - {parcellation}_labels.npy: NumPy array for fast loading
    - {parcellation}_labels.txt: Human-readable text format
    - {parcellation}_labels.csv: CSV format for data analysis
    - {parcellation}_parcel_sizes.csv: Parcel sizes for variance analysis
    """
    print(f"\nProcessing labels for: {os.path.basename(parc_path)}")
    
    # Get label file path
    label_file = get_label_file_path(parc_path)
    
    if label_file is None:
        print(f"Could not determine label file path for {parc_path}")
        return None
    
    if not os.path.exists(label_file):
        print(f"Label file not found: {label_file}")
        # List files in directory to help debug
        parc_dir = os.path.dirname(parc_path)
        print(f"Files in {parc_dir}:")
        for f in os.listdir(parc_dir):
            print(f"  {f}")
        return None
    
    print(f"Found label file: {label_file}")
    
    # Parse labels based on format
    try:
        if label_file.endswith('.txt'):
            labels = parse_schaefer_labels(label_file)
            print(f"Parsed Schaefer format: {len(labels)} labels")
        elif label_file.endswith('.csv') or label_file.endswith('.tsv'):
            labels = parse_atlas_labels(label_file)
            print(f"Parsed atlas format: {len(labels)} labels")
        else:
            print(f"ERROR: Unknown label file format: {label_file}")
            print(f"  Expected: .txt (Schaefer) or .tsv/.csv (atlas)")
            return None
    except FileNotFoundError as e:
        print(f"ERROR: Label file not found: {e}")
        return None
    except KeyError as e:
        print(f"ERROR: Missing expected column in label file: {e}")
        print(f"  Expected columns for atlas TSV: 'index' and either 'label' or 'label_17network'")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error parsing label file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Get parcellation name for output
    parc_basename = os.path.basename(parc_path)
    if "Schaefer2018" in parc_basename:
        parc_name = "Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4"
    elif "atlas-" in parc_basename:
        # For all atlas types (4S, HCP, Glasser, Gordon, Tian)
        parc_name = parc_basename.split('_space')[0]
    else:
        parc_name = parc_basename.replace('.dlabel.nii', '').replace('.nii', '')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save labels as numpy array (ordered by parcel ID)
    max_id = max(labels.keys())
    label_array = [''] * (max_id + 1)  # Initialize with empty strings
    
    for parcel_id, label_name in labels.items():
        label_array[parcel_id] = label_name
    
    # Save as both numpy array and text file for easy reference
    output_base = os.path.join(output_dir, f"{parc_name}_labels")
    
    # Save as numpy array
    np.save(f"{output_base}.npy", label_array)
    
    # Save as text file for human readability
    with open(f"{output_base}.txt", 'w') as f:
        f.write(f"# Parcel labels for {parc_name}\n")
        f.write("# Format: parcel_id\tlabel_name\n")
        for parcel_id in sorted(labels.keys()):
            f.write(f"{parcel_id}\t{labels[parcel_id]}\n")
    
    # Save as CSV for easy loading
    df = pd.DataFrame(list(labels.items()), columns=['parcel_id', 'label_name'])
    df = df.sort_values('parcel_id')
    df.to_csv(f"{output_base}.csv", index=False)
    
    # Compute and save parcel sizes
    parcel_sizes = get_parcel_sizes(parc_path)
    if parcel_sizes is not None:
        sizes_df = pd.DataFrame(list(parcel_sizes.items()), columns=['parcel_id', 'parcel_size'])
        sizes_df = sizes_df.sort_values('parcel_id')
        sizes_output = os.path.join(output_dir, f"{parc_name}_parcel_sizes.csv")
        sizes_df.to_csv(sizes_output, index=False)
        print(f"Saved parcel sizes to: {sizes_output}")
    
    print(f"Saved labels to:")
    print(f"  - {output_base}.npy")
    print(f"  - {output_base}.txt")
    print(f"  - {output_base}.csv")
    print(f"Found {len(labels)} parcels")
    
    # Print sample
    print("\nSample labels:")
    for i, (pid, label) in enumerate(sorted(labels.items())[:5]):
        print(f"  {pid}: {label}")
    if len(labels) > 10:
        print("  ...")
        for pid, label in sorted(labels.items())[-5:]:
            print(f"  {pid}: {label}")
    
    return labels

def main():
    """
    Main function to process parcellation labels.
    
    Processes all configured parcellations or a specific one based on
    command-line arguments. Saves labels in multiple formats for each
    parcellation.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract and process parcellation labels for brain atlases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all parcellations
  python 01_get_parcel_label.py
  
  # Process specific parcellation (matches substring)
  python 01_get_parcel_label.py --parcellation 456
  python 01_get_parcel_label.py --parcellation 1056
  python 01_get_parcel_label.py --parcellation Schaefer
        """
    )
    parser.add_argument('--parcellation', type=str, default='all', 
                       help='Specific parcellation to process or "all" (default: all)')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    
    if not scratch_dir:
        print("ERROR: SCRATCH_DIR environment variable not set")
        print("Please ensure .env file exists with SCRATCH_DIR defined")
        return 1
    
    atlas_dir = os.getenv("ATLAS_DIR")
    if not atlas_dir:
        print("ERROR: ATLAS_DIR environment variable not set")
        print("Please ensure .env file exists with ATLAS_DIR defined")
        return 1

    # Output directory for labels
    output_dir = os.path.join(scratch_dir, "data", "parcellation_labels")

    # atlas-4S parcellations (descending by parcel count) + other atlases
    atlas_names = [
        "atlas-4S1056Parcels", "atlas-4S956Parcels", "atlas-4S856Parcels",
        "atlas-4S756Parcels", "atlas-4S656Parcels", "atlas-4S556Parcels",
        "atlas-4S456Parcels", "atlas-4S356Parcels", "atlas-4S256Parcels",
        "atlas-4S156Parcels", "atlas-HCP", "atlas-Glasser", "atlas-Gordon",
        "atlas-Tian",
    ]

    # Define all parcellation paths
    parcellation_paths = [
        # Original Schaefer parcellation
        os.path.join(scratch_dir, "data", "parcellations",
                     "Schaefer2018_400Parcels_17Networks_order_Tian_Subcortex_S4.dlabel.nii"),
    ] + [
        os.path.join(atlas_dir, name, f"{name}_space-fsLR_den-91k_dseg.dlabel.nii")
        for name in atlas_names
    ]
    
    # Filter parcellations if specific one requested
    if args.parcellation != 'all':
        filtered_paths = []
        for path in parcellation_paths:
            if args.parcellation.lower() in path.lower():
                filtered_paths.append(path)
        
        if not filtered_paths:
            print(f"No parcellation found matching '{args.parcellation}'")
            return
        
        parcellation_paths = filtered_paths
    
    # Process each parcellation
    successful = 0
    failed = 0
    
    for parc_path in parcellation_paths:
        if not os.path.exists(parc_path):
            print(f"WARNING: Parcellation file not found: {parc_path}")
            failed += 1
            continue
        
        result = process_parcellation_labels(parc_path, output_dir)
        if result is not None:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful} parcellation(s)")
    if failed > 0:
        print(f"Failed: {failed} parcellation(s)")
    print(f"Output directory: {output_dir}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    main()
