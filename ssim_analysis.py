import os
import cv2
import pandas as pd
from tabulate import tabulate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim
from natsort import natsorted
import argparse
import sys
import glob
import logging

def setup_logging():
    """Configure the logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def setup_plotting():
    """Configure the plotting aesthetics."""
    sns.set_theme(style="darkgrid")
    
    plt.rcParams['figure.facecolor'] = '#121212'  # Dark background for the figure
    plt.rcParams['axes.facecolor'] = '#121212'    # Dark background for the axes
    plt.rcParams['axes.labelcolor'] = 'white'     # White axis labels
    plt.rcParams['xtick.color'] = 'white'         # White x-axis tick labels
    plt.rcParams['ytick.color'] = 'white'         # White y-axis tick labels
    plt.rcParams['text.color'] = 'white'          # White text
    plt.rcParams['savefig.facecolor'] = '#121212' # Background for saved figures

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SSIM Analysis Tool")
    parser.add_argument(
        '--base_dir', type=str, required=True,
        help='Base directory containing the image folders.'
    )
    parser.add_argument(
        '--degree', type=int, default=50,
        help='Degree for interpolation in heatmap. Default: 50'
    )
    parser.add_argument(
        '--output_heatmap', type=str, default=None,
        help='Path to save the heatmap figure. If not provided, the heatmap will be displayed but not saved.'
    )
    return parser.parse_args()

def get_b_folders(base_dir):
    """Retrieve all folders in base_dir that match the pattern B#."""
    pattern = os.path.join(base_dir, "B*")
    all_folders = [f for f in glob.glob(pattern) if os.path.isdir(f)]
    b_folders = [os.path.basename(folder) for folder in all_folders 
                if os.path.basename(folder).upper().startswith('B') and 
                os.path.basename(folder)[1:].isdigit()]
    if not b_folders:
        logging.error(f"No 'B#' folders found in {base_dir}.")
    else:
        logging.info(f"Detected 'B#' folders: {b_folders}")
    return natsorted(b_folders)

def get_image_files(folder_path):
    """Retrieve all image files in a folder, supporting .png, .jpg, .jpeg (case-insensitive)."""
    supported_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in supported_extensions:
        # Use glob with case-insensitive patterns
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    # Remove duplicates if any and exclude hidden files
    image_files = [f for f in set(image_files) if not os.path.basename(f).startswith('.')]
    # Sort the files naturally
    image_files = natsorted([os.path.basename(f) for f in image_files])
    logging.debug(f"Images detected in '{folder_path}': {image_files}")
    return image_files

def find_reference_image(image_files):
    """Find the image named ',' among the list of image files."""
    # Look for image named ',' with any supported extension
    for img in image_files:
        name, ext = os.path.splitext(img)
        if name == ",":
            return img
    return None

def load_image(image_path):
    """Load an image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def calculate_ssim(ref_image, comp_image):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    try:
        ssim_score, _ = ssim(ref_image, comp_image, full=True)
        return ssim_score
    except ValueError as e:
        logging.error(f"SSIM calculation failed: {e}")
        return None

def process_folders(base_dir, folders):
    """Process each folder and compute SSIM scores."""
    results = []
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        image_files = get_image_files(folder_path)
        
        if not image_files:
            logging.warning(f"No images found in {folder_path}. Skipping...")
            continue
        
        logging.info(f"Detected images in '{folder}': {image_files}")
        
        # Find the reference image named ','
        ref_image_name = find_reference_image(image_files)
        if not ref_image_name:
            logging.warning(f"No reference image named ',' found in {folder_path}. Skipping...")
            continue
        ref_image_path = os.path.join(folder_path, ref_image_name)
        ref_image = load_image(ref_image_path)
        
        if ref_image is None:
            logging.warning(f"Reference image '{ref_image_name}' not found or unreadable in {folder_path}. Skipping...")
            continue
        
        logging.info(f"Processing folder '{folder}' with reference image '{ref_image_name}'")
        
        # Compare all other images to the reference
        for img_name in image_files:
            if img_name == ref_image_name:
                continue  # Skip the reference image itself
            
            img_path = os.path.join(folder_path, img_name)
            comp_image = load_image(img_path)
            
            if comp_image is None:
                logging.warning(f"Image '{img_name}' not found or unreadable in {folder_path}. Skipping...")
                continue
            
            if ref_image.shape != comp_image.shape:
                logging.warning(f"Image '{img_name}' in {folder_path} has different dimensions. Skipping...")
                continue
            
            ssim_score = calculate_ssim(ref_image, comp_image)
            if ssim_score is None:
                logging.warning(f"Failed to calculate SSIM for image '{img_name}' in '{folder}'. Skipping...")
                continue
            
            results.append({
                "Folder": folder,
                "Reference Image": ref_image_name,
                "Compared Image": img_name,
                "SSIM": ssim_score
            })
    return pd.DataFrame(results)

def display_results(df_results):
    """Display the SSIM results in a tabular format."""
    if not df_results.empty:
        table = tabulate(df_results, headers='keys', tablefmt='grid', floatfmt=".4f")
        print(table)
    else:
        logging.error("No results to display. Ensure the images and paths are correct.")

def generate_heatmap(df_results, degree=50, output_path=None):
    """Generate and display/save a heatmap of SSIM scores."""
    # Pivot the DataFrame for heatmap visualization
    heatmap_data = df_results.pivot(index="Folder", columns="Compared Image", values="SSIM")
    
    if heatmap_data is not None and not heatmap_data.empty:
        # Interpolate data for smoother gradients
        x = np.linspace(0, heatmap_data.shape[1] - 1, heatmap_data.shape[1] * degree)  # More columns
        y = np.linspace(0, heatmap_data.shape[0] - 1, heatmap_data.shape[0] * degree)  # More rows
        x_grid, y_grid = np.meshgrid(x, y)
    
        # Flatten the original data for interpolation
        points = np.array([[i, j] for i in range(heatmap_data.shape[0]) for j in range(heatmap_data.shape[1])])
        values = heatmap_data.values.flatten()
    
        # Interpolated values
        smooth_data = griddata(points, values, (y_grid, x_grid), method='cubic')
    
        # Define original labels for compared images and folders
        original_columns = heatmap_data.columns
        original_rows = heatmap_data.index
    
        # Set extended tick labels for the interpolated data
        xticks = np.linspace(0, smooth_data.shape[1] - 1, len(original_columns))
        yticks = np.linspace(0, smooth_data.shape[0] - 1, len(original_rows))
    
        # Plot the smoother heatmap with plasma colormap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            smooth_data,
            cmap="plasma",
            cbar_kws={"label": "SSIM Score", "format": "%.2f"},
            xticklabels=False,
            yticklabels=False,
            vmin=0,
            vmax=1
        )
    
        # Add labels for compared images (x-axis) and batches (y-axis)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(original_columns, rotation=45, ha="right", color="white")
        ax.set_yticklabels(original_rows, color="white")
    
        # Title and axis labels
        plt.title("SSIM Heatmap (Compared Images to Reference ',')", color="white")
        plt.xlabel("Compared Image (Syntax)", color="white")
        plt.ylabel("Folder (Batch)", color="white")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300)
            logging.info(f"Heatmap saved to {output_path}")
        else:
            plt.show()
    else:
        logging.error("Heatmap data is empty. No valid comparisons found.")

def main():
    """Main function to orchestrate SSIM analysis."""
    setup_logging()
    args = parse_arguments()
    setup_plotting()
    
    b_folders = get_b_folders(args.base_dir)
    if not b_folders:
        sys.exit(1)
    
    df_results = process_folders(args.base_dir, b_folders)
    
    if df_results.empty:
        logging.error("No SSIM results were generated. Exiting.")
        sys.exit(1)
    
    # Sort the DataFrame by folder names naturally
    sorted_folders = natsorted(df_results['Folder'].unique())
    df_results['Folder'] = pd.Categorical(df_results['Folder'], categories=sorted_folders, ordered=True)
    df_results = df_results.sort_values('Folder')
    
    display_results(df_results)
    
    generate_heatmap(df_results, degree=args.degree, output_path=args.output_heatmap)

if __name__ == "__main__":
    main()
