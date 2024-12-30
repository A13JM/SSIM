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
        '--folders', type=str, nargs='+', default=[f"B{i}" for i in range(1, 11)],
        help='List of folder names to process. Default: B1 B2 ... B10'
    )
    parser.add_argument(
        '--image_names', type=str, nargs='+', default=[",", ".", ";", "pipe", "BREAK"],
        help='List of image names to compare. The first image is used as the reference.'
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

def load_image(image_path):
    """Load an image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def calculate_ssim(ref_image, comp_image):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    ssim_score, _ = ssim(ref_image, comp_image, full=True)
    return ssim_score

def process_folders(base_dir, folders, image_names):
    """Process each folder and compute SSIM scores."""
    results = []
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        ref_image_path = os.path.join(folder_path, f"{image_names[0]}.png")  # Reference image
        
        ref_image = load_image(ref_image_path)
        if ref_image is None:
            print(f"Reference image not found in {folder_path}. Skipping...", file=sys.stderr)
            continue
        
        for img_name in image_names[1:]:
            img_path = os.path.join(folder_path, f"{img_name}.png")
            comp_image = load_image(img_path)
            
            if comp_image is None:
                print(f"Image {img_name}.png not found in {folder_path}. Skipping...", file=sys.stderr)
                continue
            
            if ref_image.shape != comp_image.shape:
                print(f"Image {img_name}.png in {folder_path} has different dimensions. Skipping...", file=sys.stderr)
                continue
            
            ssim_score = calculate_ssim(ref_image, comp_image)
            
            results.append({
                "Folder": folder,
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
        print("No results to display. Ensure the images and paths are correct.", file=sys.stderr)

def generate_heatmap(df_results, degree=50, output_path=None):
    """Generate and display/save a heatmap of SSIM scores."""
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
    
        # Define original labels for syntaxes and batches
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
    
        # Add labels for syntaxes (x-axis) and batches (y-axis)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(original_columns, rotation=45, ha="right", color="white")
        ax.set_yticklabels(original_rows, color="white")
    
        # Title and axis labels
        plt.title("SSIM Heatmap (Syntaxes compared to Comma (,))", color="white")
        plt.xlabel("Compared Image (Syntax)", color="white")
        plt.ylabel("Folder (Batch)", color="white")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"Heatmap saved to {output_path}")
        else:
            plt.show()
    else:
        print("Heatmap data is empty. No valid comparisons found.", file=sys.stderr)

def main():
    """Main function to orchestrate SSIM analysis."""
    args = parse_arguments()
    setup_plotting()
    
    df_results = process_folders(args.base_dir, args.folders, args.image_names)
    
    # Sort the DataFrame by folder names naturally
    sorted_folders = natsorted(df_results['Folder'].unique())
    df_results['Folder'] = pd.Categorical(df_results['Folder'], categories=sorted_folders, ordered=True)
    df_results = df_results.sort_values('Folder')
    
    display_results(df_results)
    
    generate_heatmap(df_results, degree=args.degree, output_path=args.output_heatmap)

if __name__ == "__main__":
    main()
