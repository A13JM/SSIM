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
    parser = argparse.ArgumentParser(description="SSIM/SAD Analysis Tool")
    parser.add_argument(
        '--base_dir', type=str, required=True,
        help='Base directory containing the image folders.'
    )
    parser.add_argument(
        '--folders', type=str, nargs='+', default=[f"B{i}" for i in range(1, 11)],
        help='List of folder names to process. Default: B1 B2 ... B10'
    )
    parser.add_argument(
        '--image_names', type=str, nargs='+', default=[",", ".", ";", "pipe", "BREAK", "AND", "OR"],
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
    parser.add_argument(
        '--metric', type=str, choices=['ssim', 'sad'], default='ssim',
        help='Metric to use for comparison: "ssim" (default) or "sad".'
    )
    return parser.parse_args()

def load_image(image_path):
    """Load an image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def calculate_ssim_score(ref_image, comp_image):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    ssim_score, _ = ssim(ref_image, comp_image, full=True)
    return ssim_score

def calculate_sad(ref_image, comp_image):
    """Calculate the Sum of Absolute Differences (SAD) between two images."""
    abs_diff = cv2.absdiff(ref_image, comp_image)
    sad = abs_diff.sum()
    return sad

def process_folders(base_dir, folders, image_names, metric):
    """Process each folder and compute SSIM or SAD scores."""
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
            
            if metric == 'ssim':
                score = calculate_ssim_score(ref_image, comp_image)
                score_label = 'SSIM'
            else:
                score = calculate_sad(ref_image, comp_image)
                score_label = 'SAD'
            
            results.append({
                "Folder": folder,
                "Compared Image": img_name,
                score_label: score
            })
    return pd.DataFrame(results)

def display_results(df_results, metric):
    """Display the SSIM/SAD results in a tabular format."""
    if not df_results.empty:
        table = tabulate(df_results, headers='keys', tablefmt='grid', floatfmt=".4f" if metric == 'ssim' else ".0f")
        print(table)
    else:
        print("No results to display. Ensure the images and paths are correct.", file=sys.stderr)

def generate_heatmap(df_results, degree=50, output_path=None, metric='ssim'):
    """Generate and display/save a heatmap of SSIM/SAD scores."""
    if metric == 'ssim':
        pivot_column = "SSIM"
        title_metric = "SSIM Score"
    else:
        pivot_column = "SAD"
        title_metric = "SAD Value"
    
    heatmap_data = df_results.pivot(index="Folder", columns="Compared Image", values=pivot_column)
    
    if heatmap_data is not None and not heatmap_data.empty:
        # Interpolate data for smoother gradients
        x = np.linspace(0, heatmap_data.shape[1] - 1, heatmap_data.shape[1] * degree)  # More columns
        y = np.linspace(0, heatmap_data.shape[0] - 1, heatmap_data.shape[0] * degree)  # More rows
        x_grid, y_grid = np.meshgrid(x, y)
    
        # Flatten the original data for interpolation
        points = np.array([[i, j] for i in range(heatmap_data.shape[0]) for j in range(heatmap_data.shape[1])])
        values = heatmap_data.values.flatten()
    
        # Handle cases where all values might be the same
        if np.all(values == values[0]):
            smooth_data = np.tile(values, (y_grid.shape[0], x_grid.shape[1]))
        else:
            # Interpolated values
            smooth_data = griddata(points, values, (y_grid, x_grid), method='cubic')
    
        # Define original labels for syntaxes and batches
        original_columns = heatmap_data.columns
        original_rows = heatmap_data.index
    
        # Set extended tick labels for the interpolated data
        xticks = np.linspace(0, smooth_data.shape[1] - 1, len(original_columns))
        yticks = np.linspace(0, smooth_data.shape[0] - 1, len(original_rows))
    
        # Plot the smoother heatmap with appropriate colormap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            smooth_data,
            cmap="plasma",
            cbar_kws={"label": title_metric, "format": "%.4f" if metric == 'ssim' else "%.0f"},
            xticklabels=False,
            yticklabels=False,
            vmin=0,
            vmax=1 if metric == 'ssim' else None  # For SSIM, set max to 1
        )
    
        # Add labels for syntaxes (x-axis) and batches (y-axis)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(original_columns, rotation=45, ha="right", color="white")
        ax.set_yticklabels(original_rows, color="white")
    
        # Title and axis labels
        plt.title(f"{metric.upper()} Heatmap (Syntaxes compared to {df_results['Compared Image'].iloc[0]})", color="white")
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
    """Main function to orchestrate SSIM/SAD analysis."""
    args = parse_arguments()
    setup_plotting()
    
    df_results = process_folders(args.base_dir, args.folders, args.image_names, args.metric)
    
    # Sort the DataFrame by folder names naturally
    sorted_folders = natsorted(df_results['Folder'].unique())
    df_results['Folder'] = pd.Categorical(df_results['Folder'], categories=sorted_folders, ordered=True)
    df_results = df_results.sort_values('Folder')
    
    display_results(df_results, args.metric)
    
    generate_heatmap(df_results, degree=args.degree, output_path=args.output_heatmap, metric=args.metric)

if __name__ == "__main__":
    main()
