# SSIM/SAD Analysis Tool

A Python tool to compute Structural Similarity Index (SSIM) and Sum of Absolute Differences (SAD) between images in specified folders and visualize the results as a heatmap.

## Features

- **Compare Images:** Automatically compares images within specified folders.
- **SSIM and SAD Calculation:** Calculates the Structural Similarity Index (SSIM) or Sum of Absolute Differences (SAD) between reference and compared images.
- **Tabular Results:** Displays SSIM/SAD scores in a neatly formatted table.
- **Heatmap Visualization:** Generates a heatmap to visualize SSIM/SAD scores across different images and folders.

## Requirements

- Python 3.7+
- The dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ssim-analysis-tool.git
   cd SSIM_SAD
   ```
   
2. **Install Requirements**   
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run ssim_sad_analysis.py**   
   ```bash
   python ssim/sad_analysis.py --base_dir "PATH_TO_DATA_FOLDER" --metric "ssim / sad"
   ```
   ```bash
   data folder/
   │
   └── Batches "B#"
      └── Images (Supports "," "." ";" "BREAK" "pipe" "AND" "OR")
   ```
