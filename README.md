# SSIM Analysis Tool

A Python tool to compute Structural Similarity Index (SSIM) between images in specified folders and visualize the results as a heatmap.

## Features

- **Compare Images:** Automatically compares images within specified folders.
- **SSIM Calculation:** Calculates the Structural Similarity Index (SSIM) between reference and compared images.
- **Tabular Results:** Displays SSIM scores in a neatly formatted table.
- **Heatmap Visualization:** Generates a heatmap to visualize SSIM scores across different images and folders.

## Requirements

- Python 3.7+
- The dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ssim-analysis-tool.git
   cd ssim-analysis-tool
   ```
   
2. **Install Requirements**   
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run ssim_analysis.py**   
   ```bash
   python ssim_analysis.py --base_dir "PATH_TO_DATA_FOLDER"
   ```
```bash
   data folder/
│
└── Batches "B#"
   └── Images (Supports "," "." ";" "BREAK" "pipe" "AND" "OR")
