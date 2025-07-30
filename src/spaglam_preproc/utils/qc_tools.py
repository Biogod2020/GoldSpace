# spaglam_preproc/utils/qc_tools.py

import json
import logging
import math
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np

# For notebook display
try:
    from IPython.display import display
except ImportError:
    display = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def generate_summary_report(metrics: dict, output_dir: str):
    """Saves a JSON summary of the preprocessing run."""
    report_path = Path(output_dir) / "qc_summary.json"
    try:
        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=convert_numpy)
        logging.info(f"📊 QC summary report saved to: {report_path}")
    except Exception as e:
        logging.error(f"Failed to save QC summary report: {e}")

def generate_visual_artifact(samples: list, output_dir: str, num_samples: int):
    """Creates and saves a grid image of sample tiles and their gene sentences."""
    if not samples:
        logging.warning("No samples were collected, skipping visual QC artifact generation.")
        return

    num_to_display = min(num_samples, len(samples))
    if num_to_display == 0:
        return
        
    grid_size = math.ceil(math.sqrt(num_to_display))
    
    first_tile = samples[0]['tile']
    tile_w, tile_h = first_tile.size
    
    cell_w, cell_h = tile_w + 20, tile_h + 90  # Extra space for padding and text
    grid_img = Image.new("RGB", (grid_size * cell_w, grid_size * cell_h), "#F0F0F0")
    
    try:
        # Try a common system font, fallback to default
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 11)
    except IOError:
        font = ImageFont.load_default()
        font_bold = font
    
    draw = ImageDraw.Draw(grid_img)

    for i, sample in enumerate(samples[:num_to_display]):
        row, col = divmod(i, grid_size)
        x_offset, y_offset = col * cell_w, row * cell_h

        # Paste tile with a small border
        grid_img.paste(sample['tile'], (x_offset + 10, y_offset + 10))
        
        # Draw spot ID
        draw.text(
            (x_offset + 10, y_offset + tile_h + 20),
            f"Spot ID: {sample['id']}", fill="black", font=font_bold)
        
        # Draw wrapped gene sentence
        wrapped_text = textwrap.fill(f"Top Genes: {sample['sentence']}", width=45)
        draw.multiline_text(
            (x_offset + 10, y_offset + tile_h + 35),
            wrapped_text, fill="#555555", font=font)

    artifact_path = Path(output_dir) / "qc_sample_grid.png"
    try:
        grid_img.save(artifact_path)
        logging.info(f"🖼️ QC visual artifact with {num_to_display} samples saved to: {artifact_path}")
    except Exception as e:
        logging.error(f"Failed to save QC visual artifact: {e}")

def display_visual_artifact_notebook(artifact_path: str):
    """Displays the visual artifact image directly in a Jupyter notebook."""
    if not (plt and display):
        logging.warning("Matplotlib or IPython not found. Cannot display image in this environment.")
        return
        
    try:
        img = Image.open(artifact_path)
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Visual QC Samples from {Path(artifact_path).name}")
        plt.show()
    except FileNotFoundError:
        logging.error(f"Artifact file not found at {artifact_path}. Cannot display.")
    except Exception as e:
        logging.error(f"Error displaying visual artifact: {e}")
