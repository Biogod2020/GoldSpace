# GoldSpace Project

This repository contains the source code for the SpaGLaM (Spatial Graph Large Model) project, including the training framework based on `open_clip` and a new, high-performance preprocessing pipeline.

## `spaglam-preproc`: The Preprocessing Pipeline

A high-performance, single-pass data preprocessing pipeline designed for SpaGLaM. This tool efficiently converts spatial transcriptomics data (AnnData and histology images) into graph-based `webdataset` shards suitable for large-scale model training.

### Features

-   **High-Performance Single Pass**: Extracts image tiles and generates gene sentences on-the-fly, eliminating the I/O bottleneck of writing and reading millions of intermediate files.
-   **Flexible Output**: Generate `webdataset` shards containing either raw data (`.png`, `.txt`) or pre-computed OmiCLIP embeddings (`.pth`), controlled by a simple config flag.
-   **Versatile Image Support**: Natively handles Whole-Slide Images (e.g., `.svs`, `.tif`), standard images (`.png`, `.jpeg`), and images embedded in `AnnData` objects.
-   **Robust Quality Control**: Includes pre-run validation checks, live progress monitoring, and automatically generates a final QC report and a visual sample grid for easy verification.
-   **User-Friendly Interface**: A simple Command-Line Interface (CLI) driven by a clean YAML configuration file.
-   **Notebook-Ready**: The core pipeline is encapsulated in a class, allowing for easy, interactive use and visualization within Jupyter notebooks.

### Installation

It is recommended to install the project in editable mode. From the `GoldSpace` root directory:

**1. Basic Installation (for training with existing data):**

Install the base dependencies for training.
```bash
pip install -e .
# You may also need to install from your requirements files
pip install -r requirements.txt
