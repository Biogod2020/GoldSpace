# spaglam_preproc/config.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class PathConfig:
    """Configuration for all input and output paths."""
    adata_path: str
    output_dir: str
    # Image can be a WSI file, a standard image file, or loaded from adata.
    # If None, the pipeline will attempt to load the image from adata.uns['spatial'].
    image_path: Optional[str] = None
    # Optional path to a pre-computed list of highly variable genes (one gene per line or CSV column).
    hvg_list_path: Optional[str] = None

@dataclass
class PreprocessingConfig:
    """Parameters for data transformation and graph construction."""
    neighborhood_hops: int = 2
    n_top_genes_in_sentence: int = 50
    tile_size: int = 224
    precompute_embeddings: bool = True

@dataclass
class ModelConfig:
    """Model parameters, only required if precompute_embeddings is True."""
    model_path: str = "path/to/your/omiclip_model.pt"
    model_name: str = "ViT-B-32"

@dataclass
class QualityControlConfig:
    """Configuration for quality control, logging, and visualization."""
    enabled: bool = True
    num_visual_samples: int = 16  # Number of samples to include in the visual grid
    log_file_name: str = "preprocessing.log" # Name for the detailed log file

@dataclass
class PerformanceConfig:
    """Parameters to control performance and parallelization."""
    max_workers: int = 32
    max_samples_per_shard: int = 10000
    # Process a subset for quick testing. Set to -1 to process all spots.
    num_spots_to_process: int = -1 

@dataclass
class MainConfig:
    """Root configuration object that nests all other configurations."""
    paths: PathConfig
    preprocessing: PreprocessingConfig
    performance: PerformanceConfig
    qc: QualityControlConfig = field(default_factory=QualityControlConfig)
    # The model config is optional and only needed for one mode.
    model: Optional[ModelConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MainConfig':
        """Creates a MainConfig object from a dictionary, handling nested structures."""
        # This allows for easy loading from a parsed YAML file.
        return cls(
            paths=PathConfig(**config_dict['paths']),
            preprocessing=PreprocessingConfig(**config_dict['preprocessing']),
            performance=PerformanceConfig(**config_dict['performance']),
            qc=QualityControlConfig(**config_dict.get('qc', {})),
            model=ModelConfig(**config_dict['model']) if 'model' in config_dict else None
        )
