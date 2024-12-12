"""Classify and organize satellite images using YOLO classification model.

Classification has three classes: cloud, blur and normal.
"""


import argparse
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import sys

from ultralytics import YOLO
import torch
from tqdm import tqdm
from loguru import logger

sys.path.append(str(Path(__file__).parents[1]))
from utils.data_utils.functions import collect_paths, IMAGE_EXTENSIONS


def setup_output_dirs(
    output_dir: Path,
    class_names: List[str],
    clear_existing: bool = False,
    include_masks: bool = False
) -> Dict[str, Path]:
    """Create directories for each class.
    
    Parameters
    ----------
    output_dir : Path
        Base output directory.
    class_names : List[str]
        List of class names.
    clear_existing : bool
        Whether to clear existing directories.
    include_masks : bool
        Whether to include mask directories.
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping class names to their directories.
    """
    if clear_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    class_dirs = {}
    
    for class_name in class_names:
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        if include_masks:
            mask_dir = class_dir / "masks"
            mask_dir.mkdir(exist_ok=True)
            class_dir /= 'images'
            class_dir.mkdir(exist_ok=True)
        class_dirs[class_name] = class_dir
    
    return class_dirs


def classify_images(
    model: YOLO,
    image_paths: List[Path],
    class_dirs: Dict[str, Path],
    mask_paths: Optional[List[Path]] = None,
    copy_files: bool = True,
    verbose: bool = False
) -> Dict[str, int]:
    """Classify images and organize them into class directories.
    
    Parameters
    ----------
    model : YOLO
        Trained YOLO classification model.
    image_paths : List[Path]
        List of image paths to classify.
    class_dirs : Dict[str, Path]
        Dictionary mapping class names to directories.
    mask_paths : Optional[List[Path]]
        Optional list of mask paths to match with images.
    copy_files : bool
        Whether to copy files instead of moving them.
    verbose : bool
        Whether to enable verbose logging.
        
    Returns
    -------
    Dict[str, int]
        Count of images per class.
    """
    class_counts = {name: 0 for name in class_dirs.keys()}
    unclassified_count = 0
    
    for i, img_path in enumerate(tqdm(
        image_paths,
        desc="Classifying images",
        disable=not verbose
    )):
        # Get prediction
        results = model.predict(img_path, verbose=False)
        
        if len(results) > 0 and len(results[0].probs) > 0:
            # Get predicted class
            pred_cls = results[0].names[results[0].probs.top1]
            
            # Copy/move file to appropriate directory
            dest_path = class_dirs[pred_cls] / img_path.name
            if mask_paths:
                mask_dest_path = (
                    class_dirs[pred_cls].parent / "masks" / mask_paths[i].name
                )
            
            if copy_files:
                shutil.copy2(img_path, dest_path)
                if mask_paths:
                    shutil.copy2(mask_paths[i], mask_dest_path)
            else:
                shutil.move(img_path, dest_path)
                if mask_paths:
                    shutil.move(mask_paths[i], mask_dest_path)
                
            class_counts[pred_cls] += 1
        else:
            unclassified_count += 1
            logger.warning(f"Failed to classify {img_path.name}")
    
    if unclassified_count > 0:
        logger.warning(
            f"Failed to classify {unclassified_count} images"
        )
    
    return class_counts


def main(
    image_dir: Path,
    model_path: Path,
    output_dir: Path,
    mask_dir: Path,
    copy_files: bool = True,
    clear_existing: bool = False,
    verbose: bool = False
) -> None:
    """Main function to classify and organize images.
    
    Parameters
    ----------
    image_dir : Path
        Directory containing images to classify.
    model_path : Path
        Path to trained YOLO classification model.
    output_dir : Path
        Base directory for organized images.
    mask_dir : Path
        Directory containing matched masks images.
    copy_files : bool
        Whether to copy files instead of moving them.
    clear_existing : bool
        Whether to clear existing output directories.
    verbose : bool
        Whether to enable verbose logging.
    """
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Get class names from model
    dummy_result = model.predict(torch.rand(1, 3, 640, 640), verbose=verbose)
    class_names = list(dummy_result[0].names.values())
    
    # Setup output directories
    class_dirs = setup_output_dirs(
        output_dir,
        class_names,
        clear_existing,
        include_masks=mask_dir is not None
    )
    
    # Collect image paths
    image_paths = sorted(collect_paths(image_dir, IMAGE_EXTENSIONS))
    if mask_dir:
        mask_paths = sorted(collect_paths(mask_dir, IMAGE_EXTENSIONS))
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                "Number of images and masks do not match"
            )
    logger.info(f"Found {len(image_paths)} images to classify")
    
    # Classify images
    class_counts = classify_images(
        model,
        image_paths,
        class_dirs,
        mask_paths,
        copy_files,
        verbose
    )
    
    # Log results
    logger.info("Classification complete. Results:")
    for cls_name, count in class_counts.items():
        logger.info(f"{cls_name}: {count} images")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'image_dir',
        type=Path,
        help='Directory containing images to classify.'
    )
    parser.add_argument(
        'model_path',
        type=Path,
        help='Path to trained YOLO classification model.'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Base directory for organized images'
    )
    parser.add_argument(
        '--mask_dir',
        type=Path,
        help='Directory containing matched masks images.'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them'
    )
    parser.add_argument(
        '--clear_existing',
        action='store_true',
        help='Clear existing output directories'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if args.mask_dir and not args.mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {args.mask_dir}")
    
    return args


if __name__ == '__main__':
    args = parse_args()
    main(
        image_dir=args.image_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        copy_files=args.copy,
        clear_existing=args.clear_existing,
        verbose=args.verbose
    )
