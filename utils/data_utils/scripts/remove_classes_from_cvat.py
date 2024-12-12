"""Remove specified classes from CVAT xml file.

All "label" and "box" tags will be removed and a new xml file will be saved.
"""


import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


def remove_annotations(
    xml_file: Path,
    labels_to_remove: List[str],
    save_path: Optional[Path] = None
):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find all box elements
    boxes = root.findall('.//box')

    # Keep track of removed annotations
    removed_count = {label: 0 for label in labels_to_remove}

    # Remove labels from the "labels" tag
    labels_tag = root.find('.//labels')
    if labels_tag is not None:
        for label in labels_tag.findall('label'):
            if label.find('name').text in labels_to_remove:
                labels_tag.remove(label)

    # Iterate through all images
    for image in tqdm(root.findall('.//image'), desc='Removing annotations'):
        # Find all box elements within the current image
        boxes = image.findall('.//box')

        # Iterate through the boxes in reverse order
        for box in reversed(boxes):
            if box.get('label') in labels_to_remove:
                label = box.get('label')
                image.remove(box)
                removed_count[label] += 1

    # Save the modified XML
    if save_path is None:
        save_path = xml_file.parent / f'{xml_file.stem}_removed.xml'
    tree.write(save_path, encoding='utf-8', xml_declaration=True)

    # Print summary
    print("Annotations removed:")
    for label, count in removed_count.items():
        print(f"  {label}: {count}")


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'cvat_xml', type=Path,
        help='Path to CVAT xml file.')
    parser.add_argument(
        'labels_to_remove', type=str, nargs='+',
        help='Labels to remove.')
    parser.add_argument(
        '--save_pth', type=Path, default=None,
        help='Path to save an edited CVAT xml file. '
             'If not provided, the new file will be save next to original '
             'file with name "annotations_removed.xml".')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    remove_annotations(args.cvat_xml, args.labels_to_remove, args.save_pth)
