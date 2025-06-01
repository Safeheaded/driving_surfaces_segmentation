from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import json

def handle_robflow_dataset(paths: list[Path]):
    for path in paths:
        coco_path = path / '_annotations.coco.json'
        coco = COCO(coco_path)
        img_dir = path
        
        category_ids = sorted([
            cat_id for cat_id in coco.getCatIds()
            if coco.cats[cat_id]['name'] != 'drivable-surfaces'
        ])

        # Utwórz mapowanie: category_id -> class_idx (od 1) i category_id -> name
        category_id_to_class_idx = {cat_id: idx + 1 for idx, cat_id in enumerate(category_ids)}
        category_id_to_name = {cat_id: coco.cats[cat_id]['name'].replace(" ", "_") for cat_id in category_ids}

        # Mapowanie: nazwa klasy -> indeks klasy
        class_mapping = {
            category_id_to_name[cat_id]: class_idx
            for cat_id, class_idx in category_id_to_class_idx.items()
        }

        # Zapisz mapping do classes.json
        with open(path / 'classes.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)

        for image_id in coco.imgs:
            img = coco.imgs[image_id]
            image_path = os.path.join(img_dir, img['file_name'])
            original_image = Image.open(image_path)
            width, height = original_image.size

            # Zapisz oryginalny obraz
            images_dir = path / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)
            original_image.save(images_dir / f"{image_id}.png")

            # Pusta maska (0 = unlabelled)
            mask = np.zeros((height, width), dtype=np.uint8)

            # Dodaj adnotacje
            anns_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(anns_ids)

            for ann in anns:
                category_id = ann['category_id']
                class_idx = category_id_to_class_idx[category_id]
                ann_mask = coco.annToMask(ann)

                # Tylko nadpisuj, jeśli maska jeszcze nie zawiera innej klasy
                mask = np.where((ann_mask == 1) & (mask == 0), class_idx, mask)

            # Zapisz maskę wieloklasową
            labels_dir = path / 'labels'
            labels_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(labels_dir / f"{image_id}.png")

            # Usuń oryginalny obraz źródłowy
            os.remove(image_path)

        # Usuń plik COCO
        os.remove(coco_path)


def handle_google_drive_files(dataset_path: Path, source_folder_name: str = '10cm'):
    folder = 'train'
    images_dir = dataset_path / folder / 'images'
    labels_dir = dataset_path / folder / 'labels'

    source_folder =Path(os.getcwd()) / 'datasets' / 'pan_geodeta' / source_folder_name

    images = source_folder.glob('tile_img*.png')
    labels = source_folder.glob('tile_mask*.png')

    for image, label in zip(images, labels):
        image_name = image.name
        label_name = label.name
        image_path = images_dir / image_name
        label_path = labels_dir / label_name

        image_path.write_bytes(image.read_bytes())
        label_path.write_bytes(label.read_bytes())


