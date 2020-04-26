import argparse
import json
import zipfile
import os
from typing import Dict, NoReturn, List
from dotenv import load_dotenv

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateEntry,
    ImageCreateSummary,
    Region,
    Domain,
    Project,
    Tag,
)


TRAINING_ENDPOINT_ENV_VAR_NAME = "CUSTOM_VISION_TRAINING_ENDPOINT"
TRAINING_KEY_ENV_VAR_NAME = "CUSTOM_VISION_TRAINING_KEY"

CUSTOM_VISION_PROJECT_NAME = "PotDetection"


def get_batches(object_list: List[object], max_size: int) -> List[List[object]]:
    batch_obj_list: List[List[object]] = []

    for start in range(0, len(object_list), max_size):
        batch_obj_list.append(object_list[start : start + max_size])

    return batch_obj_list


def initialize_custom_vision_object_detection_project() -> (CustomVisionTrainingClient, Project):
    training_endpoint = os.environ[TRAINING_ENDPOINT_ENV_VAR_NAME]
    training_key = os.environ[TRAINING_KEY_ENV_VAR_NAME]

    trainer = CustomVisionTrainingClient(training_key, endpoint=training_endpoint)

    try:
        # See if the project has already been created.
        project = next(
            project_candidate
            for project_candidate in trainer.get_projects()
            if project_candidate.name == CUSTOM_VISION_PROJECT_NAME
        )
    except StopIteration:
        # Create a new project since existing was not found.

        # Find the object detection domain.
        obj_detection_domain: Domain = next(
            domain for domain in trainer.get_domains() if domain.type == "ObjectDetection" and domain.name == "General"
        )

        project = trainer.create_project(CUSTOM_VISION_PROJECT_NAME, domain_id=obj_detection_domain.id)

    return trainer, project


def populate_tags(trainer: CustomVisionTrainingClient, project: Project, labels_defs: List[Dict]) -> Dict[str, Tag]:
    tags: Dict[str, Tag] = {}

    for tag in trainer.get_tags(project.id):
        tags[tag.name] = tag

    desired_tags: List[str] = [label["name"] for label in labels_defs]
    # For now, start only with "pot" tag. TODO: in the future, include other tags too.
    desired_tags = ["pot"]

    for label in desired_tags:
        if label not in tags:
            tags[label] = trainer.create_tag(project.id, label)

    return tags


def get_image_regions(image_labels: Dict, tags: List[Tag]) -> List[Region]:
    regions: List[Region] = []

    image_width, image_height = image_labels["width"], image_labels["height"]

    for label in image_labels["labels"]:
        label_name: str = label["label"]

        # Non-box shapes are not supported by the Object Detection.
        if label["type"] != "box":
            continue

        # For now allow the occluded shapes. TODO: have some logic to allow some occluded shapes, but forbid others.
        if label["occluded"]:
            pass

        box_x, box_y, box_width, box_height = (
            label["points"]["x"],
            label["points"]["y"],
            label["points"]["width"],
            label["points"]["height"],
        )

        # Normalise coordinates to [0, 1] range.
        box_x, box_y, box_width, box_height = (
            box_x / image_width,
            box_y / image_height,
            box_width / image_width,
            box_height / image_height,
        )

        regions.append(Region(tag_id=tags[label_name].id, left=box_x, top=box_y, width=box_width, height=box_height))

    return regions


def upload_batch(
    image_file_names: List[str],
    image_zip_file: zipfile.ZipFile,
    labels: Dict,
    trainer: CustomVisionTrainingClient,
    project: Project,
    tags: Dict[str, Tag],
) -> bool:
    tagged_images_with_regions: List[ImageFileCreateEntry] = []

    print("Adding a batch of images...")

    for file_name in image_file_names:
        regions = get_image_regions(labels[file_name], tags)

        # We don't need pictures without any labels.
        if len(regions) > 0:
            tagged_images_with_regions.append(
                ImageFileCreateEntry(name=file_name, contents=image_zip_file.read(file_name), regions=regions)
            )

    # If the batch didn't end up having any images, trying to upload an empty list would throw.
    if len(tagged_images_with_regions) > 0:
        upload_result: ImageCreateSummary = trainer.create_images_from_files(project.id, tagged_images_with_regions)

        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print(f"Image status: {image.source_url} - {image.status}")
            return False

    return True


def upload_dataset(dataset_path: str, labels_def_path: str) -> NoReturn:
    # Get the trainer and the project to upload images to.
    trainer, project = initialize_custom_vision_object_detection_project()

    # Populate and/or Load the tags from the project to attach to the regions.
    with open(labels_def_path, "r") as labels_file:
        labels_defs: List[Dict] = json.load(labels_file)["labels"]
        tags = populate_tags(trainer, project, labels_defs)
        print(f"Loaded tags: {list(tags.keys())}")

    with zipfile.ZipFile(dataset_path, "r") as dataset_zip:
        labels: Dict = json.loads(dataset_zip.read("labels.json"))

        images: List[str] = list(labels.keys())

        # Split into batches of maximum 64 since that's the maximum that Cognitive Services can accept in one upload.
        image_batches: List[List[str]] = get_batches(images, 64)

        for image_batch in image_batches:
            uploaded = upload_batch(image_batch, dataset_zip, labels, trainer, project, tags)
            print(f"Batch upload status: {uploaded}")


def main():
    parser = argparse.ArgumentParser("Custom Vision Object Recogniser dataset uploader")
    parser.add_argument("--dataset", dest="dataset_path", required=True, help="Path to a .zip dataset folder")
    parser.add_argument(
        "--labels",
        dest="labels_def_path",
        default="../labels_config.json",
        help="Path to a file with labels definitions",
    )

    args = parser.parse_args()

    load_dotenv()

    upload_dataset(args.dataset_path, args.labels_def_path)


if __name__ == "__main__":
    main()
