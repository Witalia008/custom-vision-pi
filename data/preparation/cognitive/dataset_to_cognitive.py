import argparse
import io
import json
import os
import zipfile
from abc import ABC, abstractmethod
from itertools import islice
from typing import Callable, Dict, Generator, List, NoReturn

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Project, Region, Tag
from dotenv import load_dotenv
from PIL import Image

from custom_vision import initialize_custom_vision_project, populate_project_tags, upload_batch

TRAINING_ENDPOINT_ENV_VAR_NAME = "CUSTOM_VISION_TRAINING_ENDPOINT"
TRAINING_KEY_ENV_VAR_NAME = "CUSTOM_VISION_TRAINING_KEY"

CUSTOM_VISION_DETECTION_PROJECT_NAME = "PotDetection"
CUSTOM_VISION_CLASSIFICATION_PROJECT_NAME = "PotClassification"


class AbsCustomVisionClient(ABC):
    def __init__(self):
        self.trainer: CustomVisionTrainingClient
        self.project: Project

        self.training_endpoint: str = os.environ[TRAINING_ENDPOINT_ENV_VAR_NAME]
        self.training_key: str = os.environ[TRAINING_KEY_ENV_VAR_NAME]

    def load_label_defs(self, labels_def_path: str) -> Dict[str, Tag]:
        # Populate and/or Load the tags from the project to attach to the regions/images.
        with open(labels_def_path, "r") as labels_file:
            labels_defs: List[Dict] = json.load(labels_file)["labels"]
            return self.populate_tags(labels_defs)

    @abstractmethod
    def populate_tags(self, labels_defs: List[Dict]) -> Dict[str, Tag]:
        pass

    @abstractmethod
    def get_images_for_upload(
        self, images: List[str], image_reader: Callable[[str], bytes], labels: Dict, tags: Dict[str, Tag]
    ) -> Generator[ImageFileCreateEntry, None, None]:
        pass

    @staticmethod
    def get_image_boxes_with_attributes(image_labels: Dict, allowed_tags: List[str], normalize: bool) -> List[Dict]:
        boxes: List[Dict] = []

        image_width, image_height = image_labels["width"], image_labels["height"]

        for label in image_labels["labels"]:
            label_name: str = label["label"]

            if label_name not in allowed_tags:
                continue

            # Non-box shapes are not supported by the Object Detection.
            if label["type"] != "box":
                continue

            # For now allow the occluded shapes.
            # TODO #2: have some logic to allow some occluded shapes, but forbid others.
            if label["occluded"]:
                pass

            points: Dict = label["points"]
            box_x, box_y, box_width, box_height = points["x"], points["y"], points["width"], points["height"]

            # TODO #1: Skip, if the box is bigger than some percentage of the image (50%, half of the image).

            if normalize:
                # Normalise coordinates to [0, 1] range.
                box_x, box_width = (box_x / image_width, box_width / image_width)
                box_y, box_height = (box_y / image_height, box_height / image_height)
            else:
                # If coordinates are not normalized, then convert them to integer ones (read: pixes on the image).
                box_x, box_y, box_width, box_height = int(box_x), int(box_y), int(box_width), int(box_height)

            boxes.append(
                {
                    "tag": label_name,
                    "left": box_x,
                    "top": box_y,
                    "width": box_width,
                    "height": box_height,
                    "properties": list(label["properties"].values()),
                }
            )

        return boxes

    def upload_dataset(self, dataset_path: str, labels_def_path: str) -> NoReturn:
        tags: Dict[str, Tag] = self.load_label_defs(labels_def_path)

        with zipfile.ZipFile(dataset_path, "r") as dataset_zip:
            labels: Dict = json.loads(dataset_zip.read("labels.json"))

            images: List[str] = list(labels.keys())

            # Get a generator for Image objects for upload, that will lazily produce data for our batches.
            images_for_upload: Generator[ImageFileCreateEntry] = self.get_images_for_upload(
                images, dataset_zip.read, labels, tags
            )

            while True:
                # Get data in batches of 64, since that's maximum for one upload to Cognitive Services.
                image_batch = list(islice(images_for_upload, 64))

                # An empty batch means we exhausted all the data to upload.
                if not image_batch or len(image_batch) == 0:
                    break

                print("Uploading in a batch...")
                uploaded = upload_batch(self.trainer, self.project, image_batch)
                print(f"Batch upload status: {uploaded}")


class ObjectDetectionClient(AbsCustomVisionClient):
    def __init__(self):
        super().__init__()

        self.trainer, self.project = initialize_custom_vision_project(
            self.training_endpoint, self.training_key, CUSTOM_VISION_DETECTION_PROJECT_NAME, "ObjectDetection"
        )

    def populate_tags(self, labels_defs: List[Dict]) -> Dict[str, Tag]:
        desired_tags: List[str] = [label["name"] for label in labels_defs]
        # For now, start only with "pot" tag. TODO #3: in the future, include other tags too.
        desired_tags = ["pot"]

        return populate_project_tags(self.trainer, self.project, desired_tags)

    @staticmethod
    def boxes_to_regions(boxes: List[Dict], tags: Dict[str, Tag]) -> List[Region]:
        return [
            Region(
                tag_id=tags[box["tag"]].id, left=box["left"], top=box["top"], width=box["width"], height=box["height"]
            )
            for box in boxes
        ]

    def get_images_for_upload(
        self, images: List[str], image_reader: Callable[[str], bytes], labels: Dict, tags: Dict[str, Tag]
    ) -> Generator[ImageFileCreateEntry, None, None]:

        for file_name in images:
            # Convert each box on the image to region with a tag for detection.
            boxes = AbsCustomVisionClient.get_image_boxes_with_attributes(
                labels[file_name], list(tags.keys()), normalize=True
            )
            regions = ObjectDetectionClient.boxes_to_regions(boxes, tags)

            # We don't need pictures without any labels.
            if len(regions) > 0:
                yield ImageFileCreateEntry(name=file_name, contents=image_reader(file_name), regions=regions)
                print(f"Yielding image {file_name}.")


class ClassificationClient(AbsCustomVisionClient):
    def __init__(self):
        super().__init__()

        self.trainer, self.project = initialize_custom_vision_project(
            self.training_endpoint, self.training_key, CUSTOM_VISION_CLASSIFICATION_PROJECT_NAME, "Classification"
        )

    def populate_tags(self, labels_defs: List[Dict]):
        # Populate tags from boxes' attributes, as this is for their classification.
        desired_tags: List[str] = []
        for label in labels_defs:
            # For now, start only with "pot" tag. TODO #3: in the future, include other tags too.
            if label["name"] != "pot":
                continue

            for attribute in label["attributes"]:
                desired_tags.extend(attribute["values"])

        return populate_project_tags(self.trainer, self.project, desired_tags)

    @staticmethod
    def box_to_subimage(box: Dict, image: Image) -> bytes:
        # Extract subimage that is bounded by the box.

        # Area is (left, top, right, bottom).
        box_area = (box["left"], box["top"], box["width"] + box["left"], box["height"] + box["top"])
        subimage = image.crop(box_area)

        # Store the cut-out part back to bytes.
        subimageArray = io.BytesIO()
        subimage.save(subimageArray, format="PNG")

        return subimageArray.getvalue()

    def get_images_for_upload(
        self, images: List[str], image_reader: Callable[[str], bytes], labels: Dict, tags: Dict[str, Tag]
    ) -> Generator[ImageFileCreateEntry, None, None]:

        for file_name in images:
            # Tags for classification would be actually the attributes,
            # so pass manually the filter list of hight-level tags.
            boxes = AbsCustomVisionClient.get_image_boxes_with_attributes(labels[file_name], ["pot"], normalize=False)

            # Read image to cut out parts of it later for, possibly, multiple boxes.
            image_contents: bytes = image_reader(file_name)
            image: Image = Image.open(io.BytesIO(image_contents))

            for box_n, box in enumerate(boxes):
                # Don't care for boxes without properties, as those cannot be classified.
                # However, for 'pot' labels this shouldn't happen at all.
                if len(box["properties"]) == 0:
                    continue

                # For each box, cut-out the subimage, and append tags that correspond to that box.
                box_image: bytes = ClassificationClient.box_to_subimage(box, image)
                box_tag_ids: List[str] = [tags[tag_name].id for tag_name in box["properties"]]

                # Create image entry for each box subimage. Also, create a unique name for each of them.
                unique_file_name: str = f"{file_name}_{box_n}"
                yield ImageFileCreateEntry(name=unique_file_name, contents=box_image, tag_ids=box_tag_ids)
                print(f"Yielding box-image {unique_file_name}.")


def main():
    CLASSIFICATION_TYPE = "classification"
    DETECTION_TYPE = "detection"

    parser = argparse.ArgumentParser("Custom Vision Object Recogniser dataset uploader")
    parser.add_argument("--dataset", dest="dataset_path", required=True, help="Path to a .zip dataset folder")
    parser.add_argument(
        "--labels",
        dest="labels_def_path",
        default="../labels_config.json",
        help="Path to a file with labels definitions",
    )
    parser.add_argument(
        "--type",
        dest="project_type",
        required=True,
        help=f"Type of Custom Vision project ({DETECTION_TYPE}, {CLASSIFICATION_TYPE})",
    )

    args = parser.parse_args()

    load_dotenv()

    # Create client either for detection or classification based on the input parameter.
    client_dict = {DETECTION_TYPE: ObjectDetectionClient, CLASSIFICATION_TYPE: ClassificationClient}
    client = client_dict[args.project_type]()

    client.upload_dataset(args.dataset_path, args.labels_def_path)


if __name__ == "__main__":
    main()
