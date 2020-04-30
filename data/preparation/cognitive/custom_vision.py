from typing import Dict, List

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import (
    Domain,
    ImageCreateSummary,
    ImageFileCreateEntry,
    Project,
    Tag,
)


def initialize_custom_vision_project(
    training_endpoint: str, training_key: str, project_name: str, domain_type: str
) -> (CustomVisionTrainingClient, Project):

    trainer = CustomVisionTrainingClient(training_key, endpoint=training_endpoint)

    try:
        # See if the project has already been created.
        project = next(
            project_candidate for project_candidate in trainer.get_projects() if project_candidate.name == project_name
        )
    except StopIteration:
        # Create a new project since existing was not found.

        # Find the object detection domain.
        obj_detection_domain: Domain = next(
            domain for domain in trainer.get_domains() if domain.type == domain_type and domain.name == "General"
        )

        project = trainer.create_project(project_name, domain_id=obj_detection_domain.id)

    return trainer, project


def populate_project_tags(
    trainer: CustomVisionTrainingClient, project: Project, desired_tags: List[str]
) -> Dict[str, Tag]:

    tags: Dict[str, Tag] = {}

    for tag in trainer.get_tags(project.id):
        tags[tag.name] = tag

    # Presume that any tag called '?' is a negative case.
    for label in desired_tags:
        if label not in tags:
            tags[label] = trainer.create_tag(project.id, label, type=("Negative" if label == "?" else "Regular"))

    print(f"Loaded tags: {list(tags.keys())}")

    return tags


def upload_batch(
    trainer: CustomVisionTrainingClient, project: Project, tagged_images: List[ImageFileCreateEntry]
) -> bool:
    # If the batch didn't end up having any images, trying to upload an empty list would throw.
    if len(tagged_images) > 0:
        upload_result: ImageCreateSummary = trainer.create_images_from_files(project.id, tagged_images)

        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print(f"Image status: {image.source_url} - {image.status}")
            return False

    return True
