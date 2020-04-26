import argparse
import json
import os
import random
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import IO, Dict, List, NoReturn, Set


def get_single_clip_labels(clip_name: str, label_path: str) -> Dict[str, object]:
    root: ET.Element = ET.parse(label_path).getroot()

    labels_all = {}

    for image in root.findall("image"):
        labels_image: List = []

        for shape in image:
            current_label: Dict = {
                "label": shape.attrib["label"].strip(),
                "type": shape.tag.strip(),
                "occluded": int(shape.attrib["occluded"]) == 1,
                "points": None,
                "properties": {},
            }

            if shape.tag == "box":
                current_label["points"] = {
                    "x": float(shape.attrib["xtl"]),
                    "y": float(shape.attrib["ytl"]),
                    "width": float(shape.attrib["xbr"]) - float(shape.attrib["xtl"]),
                    "height": float(shape.attrib["ybr"]) - float(shape.attrib["ytl"]),
                }

            for prop in shape.findall("attribute"):
                current_label["properties"][prop.attrib["name"]] = prop.text.strip()

            assert current_label["points"] is not None

            labels_image.append(current_label)

        frame_name: str = image.attrib["name"].strip()

        # Do not include frames without any labels on them.
        if len(labels_image) > 0:
            labels_all[f"{clip_name}_{frame_name}"] = {
                "clip": clip_name,
                "frame": frame_name,
                "width": int(image.attrib["width"]),
                "height": int(image.attrib["height"]),
                "labels": labels_image,
            }
        else:
            print(f"Skipped: {clip_name} - {frame_name}")

    return labels_all


def filter_labels(labels: Dict[str, object], max_clip_frames: int) -> Dict[str, object]:
    if max_clip_frames <= 0:
        return labels

    frames: Set[str] = set(labels.keys())

    if len(frames) <= max_clip_frames:
        return labels

    filtered_out_frames: Set[str] = frames - set(random.sample(frames, max_clip_frames))

    for frame in filtered_out_frames:
        del labels[frame]

    return labels


def copy_labelled_frames(archive_from: str, archive_to: str, labels: Dict[str, object]) -> NoReturn:
    with tempfile.TemporaryDirectory() as frames_from:
        with zipfile.ZipFile(archive_from, "r") as zip_from:
            zip_from.extractall(frames_from)

        with zipfile.ZipFile(archive_to, "a") as zip_to:
            for result_frame in labels.keys():
                frame_path = os.path.join(frames_from, labels[result_frame]["frame"])
                zip_to.write(frame_path, arcname=result_frame)


def store_labels(archive_to: str, labels: Dict[str, object]) -> NoReturn:
    labels_file: IO
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as labels_file:
        json.dump(labels, labels_file, indent=4)

        # Since the file is not yet being closed, there might be some buffered output data.
        # Flush that data, so full content would be picked up and copied to the zip file.
        labels_file.flush()

        with zipfile.ZipFile(archive_to, "a") as zip_to:
            zip_to.write(labels_file.name, "labels.json")


def process_labels_batch(clips_folder: str, labels_folder: str, output_folder: str, max_clip_frames: int) -> NoReturn:
    clip_batch_labels: Dict[str, object] = {}

    os.makedirs(output_folder, exist_ok=True)
    current_time: str = time.strftime("%Y%m%d-%H%M%S")
    filtering_suffix: str = "" if max_clip_frames <= 0 else f"-filtered-{max_clip_frames}"
    output_archive: str = os.path.join(output_folder, f"dataset-{current_time}{filtering_suffix}.zip")

    for clip_file_name in os.listdir(clips_folder):
        print(f"Processing {clip_file_name}...")
        clip_name: str = os.path.splitext(clip_file_name)[0]

        clip_path: str = os.path.join(clips_folder, clip_file_name)
        label_path: str = os.path.join(labels_folder, f"{clip_name}.xml")

        current_clip_labels: Dict[str, object] = get_single_clip_labels(clip_name, label_path)

        # Filter out some frames, if the parameter was passed,
        # in an effort to reduce a lot of similar frames for one clip.
        current_clip_labels_filtered = filter_labels(current_clip_labels, max_clip_frames)

        copy_labelled_frames(clip_path, output_archive, current_clip_labels_filtered)

        clip_batch_labels.update(current_clip_labels)

    store_labels(output_archive, clip_batch_labels)


def main() -> NoReturn:
    parser = argparse.ArgumentParser("Process clips+labels into dataset")
    parser.add_argument("--clips", "-c", dest="clips_folder", required=True, help="Folder with .zip frame archives")
    parser.add_argument("--labels", "-l", dest="labels_folder", required=True, help="Folder with .xml CVAT label files")
    parser.add_argument(
        "--output", "-o", dest="output_folder", required=False, default="output", help="Output folder location"
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        dest="max_clip_frames",
        required=False,
        default=-1,
        type=int,
        help="Maximum number of frames per clip (if one has more, frames will be sampled)",
    )

    args = parser.parse_args()

    process_labels_batch(args.clips_folder, args.labels_folder, args.output_folder, args.max_clip_frames)


if __name__ == "__main__":
    main()
