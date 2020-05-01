import logging
import os
import subprocess
import tempfile
import zipfile
from typing import BinaryIO, List, NoReturn, Union
from urllib.parse import ParseResult, urlparse

import azure.functions as func
from azure.storage.blob import BlobClient, BlobServiceClient


def get_blob_from_uri(blob_service: BlobServiceClient, blob_uri: str) -> (BlobClient, str, str):
    # URI in format https://storagename.blob.core.windows.net/containername/blobfolders/blobfilename.blobextension'.
    blob_url_props: ParseResult = urlparse(blob_uri)

    # In a parsed URL, path = "/containername/blobname".
    # First, skip '/', then split on first occurrence, i.e. into 'containername' and the rest to be 'blobname'.
    blob_container_name, blob_name = blob_url_props.path[1:].split("/", 1)

    input_blob: BlobClient = blob_service.get_blob_client(container=blob_container_name, blob=blob_name)

    return input_blob, blob_container_name, blob_name


def extract_frames(ffmpeg_exe: str, input_file_name: str, frames_folder: str) -> NoReturn:
    frame_format: str = os.path.join(frames_folder, "frame%06d.jpg")

    # Run FFMPEG to extract frames from the video file.
    frame_rate: str = "1/1"
    logging.info(f"Extracting frames using ffmpeg using frame rate {frame_rate}, preserving quality...")
    subprocess.call([ffmpeg_exe, "-i", input_file_name, "-r", frame_rate, "-qscale", "0", frame_format])


def zip_frames(output_file: Union[str, BinaryIO], frames_folder: str) -> NoReturn:
    zipped_frames = zipfile.ZipFile(output_file, mode="w")  # "w" for ZipFile translated to "w+b"
    frame_files: List[str] = os.listdir(frames_folder)

    if len(frame_files) == 0:
        raise Exception("Didn't find any frame files")

    for frame_file in frame_files:
        # Add each of the file in the root of zip archive.
        zipped_frames.write(os.path.join(frames_folder, frame_file), frame_file)
        logging.info(f"Adding file {frame_file} to the archive.")
    logging.info(f"Stored {len(frame_files)} frame files from {frames_folder}.")


async def main(event: func.EventGridEvent, context: func.Context) -> NoReturn:
    blob_uri: str = event.get_json()["url"]
    blob_service: BlobServiceClient = BlobServiceClient.from_connection_string(os.getenv("stovedatastorage_STORAGE"))

    input_blob, input_blob_container_name, input_blob_name = get_blob_from_uri(blob_service, blob_uri)

    logging.info(f"Extracting frames from video clip: {input_blob_container_name}/{input_blob_name}.")

    # Do the processing in a temporary directory.
    input_file: BinaryIO
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".mp4") as input_file:
        # Store the file locally on the disk in a temporary file.
        input_blob.download_blob().readinto(input_file)

        # Create a temporary directory for extracted frames.
        frames_folder: str
        with tempfile.TemporaryDirectory() as frames_folder:
            ffmpeg_exe: str = os.path.join(context.function_directory, "ffmpeg")

            # Extract frames from a video clip into a frames folder.
            extract_frames(ffmpeg_exe, input_file.name, frames_folder)

            output_file: BinaryIO
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=".zip") as output_file:
                # Add all of the frames to a zip archive.
                zip_frames(output_file, frames_folder)

                output_blob_container_name: str = "extractedframes"
                output_blob_name: str = os.path.splitext(input_blob_name)[0] + ".zip"

                output_blob: BlobClient = blob_service.get_blob_client(
                    container=output_blob_container_name, blob=output_blob_name
                )

                # Since we've been writing to output_file, we need to seek to the beginning to be able to read.
                output_file.seek(0)
                output_blob.upload_blob(output_file, overwrite=True)

                logging.info(f"Extracted frames to {output_blob_container_name}/{output_blob_name}.")
