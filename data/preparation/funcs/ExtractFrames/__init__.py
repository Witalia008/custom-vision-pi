import logging

import azure.functions as func

import tempfile
import os
import subprocess
import zipfile


def main(blobin: func.InputStream, blobout: func.Out[func.InputStream], context: func.Context):
    logging.info(f"Extracting frames from video clip: {blobin.name}.")

    # Do the processing in a temporary directory.
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".mp4") as input_file:
        # Store the file locally on the disk in a temporary file.
        input_file.write(blobin.read())

        # Create a temporary directory for extracted frames.
        with tempfile.TemporaryDirectory() as frames_folder:
            frame_format = os.path.join(frames_folder, "frame%06d.jpg")
            ffmpeg_exe = os.path.join(context.function_directory, "ffmpeg")

            # Run FFMPEG to extract frames from the video file.
            frame_rate = "1/1"
            logging.info(f"Extracting frames using ffmpeg using frame rate {frame_rate}...")
            subprocess.call([ffmpeg_exe, "-i", input_file.name, "-r", frame_rate, frame_format])

            _, output_file = tempfile.mkstemp(suffix=".zip")

            # Add all of the frames to a zip folder.
            with zipfile.ZipFile(output_file, mode="w") as zipped_frames:
                frame_files = os.listdir(frames_folder)
                for frame_file in frame_files:
                    # Add each of the file in the root of zip archive.
                    zipped_frames.write(os.path.join(frames_folder, frame_file), frame_file)
                    logging.debug(f"Adding file {frame_file} to the archive.")
                logging.info(f"Stored {len(frame_files)} frame files into {output_file}.")

            # Open the zip file as a stream and pass out to be uploaded to blob storage.
            blobout.set(open(output_file, "rb"))
            logging.info(f"Extracted frames from {blobin.name}.")
