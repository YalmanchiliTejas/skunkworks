import os
import json
import shutil
import torch
from PIL import Image
import subprocess
import boto3
from dagshub import get_repo_bucket_client
import requests

# Configuration Constants
BASE_MESH_DIR = "/home/tyalaman/skunkworks/deformations/base_meshes"
DAGSHUB_USER = "YalmanchiliTejas"
DAGSHUB_REPO = "skunkworks"
BATCH_SIZE = 10
TEMP_DIR = "/home/tyalaman/skunkworks/deformations/temp"
LOG = "/home/tyalaman/skunkworks/deformations/logs/pre_process.log"

# Set environment variables for DagsHub authentication
os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["DAGSHUB_CLIENT_HOST"] = "https://dagshub.com"

# Initialize DagsHub S3 Client
s3_client = get_repo_bucket_client(f"{DAGSHUB_USER}/{DAGSHUB_REPO}", flavor="boto")
bucket_name = DAGSHUB_REPO

def write_checkpoint(last_processed):
    """Writes the last processed image path to a log file."""
    with open(LOG, 'w') as f:
        f.write(last_processed)

def get_last_processed():
    """Reads the last processed image path from the log file."""
    try:
        with open(LOG, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def download_file_from_s3(s3_key, local_path):
    """Download a file from S3 to a local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(Bucket=bucket_name, Key=s3_key, Filename=local_path)

def upload_file_to_s3(local_path, s3_key):
    """Upload a local file to S3."""
    s3_client.upload_file(Filename=local_path, Bucket=bucket_name, Key=s3_key)

def process_batch(image_batch, temp_dir):
    """
    Processes a batch of images:
    1. Downloads each image from DagsHub S3 Storage.
    2. Runs the processing script.
    3. Uploads processed results back to S3.
    """
    for cloth in image_batch:
        image_name = os.path.splitext(os.path.basename(cloth['Key']))[0]
        source_image_path = os.path.join(temp_dir, cloth['Key'])
        output_path = os.path.join(temp_dir, "output", image_name)
        os.makedirs(output_path, exist_ok=True)

        try:
            # Download image from S3 Storage
            download_file_from_s3(cloth['Key'], source_image_path)

            # Skip if the image has already been processed
            existing_files = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f"deformations/prepared_dataset/{image_name}")
            if 'Contents' in existing_files:
                print(f"Skipping {cloth['Key']} - already processed")
                continue

            # Run processing script
            process = [
                "python", "run.py",
                "./configs/instant-mesh-large.yaml",
                source_image_path,
                "--output_path", output_path,
                "--save_video",
                "--export_texmap"
            ]

            result = subprocess.run(process, capture_output=True, text=True)
            print(f"Processed {cloth['Key']}: Return code {result.returncode}")

            if result.returncode == 0:
                # Upload processed files back to S3
                for output_file in os.listdir(output_path):
                    local_file = os.path.join(output_path, output_file)
                    s3_key = f"deformations/prepared_dataset/{image_name}/{output_file}"
                    upload_file_to_s3(local_file, s3_key)

                write_checkpoint(cloth['Key'])

        except Exception as e:
            print(f"Error processing {cloth['Key']}: {str(e)}")

        finally:
            # Cleanup temp files
            if os.path.exists(source_image_path):
                os.remove(source_image_path)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

def create_target_meshes():
    """Main function to fetch images from DagsHub S3 Storage and process them in batches."""


    # Retrieve all image paths from the S3 bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="cloth_images/")
    all_images = response.get('Contents', [])
    all_images.sort(key=lambda x: x['Key'])
    last_processed = get_last_processed()

    # Find where to resume processing
    if last_processed and last_processed != "All_done":
        try:
            start_idx = next(i for i, img in enumerate(all_images) if img['Key'] == last_processed) + 1
            all_images = all_images[start_idx:]
        except (StopIteration, ValueError):
            start_idx = 0

    if not all_images:
        write_checkpoint("All_done")
        print("All files have been processed!")
        return

    temp_input_dir = os.path.join(TEMP_DIR, "input")
    
    # Process images in batches
    for i in range(0, len(all_images), BATCH_SIZE):
        batch = all_images[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1}")
        process_batch(batch, temp_input_dir)

    print("Processing completed successfully!")

if __name__ == "__main__":
    create_target_meshes()
