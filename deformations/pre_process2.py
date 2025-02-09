import os
import json
import shutil
import torch
from PIL import Image
import subprocess
from dagshub.data_engine import datasources
import dagshub.streaming as ds_streaming
import requests

# Configuration Constants
BASE_MESH_DIR = "/home/tyalaman/skunkworks/deformations/base_meshes"
DAGSHUB_REPO = "YalmanchiliTejas/skunkworks"
BATCH_SIZE = 10
TEMP_DIR = "/home/tyalaman/skunkworks/deformations/temp"
LOG = "/home/tyalaman/skunkworks/deformations/logs/pre_process.log"

# Set environment variables for DagsHub authentication
os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["DAGSHUB_CLIENT_HOST"] = "https://dagshub.com"

def write_checkpoint(last_processed):
    """Writes the last processed image path to a log file."""
    with open(LOG, 'w') as f:
        f.write(last_processed)
    return

def get_last_processed():
    """Reads the last processed image path from the log file."""
    try:
        with open(LOG, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def process_batch(ds, image_batch, temp_dir):
    """
    Processes a batch of images:
    1. Downloads each image from the DagsHub Storage Bucket.
    2. Runs the processing script.
    3. Uploads processed results back to the dataset.
    """
    for cloth in image_batch:
        image_name = os.path.splitext(os.path.basename(cloth['path']))[0]
        source_image_path = os.path.join(temp_dir, cloth['path'])
        output_path = os.path.join(temp_dir, "output", image_name)
        os.makedirs(output_path, exist_ok=True)

        try:
            # Download image using DagsHub Streaming API
            with ds_streaming.open(cloth['path'], "rb") as f:
                with open(source_image_path, "wb") as img_file:
                    img_file.write(f.read())

            # Skip if the image has already been processed
            existing_query = ds.query(ds['path'].str.startswith(f"deformations/prepared_dataset/{image_name}"))
            if len(existing_query.all().dataframe) > 0:
                print(f"Skipping {cloth['path']} - already processed")
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
            print(f"Processed {cloth['path']}: Return code {result.returncode}")

            if result.returncode == 0:
                # Upload processed files back to DagsHub dataset
                for output_file in os.listdir(output_path):
                    local_file = os.path.join(output_path, output_file)
                    ds.add_files(local_file, f"deformations/prepared_dataset/{image_name}/{output_file}")

                write_checkpoint(cloth['path'])

        except Exception as e:
            print(f"Error processing {cloth['path']}: {str(e)}")

        finally:
            # Cleanup temp files
            if os.path.exists(source_image_path):
                os.remove(source_image_path)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

def create_target_meshes():
    """Main function to fetch images from DagsHub Storage and process them in batches."""
    # Initialize DagsHub Data Engine
    ds = datasources.get(DAGSHUB_REPO, "skunkworks")
    print("Datasource initialized", flush=True)

    # Retrieve all image paths from the storage bucket
    all_images = ds.get_files(prefix="cloth_images/", file_types=["image"]).dataframe.to_dict('records')
    all_images.sort(key=lambda x: x['path'])

    last_processed = get_last_processed()

    # Find where to resume processing
    if last_processed and last_processed != "All_done":
        try:
            start_idx = next(i for i, img in enumerate(all_images) if img['path'] == last_processed) + 1
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
        process_batch(ds, batch, temp_input_dir)

    print("Processing completed successfully!")

if __name__ == "__main__":
    create_target_meshes()
