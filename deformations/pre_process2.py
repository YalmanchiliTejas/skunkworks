import os
import json
import shutil
import torch
from PIL import Image
import subprocess

BASE_MESH_DIR = "/home/tyalaman/skunkworks/deformations/base_meshes"  # Directory containing base meshes
OUTPUT_DIR = "/home/tyalaman/skunkworks/deformations/prepared_dataset"  # Directory to save processed data
DATASET_DIR = "/home/tyalaman/skunkworks/deformations/cloth_images"  # Directory containing garment images
os.environ["NUMBA_DISABLE_CACHING"] = "1"

def create_target_meshes(directory):

    
    for cloth in os.listdir(directory):
        

         
        source_image_path = os.path.join(DATASET_DIR, cloth)
        output_path = os.path.join(directory,cloth)
        if not (os.path.isdir(output_path)):
            continue
        print(output_path, flush=True)
        print(len(os.listdir(output_path)), flush=True)

        if len(os.listdir(output_path)) > 1:
            continue
        process = ["python", "run.py", "./configs/instant-mesh-large.yaml", source_image_path, "--output_path", output_path,"--save_video", "--export_texmap"]
        print("This is before the command is run", flush=True)
        result = subprocess.run(process, capture_output=True, text=True)
        print("This is after the command is run", flush=True)
        print(result.stdout, flush=True)
        print(f"Standard error: {result.stderr}", flush=True)        
        print(f"Return code:{result.returncode}", flush=True)


# Run the dataset processing
#process_clothing_dataset(DATASET_DIR, OUTPUT_DIR)
create_target_meshes(OUTPUT_DIR)

