import os
import json
import shutil
import torch
from PIL import Image
import subprocess
from dagshub.streaming import DagsHubFilesystem

BASE_MESH_DIR = "/home/tyalaman/skunkworks/deformations/base_meshes"  # Directory containing base meshes
DAGSHUB_REPO = "YalmanchiliTejas/skunkworks"
INPUT_STORAGE_PATH = "cloth_images"
OUTPUT_STORAGE_PATH = "deformations/prepared_dataset"
BATCH_SIZE = 10
TEMP_DIR = "/home/tyalaman/skunkworks/deformations/temp"
LOG = "/home/tyalaman/skunkworks/deformations/logs/pre_process.log"
os.environ["NUMBA_DISABLE_CACHING"] = "1"

#process = ["python", "run.py", "./configs/instant-mesh-large.yaml", source_image_path, "--output_path", output_path,"--save_video", "--export_texmap"]
# result = subprocess.run(process, capture_output=True, text=True)

#write to log the latest file completed or will overwrite the entire log with "Batch_done" once the batch is done
def write_checkpoint(str):
    with open (LOG, 'w') as f:
        f.write(str)
    return
def get_last_processed():
    try:
        with open(LOG, 'r') as f:
            return f.read()
    except:
        return None

def process_batch(fs, image_batch, temp_dir):
    for cloth in image_batch:
        # Create unique directories for this image
        image_name = os.path.splitext(cloth)[0]  # Remove file extension
        source_image_path = os.path.join(temp_dir, cloth)
        output_path = os.path.join(temp_dir, "output", image_name)
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Download the source image
            fs.get(f"{INPUT_STORAGE_PATH}/{cloth}", source_image_path)
            
            # Check if already processed by checking remote storage
            try:
                existing_files = fs.listdir(f"{OUTPUT_STORAGE_PATH}/{image_name}")
                if len(existing_files) > 0:
                    print(f"Skipping {cloth} - already processed")
                    continue
            except:
                pass  # Directory doesn't exist, proceed with processing
            
            # Process the image
            process = ["python", "run.py", 
                      "./configs/instant-mesh-large.yaml", 
                      source_image_path, 
                      "--output_path", output_path,
                      "--save_video", 
                      "--export_texmap"]
            
            result = subprocess.run(process, capture_output=True, text=True)
            print(f"Processed {cloth}: Return code {result.returncode}")
            
            # Upload results back to DagsHub in organized structure
            if result.returncode == 0:
                # Create remote directory structure for this image
                remote_image_dir = f"{OUTPUT_STORAGE_PATH}/{image_name}"
                
                # Upload each output file to its corresponding directory
                for output_file in os.listdir(output_path):
                    local_file = os.path.join(output_path, output_file)
                    remote_path = f"{remote_image_dir}/{output_file}"
                    
                    # Upload the file
                    fs.put(local_file, remote_path)
                    print(f"Uploaded {output_file} to {remote_path}")
                
                # Update checkpoint after successful processing and upload
                write_checkpoint(cloth)
            
        except Exception as e:
            print(f"Error processing {cloth}: {str(e)}")
        finally:
            # Clean up temporary files
            if os.path.exists(source_image_path):
                os.remove(source_image_path)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

def create_target_meshes():
  
    fs = DagsHubFilesystem(DAGSHUB_REPO)
    
    # Get all images and find starting point
    all_images = sorted(fs.listdir(INPUT_STORAGE_PATH))
    last_processed = get_last_processed()
    
    if last_processed and last_processed != "All_done":
        try:
            start_idx = all_images.index(last_processed) + 1
            all_images = all_images[start_idx:]
        except ValueError:
            start_idx = 0
    
    if not all_images:
        write_checkpoint("All_done")
        print("All files have been processed!")
        return
    
    # Process images in batches
    temp_input_dir = os.path.join(TEMP_DIR, "input")
    for i in range(0, len(all_images), BATCH_SIZE):
        batch = all_images[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}")
        process_batch(fs, batch, temp_input_dir)
    print("Processing completed successfully!")

if __name__ == "__main__":
    create_target_meshes()
    