import os
import json
import shutil
import torch
from PIL import Image
import subprocess
from dagshub.data_engine import datasources
import requests

BASE_MESH_DIR = "/home/tyalaman/skunkworks/deformations/base_meshes"
DAGSHUB_REPO = "YalmanchiliTejas/skunkworks"
BATCH_SIZE = 10
TEMP_DIR = "/home/tyalaman/skunkworks/deformations/temp"
LOG = "/home/tyalaman/skunkworks/deformations/logs/pre_process.log"
os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["DAGSHUB_CLIENT_HOST"] = "https://dagshub.com"

def write_checkpoint(str):
    with open(LOG, 'w') as f:
        f.write(str)
    return

def get_last_processed():
    try:
        with open(LOG, 'r') as f:
            return f.read()
    except:
        return None

def process_batch(ds, image_batch, temp_dir):
    for cloth in image_batch:
        image_name = os.path.splitext(cloth['path'])[0]
        source_image_path = os.path.join(temp_dir, cloth['path'])
        output_path = os.path.join(temp_dir, "output", image_name)
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Download using download_url from datasource
            with open(source_image_path, 'wb') as f:
                response = requests.get(cloth['dagshub_download_url'])
                f.write(response.content)
            
            # Check if already processed
            try:
                existing_query = ds[ds['path'].str.startswith(f"deformations/prepared_dataset/{image_name}")]
                if len(existing_query.all().dataframe) > 0:
                    print(f"Skipping {cloth['path']} - already processed")
                    continue
            except:
                pass
            
            process = ["python", "run.py", 
                      "./configs/instant-mesh-large.yaml", 
                      source_image_path, 
                      "--output_path", output_path,
                      "--save_video", 
                      "--export_texmap"]
            
            result = subprocess.run(process, capture_output=True, text=True)
            print(f"Processed {cloth['path']}: Return code {result.returncode}")
            
            if result.returncode == 0:
                # Upload results using DagsHub's data engine
                for output_file in os.listdir(output_path):
                    local_file = os.path.join(output_path, output_file)
                    ds.add_files(local_file, f"deformations/prepared_dataset/{image_name}/{output_file}")
                
                write_checkpoint(cloth['path'])
            
        except Exception as e:
            print(f"Error processing {cloth['path']}: {str(e)}")
        finally:
            if os.path.exists(source_image_path):
                os.remove(source_image_path)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

def create_target_meshes():
    # Initialize datasource
    ds = datasources.get(DAGSHUB_REPO, 'skunkworks')
    print("Datasource initialized", flush=True)
    
    print(ds['path'], flush=True)
    # Query all images from cloth_images directory
    query = ds.query(
        ds['path'].str.startswith('cloth_images/') &
        ds['media type'].str.contains('image')
    )
    print(query, flush=True)
    all_images = query.all().dataframe.to_dict('records')
    all_images.sort(key=lambda x: x['path'])
    
    last_processed = get_last_processed()
    
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
    for i in range(0, len(all_images), BATCH_SIZE):
        batch = all_images[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}")
        process_batch(ds, batch, temp_input_dir)
    print("Processing completed successfully!")

if __name__ == "__main__":
    create_target_meshes()
