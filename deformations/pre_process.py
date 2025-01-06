import os
import json
import shutil
import torch
from PIL import Image
from salesforce.lavis.models import load_model_and_preprocess

# Define the mapping of base meshes to garment-related keywords
MESH_KEYWORDS = {
    "tshirt": ["t-shirt", "shirt", "short-sleeve", "tee"],
    "longsleeve": ["long-sleeve", "long sleeve", "sweater"],
    "tanktop": ["tank top", "sleeveless"],
    "polo": ["polo"],
    "poncho": ["poncho"],
    "dress_shortsleeve": ["short-sleeve dress", "short sleeve dress", "dress"],
}

# Base mesh paths
BASE_MESH_DIR = "meshes"  # Directory containing base meshes
OUTPUT_DIR = "prepared_dataset"  # Directory to save processed data
DATASET_DIR = "clothing_dataset"  # Directory containing garment images

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the Salesforce-LAVIS captioning model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="base_coco", is_eval=True, device=device
)


def generate_caption(image_path):
    """Generate a caption for the given image using Salesforce-LAVIS."""
    image = Image.open(image_path).convert("RGB")
    processed_image = vis_processors["eval"](image).unsqueeze(0).to(device)
    caption = model.generate({"image": processed_image})[0]
    return caption.lower()


def match_base_mesh(caption):
    """Match the generated caption to the appropriate base mesh."""
    for mesh, keywords in MESH_KEYWORDS.items():
        if any(keyword in caption for keyword in keywords):
            return mesh
    return None  # No match found


def process_clothing_dataset(dataset_dir, output_dir):
    # Initialize a mapping dictionary to save
    dataset_mapping = {}

    # Iterate over each clothing item in the dataset directory
    for clothing_item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, clothing_item)

        # Ensure it's a directory (one folder per clothing item)
        if not os.path.isdir(item_path):
            continue

        # Look for an image file in the folder
        image_files = [f for f in os.listdir(item_path) if f.endswith((".jpg", ".png"))]
        if not image_files:
            print(f"Skipping {clothing_item}: No image found.")
            continue

        image_path = os.path.join(item_path, image_files[0])  # Use the first image
        caption = generate_caption(image_path)  # Generate a caption for the image
        print(f"Generated caption for {clothing_item}: {caption}")

        # Match the caption to a base mesh
        base_mesh_key = match_base_mesh(caption)
        if base_mesh_key:
            base_mesh_path = os.path.join(BASE_MESH_DIR, f"{base_mesh_key}.obj")
        else:
            print(f"Skipping {clothing_item}: No matching base mesh for caption '{caption}'.")
            continue

        # Copy the base mesh to the item's folder in the output directory
        item_output_dir = os.path.join(output_dir, clothing_item)
        os.makedirs(item_output_dir, exist_ok=True)
        shutil.copy(base_mesh_path, os.path.join(item_output_dir, "base_mesh.obj"))

        # Save the mapping of the item to its base mesh
        dataset_mapping[clothing_item] = {
            "caption": caption,
            "base_mesh": os.path.join(item_output_dir, "base_mesh.obj"),
        }

    # Save the mapping to a JSON file for reference
    mapping_file = os.path.join(output_dir, "dataset_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(dataset_mapping, f, indent=4)
    print(f"Dataset processed and saved to {output_dir}")


# Run the dataset processing
process_clothing_dataset(DATASET_DIR, OUTPUT_DIR)