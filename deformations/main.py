from Jacobian import SourceMesh, PoissonSystem, MeshProcessor
import torch
from pytorch3d.structures import Meshes
from deformations.util import *
from fashion_clip import FashionCLIP
from pytorch3d.io import save_obj
from utilities.video import Video
from utilities.clip_visualencoder import CLIPVisualEncoder
from pytorch3d.ops import sample_points_from_meshes, chamfer_distance, mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing
import tqdm
from utilities.camera_batch import CameraBatch
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
)


def update_packed_verts(mesh, new_verts):
    faces = mesh.faces_packed()
    textures = mesh.textures
    updated_mesh = Meshes(verts=[new_verts], faces=[faces], textures=textures)
    return updated_mesh


def total_triangle_areas(mesh):
    vertices = mesh.verts_packed()
    faces = mesh.faces_packed()
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
    areas = 0.5 * torch.norm(cross_product, dim=1)
    return torch.sum(areas).item()


def triangle_area_regularization(mesh):
    return total_triangle_areas(mesh) ** 2

def sde_edit(texture, image_caption, device):

    sde_edit = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

    return sde_edit(prompt=image_caption, image=texture, strength=0.5,num_inference_steps=50, guidance_scale=7.5).images[0]
def deformations(input_mesh, target_mesh, epochs, output_path):
    lr = 0.0025
    clip_weight = 2.5
    delta_clip_weight = 5
    regularize_jacobians_weight = 0.15
    consistency_loss_weight = 0.1
    batch_size = 24
    consistency_clip_model = 'ViT-B/32'
    consistency_vit_stride = 8
    consistency_elev_filter =  30
    consistency_azim_filter =  20
    cams_data = CameraBatch(
        512,
        [2.5, 3.5],
        [0.0,360.0],
        [1.0, 5.0, 60.0],
        [30.0, 90.0],
        1,
        1,
        0,
        batch_size,
        rand_solid=True
    )
    
   
    cams = torch.utils.data.DataLoader(cams_data, batch_size, num_workers=0, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fe = CLIPVisualEncoder(consistency_clip_model, consistency_vit_stride, device)
    input_mesh_loaded = get_mesh(input_mesh, "./input_mesh", False, False, 'input_mesh.obj')
    target_mesh_loaded = get_mesh(target_mesh, "./target_mesh", False, False, 'target_mesh.obj')

    ground_truth = SourceMesh.SourceMesh(0, "./source_mesh", {}, 1, ttype=torch.float32, 
                                         use_wks=False, random_centering=False, cpuonly=False)
    ground_truth.load()
    ground_truth.to(device)

    

    fclip = FashionCLIP('fashion-clip')
    clip_mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    with torch.no_grad():
        ground_truth_jacobians = ground_truth.jacobians_from_vertices(
            input_mesh_loaded.verts_packed().unsqueeze(0))

    ground_truth_jacobians.requires_grad_(True)
    optimizer = torch.optim.Adam([ground_truth_jacobians], lr=lr)

    video = Video(output_path)
    

    training_loop = tqdm(range(epochs), leave=False)
    for t in training_loop:

        camera_params = next(iter(cams))
        
        cameras = setup_cameras(camera_params, device=device)
        lights = setup_lights(camera_params,device=device)
        renderer = setup_renderer(image_size=512, cameras=cameras,device=device, lights=lights)
        target_fragments = renderer.rasterizer(target_mesh)

        target_rendered_image = renderer.shader(target_fragments, target_mesh)

        total_loss = 0

        updated_vertices = ground_truth.vertices_from_jacobians(ground_truth_jacobians).squeeze()
        input_mesh_loaded = update_packed_verts(input_mesh_loaded, updated_vertices)

        target_sample = sample_points_from_meshes(target_mesh_loaded, 10000)
        input_sample = sample_points_from_meshes(input_mesh_loaded, 10000)
        chamfer_dist = chamfer_distance(target_sample, input_sample)
        ##REGULARIZATION LOSSES
        edge_loss = mesh_edge_loss(input_mesh_loaded)
        normal_loss = mesh_normal_consistency(input_mesh_loaded)
        laplacian_loss = mesh_laplacian_smoothing(input_mesh_loaded)
        triangle_area_regularization = triangle_area_regularization(input_mesh_loaded)/100000.
        ######REGULARIZATION LOSSES END

        jacobian_reg_loss = ((ground_truth_jacobians - torch.eye(3, 3, device=device)) ** 2).mean()

        input_fragments = renderer.rasterizer(input_mesh_loaded)
        input_rendered_image = renderer.shader(input_fragments, input_mesh_loaded)
        l1_loss = torch.nn.functional.l1_loss(input_rendered_image, target_rendered_image, reduction='mean')

        deformed_features = fclip.encode_image_tensors(input_rendered_image)
        target_features = fclip.encode_image_tensors(target_rendered_image)
        clip_loss = -1 * torch.nn.functional.cosine_similarity(deformed_features, target_features, dim=-1).mean()

        train_rast_map = {
                            "depth": input_fragments.zbuf.squeeze(-1),
                            "barycentric": input_fragments.bary_coords,
                            "face_indices": input_fragments.pix_to_face
                        }
        params_camera = get_camera_params(
                                        elev_angle=30,  # Example elevation angle
                                        azim_angle=45,  # Example azimuth angle
                                        distance=3.0,   # Example distance from object
                                        resolution=512, # Resolution matching the renderer
                                        fov=60
                                    )

        # Normalize input_rendered_image for consistency loss
        normalized_clip_render = (input_rendered_image - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]

        # Example configuration settings for compute_mv_cl
        cfg = {
            "consistency_vit_layer": 11,      # Adjust based on your feature extractor
            "batch_size": batch_size,                 # Number of views/batches
            "consistency_elev_filter": consistency_elev_filter,   # Elevation filter threshold in degrees
            "consistency_azim_filter": consistency_azim_filter    # Azimuth filter threshold in degrees
        }

        consistency_loss = consistency_loss_weight * compute_mv_cl(
                                                                        final_mesh=input_mesh_loaded,
                                                                        fe=fe, 
                                                                        normalized_clip_render=normalized_clip_render,
                                                                        params_camera=params_camera,
                                                                        train_rast_map=train_rast_map,
                                                                        cfg=cfg,
                                                                        device=device
                                                                    )

        total_loss += (clip_weight * clip_loss + delta_clip_weight * l1_loss +
                       regularize_jacobians_weight * jacobian_reg_loss +
                       chamfer_dist + edge_loss + normal_loss + laplacian_loss + consistency_loss + triangle_area_regularization)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        training_loop.set_description(
            f"Loss: {total_loss.item():.4f}, CLIP: {clip_loss.item():.4f}, Chamfer: {chamfer_dist.item():.4f}")

        if t % 10 == 0:
            frame = input_rendered_image.permute(0, 2, 3, 1).cpu().numpy()[0]
            video.add_frame(frame)

    video.close()

    save_obj(str(output_path / 'mesh_final.obj'), input_mesh_loaded.verts_packed(), input_mesh_loaded.faces_packed())

    return


def texture_estimation(input_mesh, output_path, image_path, depth_weight, inpainting_weight ,device):

    epochs = 100
    ## SETTING UP THE CONTROLNET PIPELINE
    depth_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_depth", torch_dtype=torch.float32).to(device)
    inpainting_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpainting", torch_dtype=torch.float32).to(device)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=[depth_controlnet, inpainting_controlnet], torch_dtype=torch.float32).to(device)
    #### END CONTROLNET PIPELINE SETUP
    
    ## SETTING UP THE RENDERER AND RENDERING BOTH FRONT AND BACK IMAGES SIMULTANEOUSLY
    front_view_params = get_camera_params(elev_angle=0, azim_angle=0, distance=3.0, resolution=512, fov=60)
    back_view_params = get_camera_params(elev_angle=0, azim_angle=180, distance=3.0, resolution=512, fov=60)
    front_view = setup_cameras(front_view_params, device=device)
    back_view = setup_cameras(back_view_params, device=device)
    front_light = setup_lights(front_view, device=device)
    back_light = setup_lights(back_view, device=device)
    front_renderer = setup_renderer(image_size=512, cameras=front_view, device=device, lights=front_light)
    back_renderer = setup_renderer(image_size=512, cameras=back_view, device=device, lights=back_light)

    front_image_fragements = front_renderer.rasterize(meshes=input_mesh)
    front_image_rendered = front_renderer.shader(front_image_fragements, input_mesh)
    back_image_fragements = back_renderer.rasterize(meshes=input_mesh)
    back_image_rendered = back_renderer.shader(back_image_fragements, input_mesh)
    front_depth = front_image_fragements.zbuf.squeeze(-1)
    front_depth = (front_depth - front_depth.min()) / (front_depth.max() - front_depth.min()) ##Normalizing it to make it simpler for prediction
    back_depth = back_image_fragements.zbuf.squeeze(-1)
    back_depth = (back_depth - back_depth.min()) / (back_depth.max() - back_depth.min()) ##Normalizing it to make it simpler for prediction

    ## END SETUP AND RENDERING THE FRONT AND BACK IMAGES

    ##TODO: USE AN IMAGE CAPTIONING MODEL TO GET THE CAPTION OF THE IMAGE
    image_caption = None

    ##END CAPTION MODEL ####
    depth_weight = torch.nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))
    inpainting_weight = torch.nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))

    # Optimizer for the weights
    optimizer = torch.optim.Adam([depth_weight, inpainting_weight], lr=0.0025)

    for epoch in range(epochs):
        optimizer.zero_grad()
        front_texture = pipeline(
            prompt=image_caption,
            image=front_image_rendered,
            controlnet_conditioning_image=front_depth,
            num_inference_steps=50,
            guidance_scale=7.5,
            controlnet_conditioning_scale=[depth_weight.item(), inpainting_weight.item()]
        ).images[0]

        back_texture= pipeline(
            prompt=image_caption,
            image=back_image_rendered,
            controlnet_conditioning_image=back_depth,
            num_inference_steps=50,
            guidance_scale=7.5,
            controlnet_conditioning_scale=[depth_weight.item(), inpainting_weight.item()]
        ).images[0]

        texture_map = initialize_texture_map(input_mesh, device)
        texture_map = update_texture_map(texture_map, front_texture)
        texture_map = update_texture_map(texture_map, back_texture)
        binary_mask = create_binary_mask(texture_map)
        input_mesh = update_mesh_texture(input_mesh, texture_map)

        candidate_views = generate_candidate_views(12)
        for view in candidate_views:
            candidate_masks = calculate_candidate_masks(binary_mask, candidate_views, input_mesh, device)

            least_painted_view = select_least_painted_view(candidate_masks)
            selected_camera = candidate_views[least_painted_view]
            selected_camera_params = get_camera_params(elev_angle=selected_camera['elev_angle'], azim_angle=selected_camera['azim_angle'], distance=selected_camera['distance'], resolution=512, fov=60)
            camera = setup_cameras(selected_camera_params, device=device)
            lights = setup_lights(camera, device=device)
            view_renderer = setup_renderer(image_size=512, cameras=camera, device=device, lights=lights)
            view_fragements = view_renderer.rasterize(meshes=input_mesh)
            view_rendered = view_renderer.shader(view_fragements, input_mesh)
            view_depth = view_fragements.zbuf.squeeze(-1)
            view_depth = (view_depth - view_depth.min()) / (view_depth.max() - view_depth.min()) ##Normalizing it to make it simpler for prediction
            original_mask = binary_mask.clone()

            view_texture = pipeline(
            prompt=image_caption,
            image=view_rendered,
            controlnet_conditioning_image=view_depth,
            num_inference_steps=50,
            guidance_scale=7.5,
            controlnet_conditioning_scale=[depth_weight.item(), inpainting_weight.item()]
            ).images[0]

            ##SDEedit refinement stage for the texture
            refined_texture = sde_edit(view_texture, image_caption, device) 
            texture_map = update_texture_map(texture_map, refined_texture)
            binary_mask = create_binary_mask(texture_map)
            input_mesh = update_mesh_texture(input_mesh, texture_map)
            after_appl_renderer = setup_renderer(image_size=512, cameras=camera, device=device, lights=lights)
            after_appl_fragments = after_appl_renderer.rasterize(meshes=input_mesh)
            after_appl_rendered = after_appl_renderer.shader(after_appl_fragments, input_mesh)
            after_appl_depth = after_appl_fragments.zbuf.squeeze(-1)
            after_appl_depth = (after_appl_depth - after_appl_depth.min()) / (after_appl_depth.max() - after_appl_depth.min())

            inpainting_loss = compute_inpainting_loss(original_mask, binary_mask, device)
            depth_loss = torch.nn.functional.mse_loss(after_appl_depth,view_depth)
            total_textured_loss = inpainting_loss + depth_loss
            print(f"The total textured loss, is: inpainting: {inpainting_loss.item()}, depth: {depth_loss.item()} and the total textured_loss: {total_textured_loss.item()}")
            total_textured_loss.backward()
            optimizer.step()
        save_obj(str(output_path / 'mesh_final_texutred.obj'), input_mesh.verts_packed(), input_mesh.faces_packed())
