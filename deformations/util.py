import torch
import glm
from pytorch3d.renderer import (
    MeshRenderer, 
    MeshRasterizer,
    RasterizationSettings,
    ShaderBase,
    TexturesUV,
    PerspectiveCameras,
    PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
import pymeshlab
import numpy as np


class MaterialMaps:
    """Class to hold material texture maps"""
    def __init__(self, device):
        # Initialize texture maps as None
        self.diffuse_map = None
        self.normal_map = None
        self.specular_map = None 
        self.roughness_map = None
        self.device = device
    
    def set_maps(self, diffuse=None, normal=None, specular=None, roughness=None):
        if diffuse is not None:
            self.diffuse_map = diffuse.to(self.device)
        if normal is not None:
            self.normal_map = normal.to(self.device)
        if specular is not None:
            self.specular_map = specular.to(self.device)
        if roughness is not None:
            self.roughness_map = roughness.to(self.device)

class FlexibleMaterialShader(ShaderBase):
    def __init__(self, device="cpu", cameras=None, lights=None, materials=None):
        super().__init__(device=device, cameras=cameras, lights=lights, materials=materials)
    
    def blend_images(self, fragments, shaded, alpha):
        """
        Perform front-to-back alpha compositing of multiple fragments.

        Args:
            fragments: A RasterizationFragments object from PyTorch3D.
                       - fragments.zbuf: [N, H, W, K] for depth values (K fragments per pixel).
            shaded: Shaded colors for each fragment. Shape: [N, H, W, K, 3].
            alpha:  Alpha values for each fragment. Shape: [N, H, W, K].

        Returns:
            blended_image: The final image after front-to-back compositing (N, H, W, 3).
        """
        # -- (A) Sort fragments by ascending depth so we blend front-to-back
        zbuf, sort_idx = fragments.zbuf.sort(dim=-1)  # [N, H, W, K] sorted by depth
        # Gather 'shaded' and 'alpha' in sorted order
        shaded_sorted = torch.gather(
            shaded,
            dim=-2,  # gather along the K dimension
            index=sort_idx.unsqueeze(-1).expand_as(shaded)
        )  # [N, H, W, K, 3]
        alpha_sorted = torch.gather(alpha, dim=-1, index=sort_idx)  # [N, H, W, K]

        # -- (B) Front-to-back compositing loop
        # Start with zero color and zero coverage
        composite = torch.zeros_like(shaded_sorted[..., 0, :])  # [N, H, W, 3]
        composite_alpha = torch.zeros_like(alpha_sorted[..., 0]) # [N, H, W]

        K = shaded_sorted.shape[-2]
        for k in range(K):
            color_k = shaded_sorted[..., k, :]    # [N, H, W, 3]
            alpha_k = alpha_sorted[..., k]        # [N, H, W]
            one_minus_ca = (1.0 - composite_alpha)

            # standard "over" operator
            composite = composite + color_k * alpha_k.unsqueeze(-1) * one_minus_ca.unsqueeze(-1)
            composite_alpha = composite_alpha + alpha_k * one_minus_ca

        return composite  # shape: (N, H, W, 3)

    def forward(self, fragments, meshes, **kwargs):
        """
        Shader forward pass with PyTorch3D-based lighting & material usage.

        Args:
            fragments: Output of rasterization (RasterizationFragments).
            meshes: The mesh with textures/UV.
            **kwargs: Potentially includes "cameras", "lights", etc.
        """
        cameras = kwargs.get("cameras", self.cameras)
        lights = kwargs.get("lights", self.lights)
        material_maps = kwargs.get("materials", self.material_maps)

        # 1) Sample texture (UV-based diffuse) from the mesh
        texels = meshes.sample_textures(fragments)  # shape: [N, H, W, K, 4] typically
        diffuse_color = texels[..., :3]
        alpha = texels[..., 3:4] if texels.shape[-1] > 3 else torch.ones_like(diffuse_color[..., :1])

        # 2) Interpolate vertex normals -> per-pixel normals
        verts_normals = meshes.verts_normals_packed()  # shape: (V, 3)
        faces = meshes.faces_packed()                  # shape: (F, 3)
        pixel_faces = fragments.pix_to_face           # shape: (N, H, W, K)
        bary_coords = fragments.bary_coords           # shape: (N, H, W, K, 3)

        # Gather the vertex normals for each face
        face_normals = verts_normals[faces]   # shape: (F, 3, 3)
        # Interpolate
        pixel_normals = torch.sum(
            face_normals[pixel_faces] * bary_coords.unsqueeze(-1),
            dim=-2
        )  # shape: (N, H, W, 3)

        # Normalize
        pixel_normals = torch.nn.functional.normalize(pixel_normals, dim=-1)

        # 3) If you want to sample the separate "diffuse_map" from material_maps, do so here
        # e.g. if material_maps.diffuse_map is a (512, 512, 3) texture, you'd sample it with UVs
        # the same way PyTorch3D does. For simplicity, let's assume we just multiply them:
        if material_maps is not None and material_maps.diffuse_map is not None:
            # shape: (1, 3, 512, 512) if you did something like material_maps.diffuse_map.unsqueeze(0).permute(2, 0, 1)
            # you'd need to sample from it using 'texels' UV coords or something similar.
            # For a minimal example, let's just do a naive scaling:
            diffuse_color = diffuse_color * 0.8  # or some placeholder logic

        # 4) Lighting
        if lights is not None and cameras is not None:
            # Basic Lambert + Blinn-Phong
            # 4a) Light direction
            light_pos = lights.location  # shape: (batch_size, 3)
            # Expand to match pixel shape if needed
            light_dir = light_pos.view(-1, 1, 1, 1, 3) - fragments.world_coordinates  # approximate
            light_dir = torch.nn.functional.normalize(light_dir, dim=-1)

            # Dot for lambert
            lambert = (pixel_normals * light_dir).sum(dim=-1).clamp(min=0)

            # 4b) If specular_map is present, sample or use the map:
            # For minimal example, let's do a Blinn-Phong spec with a fixed shininess:
            specular_color = 0
            if material_maps is not None and material_maps.specular_map is not None:
                # You can do more advanced sampling from the specular map
                # For now, let's assume you do a simple highlight:
                view_dir = -fragments.world_coordinates  # e.g. camera at origin
                view_dir = torch.nn.functional.normalize(view_dir, dim=-1)
                half_dir = torch.nn.functional.normalize(light_dir + view_dir, dim=-1)
                spec_angle = (pixel_normals * half_dir).sum(dim=-1).clamp(min=0)
                specular_color = spec_angle ** 32  # arbitrary shininess

            # Combine
            # lights.diffuse_color: shape (batch_size, 3)
            out_rgb = (
                diffuse_color * lambert.unsqueeze(-1) * lights.diffuse_color.view(-1, 1, 1, 1, 3) +
                specular_color.unsqueeze(-1) * lights.specular_color.view(-1, 1, 1, 1, 3) +
                lights.ambient_color.view(-1, 1, 1, 1, 3)
            )
        else:
            # No lighting => just raw diffuse
            out_rgb = diffuse_color

        # 5) Alpha blend
        blended_image = self.blend_images(fragments, out_rgb, alpha)
        return blended_image
        

# Helper functions to create texture maps
def create_texture_map(size, device):
    return torch.from_numpy(np.random.uniform(size=[size, size, 3], low=0.0, high=1.0)).float().to(device)

def create_normal_map(size, device):
    return torch.full((size, size, 3), 1.0).float().to(device)

def create_specular_map(size, device):
    return torch.zeros((size, size, 3)).float().to(device)

def get_mesh(mesh_path, output_path, triangulate_flag, mesh_name="mesh.obj", device="cuda"):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    if triangulate_flag:
        print('Retriangulating shape...')
        ms.meshing_isotropic_explicit_remeshing()

    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
    
    ms.save_current_mesh(str(output_path /mesh_name))

    # Load vertices, faces, and UVs from PyMeshLab
    vertices = torch.tensor(ms.current_mesh().vertex_matrix(), dtype=torch.float32).to(device)
    faces = torch.tensor(ms.current_mesh().face_matrix(), dtype=torch.int64).to(device)
    tex_coords = torch.tensor(ms.current_mesh().wedge_tex_coord_matrix(), dtype=torch.float32).to(device)
    face_uvs = torch.tensor(ms.current_mesh().wedge_tex_coord_index_matrix(), dtype=torch.int64).to(device)

    # Create different texture maps
    diffuse_map = create_texture_map(512, device)
    normal_map = create_normal_map(512, device)
    specular_map = create_specular_map(512, device)
    
    # Create material maps
    material_maps = MaterialMaps(device)
    material_maps.set_maps(
        diffuse=diffuse_map,
        normal=normal_map,
        specular=specular_map
    )
    
    # Create base textures
    textures = TexturesUV(
        maps=diffuse_map.unsqueeze(0),
        faces_uvs=[face_uvs],
        verts_uvs=[tex_coords]
    )

    # Scale vertices to unit size
    verts_center = vertices.mean(dim=0)
    vertices = vertices - verts_center
    scale = 1.0 / vertices.abs().max()
    vertices = vertices * scale

    mesh = Meshes(
        verts=[vertices],
        faces=[faces],
        textures=textures
    )
    return mesh, material_maps

def setup_renderer(image_size, cameras, lights, device, material_maps=None):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=FlexibleMaterialShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=material_maps
        )
    )
    return renderer

def get_og_mesh(mesh_path, output_path, triangulate_flag, bsdf_flag, mesh_name='mesh.obj', device="cuda"):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    if triangulate_flag:
        print('Retriangulating shape...')
        ms.meshing_isotropic_explicit_remeshing()

    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)

    temp_path = str(output_path / 'tmp' / mesh_name)
    ms.save_current_mesh(temp_path)

    vertices = torch.tensor(ms.current_mesh().vertex_matrix(), dtype=torch.float32).to(device)
    faces = torch.tensor(ms.current_mesh().face_matrix(), dtype=torch.int64).to(device)
    tex_coords = torch.tensor(ms.current_mesh().wedge_tex_coord_matrix(), dtype=torch.float32).to(device)
    face_uvs = torch.tensor(ms.current_mesh().wedge_tex_coord_index_matrix(), dtype=torch.int64).to(device)

    texture_map = create_texture_map(512, device)
    textures = TexturesUV(
        maps=texture_map.unsqueeze(0),
        faces_uvs=[face_uvs],
        verts_uvs=[tex_coords]
    )

    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    return mesh

# Function to compute multi-view consistency loss
def compute_mv_cl(final_mesh, fe, normalized_clip_render, params_camera, train_rast_map, cfg, device):
    curr_vp_map = get_vp_map(final_mesh.verts_list()[0], params_camera['mvp'], 224)

    for idx, rast_faces in enumerate(train_rast_map[:, :, :, 3].view(cfg.batch_size, -1)):
        u_faces = rast_faces.unique().long()[1:] - 1
        t = torch.arange(len(final_mesh.verts_list()[0]), device=device)
        u_ret = torch.cat([t, final_mesh.faces_list()[0][u_faces].flatten()]).unique(return_counts=True)
        non_verts = u_ret[0][u_ret[1] < 2]
        curr_vp_map[idx][non_verts] = torch.tensor([224, 224], device=device)

    med = (fe.old_stride - 1) / 2
    curr_vp_map[curr_vp_map < med] = med
    curr_vp_map[(curr_vp_map > 224 - fe.old_stride) & (curr_vp_map < 224)] = 223 - med
    curr_patch_map = ((curr_vp_map - med) / fe.new_stride).round()
    flat_patch_map = curr_patch_map[..., 0] * (((224 - fe.old_stride) / fe.new_stride) + 1) + curr_patch_map[..., 1]

    patch_feats = fe(normalized_clip_render)
    flat_patch_map[flat_patch_map > patch_feats[0].shape[-1] - 1] = patch_feats[0].shape[-1]
    flat_patch_map = flat_patch_map.long()[:, None, :].repeat(1, patch_feats[0].shape[1], 1)

    deep_feats = patch_feats[cfg.consistency_vit_layer]
    deep_feats = torch.nn.functional.pad(deep_feats, (0, 1))
    deep_feats = torch.gather(deep_feats, dim=2, index=flat_patch_map)
    deep_feats = torch.nn.functional.normalize(deep_feats, dim=1, eps=1e-6)

    elev_d = torch.cdist(params_camera['elev'].unsqueeze(1), params_camera['elev'].unsqueeze(1)).abs() < torch.deg2rad(
        torch.tensor(cfg.consistency_elev_filter))
    azim_d = torch.cdist(params_camera['azim'].unsqueeze(1), params_camera['azim'].unsqueeze(1)).abs() < torch.deg2rad(
        torch.tensor(cfg.consistency_azim_filter))

    cosines = torch.einsum('ijk, lkj -> ilk', deep_feats, deep_feats.permute(0, 2, 1))
    cosines = (cosines * azim_d.unsqueeze(-1) * elev_d.unsqueeze(-1)).permute(2, 0, 1).triu(1)
    consistency_loss = cosines[cosines != 0].mean()
    return consistency_loss
def persp_proj(fov_x=45, ar=1, near=1.0, far=50.0):
    """
    From https://github.com/rgl-epfl/large-steps-pytorch by @bathal1 (Baptiste Nicolet)

    Build a perspective projection matrix.
    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)

    tanhalffov = np.tan( (fov_rad / 2) )
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)
    
    return proj_mat
def get_camera_params(elev_angle, azim_angle, distance, resolution, fov=60, look_at=[0, 0, 0], up=[0, -1, 0]):
    
    elev = np.radians( elev_angle )
    azim = np.radians( azim_angle ) 
    
    # Generate random view
    cam_z = distance * np.cos(elev) * np.sin(azim)
    cam_y = distance * np.sin(elev)
    cam_x = distance * np.cos(elev) * np.cos(azim)

    modl = glm.mat4()
    view  = glm.lookAt(
        glm.vec3(cam_x, cam_y, cam_z),
        glm.vec3(look_at[0], look_at[1], look_at[2]),
        glm.vec3(up[0], up[1], up[2]),
    )

    a_mv = view * modl
    a_mv = np.array(a_mv.to_list()).T
    proj_mtx = persp_proj(fov)
    
    a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]
    
    a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
    a_campos = a_lightpos

    return {
        'mvp' : a_mvp,
        'lightpos' : a_lightpos,
        'campos' : a_campos,
        'resolution' : [resolution, resolution], 
        }


def get_vp_map(v_pos, mtx_in, resolution):
    """
    Compute the viewport map using PyTorch3D without nvdiffmodeling.

    Args:
        v_pos (torch.Tensor): Vertices of the mesh (N, 3).
        mtx_in (torch.Tensor): 4x4 transformation matrix (e.g., camera view-projection matrix).
        resolution (int): Resolution of the output viewport.

    Returns:
        torch.Tensor: Viewport map with pixel coordinates for each vertex (N, 2).
    """
    device = v_pos.device

    # Define the viewport transformation matrix
    vp_mtx = torch.tensor([
        [resolution / 2, 0., 0., (resolution - 1) / 2],
        [0., resolution / 2, 0., (resolution - 1) / 2],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ], device=device)

    # Step 1: Apply the transformation matrix to vertex positions
    v_pos_homo = torch.cat([v_pos, torch.ones_like(v_pos[:, :1])], dim=-1)  # Convert to homogeneous coordinates (N, 4)
    v_pos_clip = v_pos_homo @ mtx_in.T  # Apply transformation matrix (N, 4)

    # Step 2: Perform perspective divide
    v_pos_ndc = v_pos_clip[:, :3] / v_pos_clip[:, 3:4]  # Normalize by w-component to get normalized device coordinates (N, 3)

    # Step 3: Map normalized device coordinates to viewport space
    v_pos_vp = (vp_mtx @ v_pos_ndc.T).T[..., :2]  # Map to viewport space (N, 2)

    # Step 4: Ensure coordinates are within valid range
    v_pos_vp = v_pos_vp.int()  # Convert to integer pixel values
    v_pos_vp = torch.flip(v_pos_vp, dims=[-1])  # Flip x and y coordinates to match image conventions
    v_pos_vp[(v_pos_vp < 0) | (v_pos_vp >= resolution)] = resolution  # Clamp out-of-bounds values to a placeholder

    return v_pos_vp.long()
def setup_cameras(camera_batch_data, device):
    """
    Setup PyTorch3D cameras using data from CameraBatch.
    Args:
        camera_batch_data (dict): A batch of camera parameters from CameraBatch.
        device (torch.device): The device to place the cameras on.
    Returns:
        PerspectiveCameras: PyTorch3D cameras object.
    """
    # Extract parameters
    azim = camera_batch_data['azim'].to(device)  # Azimuth (radians)
    elev = camera_batch_data['elev'].to(device)  # Elevation (radians)
    campos = camera_batch_data['campos'].to(device)  # Camera position

    # Compute rotation matrices
    R = compute_rotation_matrix(azim, elev)

    # Translation is the negative of camera position (campos)
    T = -campos

    # Create PyTorch3D cameras
    cameras = PerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    return cameras

def compute_rotation_matrix(azim, elev):
    """
    Compute a camera rotation matrix from azimuth and elevation angles.
    Args:
        azim (torch.Tensor): Azimuth angles in radians (batch_size,).
        elev (torch.Tensor): Elevation angles in radians (batch_size,).
    Returns:
        torch.Tensor: Rotation matrices (batch_size, 3, 3).
    """
    batch_size = azim.shape[0]

    # Rotation around the Y-axis (azimuth)
    azim_rotation = torch.stack([
        torch.stack([torch.cos(azim), torch.zeros(batch_size), torch.sin(azim)], dim=-1),
        torch.stack([torch.zeros(batch_size), torch.ones(batch_size), torch.zeros(batch_size)], dim=-1),
        torch.stack([-torch.sin(azim), torch.zeros(batch_size), torch.cos(azim)], dim=-1),
    ], dim=1)

    # Rotation around the X-axis (elevation)
    elev_rotation = torch.stack([
        torch.stack([torch.ones(batch_size), torch.zeros(batch_size), torch.zeros(batch_size)], dim=-1),
        torch.stack([torch.zeros(batch_size), torch.cos(elev), -torch.sin(elev)], dim=-1),
        torch.stack([torch.zeros(batch_size), torch.sin(elev), torch.cos(elev)], dim=-1),
    ], dim=1)

    # Combine rotations
    return torch.matmul(elev_rotation, azim_rotation)
def setup_lights(camera_batch_data, device):
    """
    Setup PyTorch3D lights using data from CameraBatch.
    Args:
        camera_batch_data (dict): A batch of camera parameters from CameraBatch.
        device (torch.device): The device to place the lights on.
    Returns:
        PointLights: Configured PyTorch3D lights object.
    """
    lightpos = camera_batch_data['lightpos'].to(device)  # Light positions
    lights = PointLights(
        location=lightpos,
        ambient_color=((0.5, 0.5, 0.5),),  # Example ambient light color
        diffuse_color=((0.8, 0.8, 0.8),),  # Example diffuse light color
        specular_color=((0.3, 0.3, 0.3),),  # Example specular light color
        device=device
    )
    return lights

def project_texture_to_uv_space(generated_texture):
    """
    Projects the generated texture into the UV space of the mesh.
    
    Args:
        generated_texture: 2D texture generated by ControlNet.
        camera: Camera parameters used for the projection.
        
    Returns:
        uv_texture: Texture mapped into UV space.
    """
    # Convert the image texture to a format compatible with UV mapping
    uv_texture = torch.tensor(np.array(generated_texture), dtype=torch.float32) / 255.0  # Normalize
    uv_texture = uv_texture.permute(2, 0, 1)  # Convert to (C, H, W) format for PyTorch

    # Resize to UV space resolution if needed
    uv_texture = torch.nn.functional.interpolate(
        uv_texture.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False
    ).squeeze(0)

    return uv_texture.permute(1, 2, 0)  # Convert back to (H, W, C)
def initialize_texture_map(mesh, device, resolution=512):
    """
    Initializes an empty UV texture map.
    
    Args:
        mesh: The input mesh object.
        device: The device (CPU/GPU) to create the texture on.
        resolution: The resolution of the UV texture map.
        
    Returns:
        texture_map: A blank UV texture map.
    """
    texture_map = torch.zeros((resolution, resolution, 3), device=device)  # RGB texture map
    return texture_map

def update_texture_map(texture_map, generated_texture, uv_coords):
    """
    Updates the UV texture map with a generated texture, backprojected into UV space.

    Args:
        texture_map (torch.Tensor): Current UV texture map (H, W, C).
        generated_texture (torch.Tensor): Texture generated from the pipeline (C, H, W).
        uv_coords (torch.Tensor): UV coordinates normalized to [-1, 1] (N, 2).

    Returns:
        updated_texture_map (torch.Tensor): UV texture map with the new texture applied.
    """
    generated_texture_pytorch3d = generated_texture.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)

    # Ensure UV coordinates are normalized to [-1, 1]
    uv_coords_normalized = 2.0 * uv_coords - 1.0  # Normalize UV coordinates to [-1, 1]

    # Sample the texture map using `grid_sample`
    sampled_texture = torch.nn.functional.grid_sample(
        generated_texture_pytorch3d, 
        uv_coords_normalized.unsqueeze(2), 
        mode="bilinear",
        align_corners=True
    ).squeeze(0).squeeze(-1).permute(1, 0)  # Shape: (N, C)

    # Modify the texture map using the sampled texture
    updated_texture_map = texture_map.clone()
    uv_indices = (uv_coords * texture_map.size(0)).long()  # Map UVs to texture map indices
    updated_texture_map[uv_indices[:, 1], uv_indices[:, 0]] = sampled_texture  # Update UV map

    return updated_texture_map

def create_binary_mask(texture_map):
    """
    Creates a binary mask from the UV texture map to identify painted pixels.
    
    Args:
        texture_map: UV texture map.
        
    Returns:
        binary_mask: Binary mask marking painted pixels (1 for painted, 0 for unpainted).
    """
    binary_mask = (texture_map.sum(dim=-1) > 0).float()  # Non-zero pixels are considered painted
    return binary_mask

def project_binary_mask_to_uv_space(binary_mask, renderer, mesh, camera,  device):
    """
    Projects the binary mask into the UV space of the mesh.
    
    Args:
        binary_mask: Current binary mask.
        camera: Camera parameters.
        mesh: Input mesh.
        device: Device (CPU/GPU).
        
    Returns:
        uv_binary_mask: Binary mask in UV space.
    """
    # Render the binary mask using the current camera parameters
    fragements = renderer.rasterizer(mesh)
    rendered_mask = renderer.shader(fragements, mesh)

    # Convert the rendered mask to UV space
    uv_binary_mask = project_texture_to_uv_space(rendered_mask, camera)
    return (uv_binary_mask > 0).float()

def generate_candidate_views(num_views, resolution=512):
    """
    Generates a set of uniformly distributed candidate views for 3D rendering.

    Args:
        num_views (int): Number of candidate views to generate.
        resolution (int): Resolution of the rendered views.

    Returns:
        candidate_views (list[dict]): List of camera parameters for each candidate view.
    """
    candidate_views = []
    for i in range(num_views):
        # Uniformly sample azimuth and elevation angles
        elev_angle = np.random.uniform(-90, 90)  # Elevation: from -90° to 90°
        azim_angle = np.random.uniform(0, 360)   # Azimuth: from 0° to 360°
        distance = np.random.uniform(2.5, 3.5)  # Distance: example range

        # Append the camera parameters
        candidate_views.append({
            "elev_angle": elev_angle,
            "azim_angle": azim_angle,
            "distance": distance,
            "resolution": resolution
        })

    return candidate_views

def calculate_candidate_masks(binary_mask, candidate_views, input_mesh, device):
    """
    Calculates binary masks for each candidate view based on the current texture map.

    Args:
        binary_mask (torch.Tensor): Current binary mask of the texture (H, W).
        candidate_views (list[dict]): List of camera parameters for candidate views.
        input_mesh (Meshes): 3D mesh object.
        device (torch.device): Device (CPU/GPU).

    Returns:
        candidate_masks (list[torch.Tensor]): List of binary masks for each candidate view.
    """
    candidate_masks = []
    for camera_params in candidate_views:
        # Render the view using the camera parameters
        renderer = setup_renderer(image_size=512, cameras=None, lights=None, device=device)
        rendered_image_fragments = renderer.rasterizer(input_mesh)
        rendered_image = renderer.shader(rendered_image_fragments, input_mesh)

        # Convert rendered image to binary mask (1 for painted regions, 0 for unpainted)
        rendered_binary_mask = (rendered_image.sum(dim=-1) > 0).float()  # Non-zero pixels are painted

        # Combine the current texture's binary mask with the rendered mask
        view_mask = binary_mask + rendered_binary_mask
        view_mask = torch.clamp(view_mask, 0, 1)  # Ensure binary values (0 or 1)

        candidate_masks.append(view_mask)

    return candidate_masks

def select_least_painted_view(candidate_masks):
    """
    Selects the candidate view with the most unpainted pixels.

    Args:
        candidate_masks (list[torch.Tensor]): List of binary masks for each candidate view.

    Returns:
        least_painted_view_idx (int): Index of the least painted view.
    """
    least_painted_pixel_counts = []

    for mask in candidate_masks:
        # Count the number of unpainted pixels (0s in the mask)
        unpainted_pixels = (mask == 0).sum().item()
        least_painted_pixel_counts.append(unpainted_pixels)

    # Find the index of the view with the maximum unpainted pixels
    least_painted_view_idx = int(np.argmax(least_painted_pixel_counts))

    return least_painted_view_idx

def compute_inpainting_loss(texture_map, original_binary_mask, binary_mask):
    """
    Computes the inpainting loss by evaluating smoothness in newly inpainted regions.

    Args:
        texture_map (torch.Tensor): The UV texture map (H, W, C).
        original_binary_mask (torch.Tensor): Binary mask before inpainting (H, W).
        binary_mask (torch.Tensor): Updated binary mask after inpainting (H, W).

    Returns:
        inpainting_loss (torch.Tensor): The computed inpainting loss value.
    """
    # Identify newly inpainted regions
    newly_inpainted = ((original_binary_mask == 0) & (binary_mask == 1)).float()

    # Compute texture gradients for smoothness
    dx = torch.abs(texture_map[:, :-1, :] - texture_map[:, 1:, :])  # Horizontal gradient
    dy = torch.abs(texture_map[:-1, :, :] - texture_map[1:, :, :])  # Vertical gradient

    # Apply the newly inpainted region mask
    inpaint_loss_x = (dx * newly_inpainted[:-1, :, None]).mean()  # Horizontal loss
    inpaint_loss_y = (dy * newly_inpainted[:, :-1, None]).mean()  # Vertical loss

    # Combine the losses
    inpainting_loss = inpaint_loss_x + inpaint_loss_y

    return inpainting_loss

def update_mesh_texture(mesh, texture_map):
    """
    Updates the PyTorch3D Meshes object with the new texture map using differentiable backprojection.

    Args:
        mesh (Meshes): PyTorch3D Meshes object.
        texture_map (torch.Tensor): The UV texture map (H, W, C).

    Returns:
        updated_mesh (Meshes): Meshes object with the new texture applied.
    """
    # Convert texture_map to PyTorch3D-compatible format
    texture_map_pytorch3d = texture_map.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)

    # Ensure UV coordinates are normalized to [-1, 1] for `grid_sample`
    uv_coords = mesh.textures.verts_uvs_padded()  # (B, N, 2)
    uv_coords_normalized = 2.0 * uv_coords - 1.0  # Normalize to [-1, 1]

    # Backproject texture using `grid_sample`
    sampled_texture = torch.nn.functional.grid_sample(
        texture_map_pytorch3d,  # Texture map as a grid
        uv_coords_normalized.unsqueeze(2),  # UV coordinates as grid
        mode="bilinear",
        align_corners=True
    )  # Output shape: (1, C, N, 1)

    # Update the mesh textures
    sampled_texture = sampled_texture.squeeze(0).squeeze(-1).permute(1, 0)  # Shape: (N, C)

    # Modify the texture map using the sampled texture
    updated_texture_map = texture_map.clone()
    uv_indices = (uv_coords * texture_map.size(0)).long()  # Map UVs to texture map indices
    updated_texture_map[uv_indices[:, 1], uv_indices[:, 0]] = sampled_texture  # Apply sampled textures to the UV map

    # Create a new TexturesUV object with the updated texture map
    updated_textures = TexturesUV(
        maps=updated_texture_map.unsqueeze(0).permute(2, 0, 1),  # (H, W, C) -> (1, C, H, W)
        faces_uvs=mesh.textures.faces_uvs_padded(),
        verts_uvs=mesh.textures.verts_uvs_padded()
    )
    updated_mesh = mesh.clone()
    updated_mesh.textures = updated_textures

    return updated_mesh
