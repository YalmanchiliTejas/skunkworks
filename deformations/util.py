

import torch
import glm
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    ShaderBase,
    TexturesUV
)
from pytorch3d.structures import Meshes
import pymeshlab
import numpy as np


class MaterialMaps:
    def __init__(self, device):
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
        Perform alpha blending on shaded images.

        Args:
            fragments: Rasterization fragments.
            shaded: Shaded colors for each fragment.
            alpha: Alpha values for transparency blending.

        Returns:
            blended_image: The final alpha-blended image.
        """
        alpha = alpha.unsqueeze(-1)  # Ensure alpha has the correct dimensions
        blended_image = (shaded * alpha).sum(dim=-2)  # Blend along faces-per-pixel
        return blended_image

    def forward(self, fragments, meshes, **kwargs):
        texels = meshes.sample_textures(fragments)  # Sample the texture maps

        # Extract material maps
        diffuse_map = texels[..., :3]  # RGB diffuse values
        alpha = texels[..., 3:4] if texels.shape[-1] > 3 else torch.ones_like(texels[..., :1])  # Alpha channel

        # Perform shading (example: simple diffuse shading based on light direction)
        shaded = diffuse_map  # Here you can integrate lighting calculations

        # Blend shaded results to produce the final image
        blended_image = self.blend_images(fragments, shaded, alpha)

        return blended_image

# Helper functions to create texture maps
def create_texture_map(size, device):
    return torch.from_numpy(np.random.uniform(size=[size, size, 3], low=0.0, high=1.0)).float().to(device)

def create_normal_map(size, device):
    return torch.full((size, size, 3), 1.0).float().to(device)

def create_specular_map(size, device):
    return torch.zeros((size, size, 3)).float().to(device)

def get_mesh(mesh_path, output_path, triangulate_flag, device="cuda"):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    if triangulate_flag:
        print('Retriangulating shape...')
        ms.meshing_isotropic_explicit_remeshing()

    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)

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

def setup_renderer(image_size, cameras, lights, device):
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
            lights=lights
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
