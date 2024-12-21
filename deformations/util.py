import torch
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes
import pymeshlab
import numpy as np
from utilities.helpers import get_vp_map


def get_mesh(mesh_path, output_path, triangulate_flag, bsdf_flag, mesh_name='mesh.obj'):
    """
    Load and preprocess a mesh using PyTorch3D.

    Parameters:
    mesh_path (str): Path to the input mesh file.
    output_path (Path): Directory for saving temporary results.
    triangulate_flag (bool): Whether to retriangulate the mesh.
    bsdf_flag (bool): Placeholder for BSDF material setup.
    mesh_name (str): Name for the output mesh.

    Returns:
    PyTorch3D Meshes object with processed vertices and textures.
    """
    # Initialize PyMeshLab for preprocessing
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    # Retriangulate the mesh if necessary
    if triangulate_flag:
        print('Retriangulating shape...')
        ms.meshing_isotropic_explicit_remeshing()

    # Ensure UV coordinates exist
    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)

    # Save the processed mesh to a temporary directory
    temp_mesh_path = str(output_path / 'tmp' / mesh_name)
    ms.save_current_mesh(temp_mesh_path)

    # Load the processed mesh with PyTorch3D
    mesh = load_objs_as_meshes([temp_mesh_path], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Normalize the mesh to unit size
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = (verts - center).abs().max()
    mesh = mesh.offset_verts(-center).scale_verts(1.0 / scale)

    # Add a basic vertex-based texture
    verts_features = torch.ones_like(mesh.verts_packed())  # White texture
    mesh.textures = TexturesVertex(verts_features=verts_features)

    return mesh


def get_og_mesh(mesh_path, output_path, triangulate_flag, bsdf_flag, mesh_name='mesh.obj'):
    """
    Load and preprocess the original mesh without resizing.

    Parameters:
    mesh_path (str): Path to the input mesh file.
    output_path (Path): Directory for saving temporary results.
    triangulate_flag (bool): Whether to retriangulate the mesh.
    bsdf_flag (bool): Placeholder for BSDF material setup.
    mesh_name (str): Name for the output mesh.

    Returns:
    PyTorch3D Meshes object with processed vertices and textures.
    """
    # Initialize PyMeshLab for preprocessing
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    # Retriangulate the mesh if necessary
    if triangulate_flag:
        print('Retriangulating shape...')
        ms.meshing_isotropic_explicit_remeshing()

    # Ensure UV coordinates exist
    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)

    # Save the processed mesh to a temporary directory
    temp_mesh_path = str(output_path / 'tmp' / mesh_name)
    ms.save_current_mesh(temp_mesh_path)

    # Load the processed mesh with PyTorch3D
    mesh = load_objs_as_meshes([temp_mesh_path], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Add a basic vertex-based texture
    verts_features = torch.ones_like(mesh.verts_packed())  # White texture
    mesh.textures = TexturesVertex(verts_features=verts_features)

    return mesh


def compute_mv_cl(final_mesh, fe, normalized_clip_render, params_camera, train_rast_map, cfg, device):
    """
    Compute multi-view consistency loss for the deformed mesh.

    Parameters:
    final_mesh (Meshes): Deformed mesh.
    fe (torch.nn.Module): Feature extractor (e.g., CLIP).
    normalized_clip_render (torch.Tensor): Normalized rendered image.
    params_camera (dict): Camera parameters.
    train_rast_map (torch.Tensor): Rasterized map from rendering.
    cfg (dict): Configuration dictionary.
    device (torch.device): Device for computation.

    Returns:
    torch.Tensor: Multi-view consistency loss.
    """
    # Map vertices to pixels
    curr_vp_map = get_vp_map(final_mesh.verts_packed(), params_camera['mvp'], 224)

    # Process each batch for visibility
    for idx, rast_faces in enumerate(train_rast_map[:, :, :, 3].view(cfg.batch_size, -1)):
        u_faces = rast_faces.unique().long()[1:] - 1
        t = torch.arange(len(final_mesh.verts_packed()), device=device)
        u_ret = torch.cat([t, final_mesh.faces_packed()[u_faces].flatten()]).unique(return_counts=True)
        non_verts = u_ret[0][u_ret[1] < 2]
        curr_vp_map[idx][non_verts] = torch.tensor([224, 224], device=device)

    # Convert vertex-to-pixel mapping to patches
    med = (fe.old_stride - 1) / 2
    curr_vp_map[curr_vp_map < med] = med
    curr_vp_map[(curr_vp_map > 224 - fe.old_stride) & (curr_vp_map < 224)] = 223 - med
    curr_patch_map = ((curr_vp_map - med) / fe.new_stride).round()
    flat_patch_map = curr_patch_map[..., 0] * (((224 - fe.old_stride) / fe.new_stride) + 1) + curr_patch_map[..., 1]

    # Extract deep features
    patch_feats = fe(normalized_clip_render)
    flat_patch_map[flat_patch_map > patch_feats[0].shape[-1] - 1] = patch_feats[0].shape[-1]
    flat_patch_map = flat_patch_map.long()[:, None, :].repeat(1, patch_feats[0].shape[1], 1)

    deep_feats = patch_feats[cfg.consistency_vit_layer]
    deep_feats = torch.nn.functional.pad(deep_feats, (0, 1))
    deep_feats = torch.gather(deep_feats, dim=2, index=flat_patch_map)
    deep_feats = torch.nn.functional.normalize(deep_feats, dim=1, eps=1e-6)

    # Angular consistency
    elev_d = torch.cdist(params_camera['elev'].unsqueeze(1), params_camera['elev'].unsqueeze(1)).abs() < torch.deg2rad(
        torch.tensor(cfg.consistency_elev_filter))
    azim_d = torch.cdist(params_camera['azim'].unsqueeze(1), params_camera['azim'].unsqueeze(1)).abs() < torch.deg2rad(
        torch.tensor(cfg.consistency_azim_filter))

    cosines = torch.einsum('ijk, lkj -> ilk', deep_feats, deep_feats.permute(0, 2, 1))
    cosines = (cosines * azim_d.unsqueeze(-1) * elev_d.unsqueeze(-1)).permute(2, 0, 1).triu(1)
    consistency_loss = cosines[cosines != 0].mean()
    return consistency_loss
