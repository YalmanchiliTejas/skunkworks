from Jacobian import SourceMesh, PoissonSystem, MeshProcessor
import torch
from pytorch3d.structures import Meshes
from deformations.util import *



def update_packed_verts(mesh, new_verts):
    """
    Update the vertices of a PyTorch3D Meshes object.

    Parameters:
    mesh (Meshes): The original PyTorch3D mesh object.
    new_verts (torch.Tensor): The updated vertex positions (N, 3).

    Returns:
    Meshes: A new PyTorch3D Meshes object with updated vertices.
    """
    # Ensure the faces and textures remain unchanged
    faces = mesh.faces_packed()
    textures = mesh.textures

    # Create a new Meshes object with updated vertices
    updated_mesh = Meshes(verts=[new_verts], faces=[faces], textures=textures)
    return updated_mesh

def total_traingle_areas(mesh):

    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]


    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
    areas = 0.5 * torch.norm(cross_product, dim=1)
    return torch.sum(areas).item()

def triangle_area_regualrization(mesh):
    return total_traingle_areas(mesh) ** 2



def deformations(input_mesh, target_mesh, epochs):

    #Jacobian Deformation solving for Poisson Optimization problem
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_mesh_loaded = get_mesh(input_mesh, "./input_mesh", False, False, 'input_mesh.obj')
    target_mesh_loaded = get_mesh(target_mesh, "./target_mesh", False, False, 'target_mesh.obj')
    ground_truth = SourceMesh.SourceMesh(0, "./source_mesh",{}, 1 , ttype=torch.float32, use_wks=False, random_centering=False, cpuonly=False)
    ground_truth.load()
    ground_truth.to(device)
    fclip = FashionCLIP('fashion-clip')

    with torch.no_grad():
        ground_truth_jacobians = ground_truth.jacobians_from_vertices(input_mesh_loaded.verts_packed().unsqqueeze(0))
    ground_truth_jacobians.requires_grad_(True)
    optimizer = torch.optim.Adam([ground_truth_jacobians], lr=0.1)

    training_loop = tqdm(range(epochs), leave=False)
    for t in training_loop:
        total_loss = 0
        updated_vertices = ground_truth.vertices_from_jacobians(ground_truth_jacobians).squeeze()
        input_mesh_loaded = update_packed_verts(input_mesh_loaded, updated_vertices)

        target_sample = sample_points_from_meshes(target_mesh_loaded, 10000)
        input_mesh_loaded_sample = sample_points_from_meshes(input_mesh_loaded, 10000)
        chamfer_distance = chamfer_distance(target_sample, input_mesh_loaded_sample)
        mesh_edge_loss = mesh_edge_loss(input_mesh_loaded)
        normal_loss = mesh_normal_consistency(input_mesh_loaded)
        laplacian_loss = mesh_laplacian_smoothing(input_mesh_loaded)

        #INSERT THE FASHIONCLIP EMBEDDING SUPERVISION

        #Jacobian regularization loss:
        jacbian_regularization_loss = (((ground_truth_jacobians - torch.eye(3,3, device=device)) ** 2).mean())
        total_loss += (chamfer_distance + mesh_edge_loss + normal_loss + laplacian_loss + jacbian_regularization_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()






    



    

    






