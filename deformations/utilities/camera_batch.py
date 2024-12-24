import torch
import numpy as np
import glm
import random
import torchvision.transforms as transforms
from .resize_right import resize
from ..util import persp_proj


class CameraBatch(torch.utils.data.Dataset):
    def __init__(
        self,
        image_resolution,
        distances,
        azimuths,
        elevation_params,
        fovs,
        aug_loc, 
        aug_light,
        aug_bkg,
        bs,
        look_at=[0, 0, 0], up=[0, -1, 0],
        rand_solid=False
    ):

        self.res = image_resolution

        self.dist_min = distances[0]
        self.dist_max = distances[1]

        self.azim_min = azimuths[0]
        self.azim_max = azimuths[1]

        self.fov_min = fovs[0]
        self.fov_max = fovs[1]
        
        self.elev_alpha = elevation_params[0]
        self.elev_beta  = elevation_params[1]
        self.elev_max   = elevation_params[2]

        self.aug_loc   = aug_loc
        self.aug_light = aug_light
        self.aug_bkg   = aug_bkg

        self.look_at = look_at
        self.up = up

        self.batch_size = bs
        self.rand_solid = rand_solid

    def __len__(self):
        return self.batch_size
        
    def __getitem__(self, index):

        elev = np.radians( np.random.beta( self.elev_alpha, self.elev_beta ) * self.elev_max )
        azim = np.radians( np.random.uniform( self.azim_min, self.azim_max+1.0 ) )
        dist = np.random.uniform( self.dist_min, self.dist_max )
        fov = np.random.uniform( self.fov_min, self.fov_max )
        
        proj_mtx = persp_proj(fov)
        
        # Generate random view
        cam_z = dist * np.cos(elev) * np.sin(azim)
        cam_y = dist * np.sin(elev)
        cam_x = dist * np.cos(elev) * np.cos(azim)
        
        if self.aug_loc:

            # Random offset
            limit  = self.dist_min // 2
            rand_x = np.random.uniform( -limit, limit )
            rand_y = np.random.uniform( -limit, limit )

            modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))

        else:
        
            modl = glm.mat4()
            
        view  = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(self.look_at[0], self.look_at[1], self.look_at[2]),
            glm.vec3(self.up[0], self.up[1], self.up[2]),
        )

        r_mv = view * modl
        r_mv = np.array(r_mv.to_list()).T

        mvp     = np.matmul(proj_mtx, r_mv).astype(np.float32)
        campos  = np.linalg.inv(r_mv)[:3, 3]

        if self.aug_light:
            lightpos = self.cosine_sample(campos)*dist
        else:
            lightpos = campos*dist

        if self.aug_bkg:
            bkgs = self.get_random_bg(self.res, self.res, self.rand_solid).squeeze(0)
        else:
            bkgs = torch.ones(self.res, self.res, 3)

        return {
            'mvp': torch.from_numpy( mvp ).float(),
            'lightpos': torch.from_numpy( lightpos ).float(),
            'campos': torch.from_numpy( campos ).float(),
            'bkgs': bkgs,
            'azim': torch.tensor(azim).float(),
            'elev': torch.tensor(elev).float(),
        }
    
    def get_random_bg(self, h, w, rand_solid=False):

        p = torch.rand(1)

        if p > 0.66666:
            if rand_solid:
                background = torch.vstack([
                    torch.full( (1, h, w), torch.rand(1).item()),
                    torch.full( (1, h, w), torch.rand(1).item()),
                    torch.full( (1, h, w), torch.rand(1).item()),
                ]).unsqueeze(0) + torch.rand(1, 3, h, w)
                background = ((background - background.amin()) / (background.amax() - background.amin()))
                background = self.blurs[random.randint(0, 3)](background).permute(0, 2, 3, 1)
            else:
                background =  self.blurs[random.randint(0, 3)]( torch.rand((1, 3, h, w)) ).permute(0, 2, 3, 1)
        elif p > 0.333333:
            size = random.randint(5, 10)
            background = torch.vstack([
                torch.full( (1, size, size), torch.rand(1).item() / 2),
                torch.full( (1, size, size), torch.rand(1).item() / 2 ),
                torch.full( (1, size, size), torch.rand(1).item() / 2 ),
            ]).unsqueeze(0)

            second = torch.rand(3)

            background[:, 0, ::2, ::2] = second[0]
            background[:, 1, ::2, ::2] = second[1]
            background[:, 2, ::2, ::2] = second[2]

            background[:, 0, 1::2, 1::2] = second[0]
            background[:, 1, 1::2, 1::2] = second[1]
            background[:, 2, 1::2, 1::2] = second[2]

            background = self.blurs[random.randint(0, 3)]( resize(background, out_shape=(h, w)) )

            background = background.permute(0, 2, 3, 1)

        else:
            background = torch.vstack([
                torch.full( (1, h, w), torch.rand(1).item()),
                torch.full( (1, h, w), torch.rand(1).item()),
                torch.full( (1, h, w), torch.rand(1).item()),
            ]).unsqueeze(0).permute(0, 2, 3, 1)

        return background

    def cosine_sample(N : np.ndarray) -> np.ndarray:
        """
        #----------------------------------------------------------------------------
        # Cosine sample around a vector N
        #----------------------------------------------------------------------------

        Copied from nvdiffmodelling

        """
        # construct local frame
        N = N/np.linalg.norm(N)

        dx0 = np.array([0, N[2], -N[1]])
        dx1 = np.array([-N[2], 0, N[0]])

        dx = dx0 if np.dot(dx0,dx0) > np.dot(dx1,dx1) else dx1
        dx = dx/np.linalg.norm(dx)
        dy = np.cross(N,dx)
        dy = dy/np.linalg.norm(dy)

        # cosine sampling in local frame
        phi = 2.0*np.pi*np.random.uniform()
        s = np.random.uniform()
        costheta = np.sqrt(s)
        sintheta = np.sqrt(1.0 - s)

        # cartesian vector in local space
        x = np.cos(phi)*sintheta
        y = np.sin(phi)*sintheta
        z = costheta

        # local to world
        return dx*x + dy*y + N*z

