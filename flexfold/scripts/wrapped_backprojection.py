




import torch 
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt
from flexfold.fsc import fourier_shell_correlation, fsc_auc,fsc_thresh, spherical_soft_mask, fourier_mask
from cryodrgn.mrcfile import parse_mrc, write_mrc
import numpy as np
from Bio.PDB import PDBParser, Superimposer, is_aa
from io import StringIO
from openfold.utils.tensor_utils import tensor_tree_map
from flexfold.core import struct_to_pdb
import pickle 
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import os
import glob
import time
from flexfold import dataset
from flexfold.pose import PoseTracker
from flexfold.models import HetOnlyVAE, struct_to_crd
from flexfold.lattice import Lattice
from cryodrgn import __version__, ctf
from flexfold.core import ifft2_center, unsymmetrize_ht, fft3center, ifft3center, dcd2numpyArr, numpyArr2dcd
from cryodrgn.fft import fft2_center, symmetrize_ht
from cryodrgn.mrcfile import parse_mrc, write_mrc
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import tqdm

def rotateVolume( V, R, align_corners=True):
        # Input shape: R [B, 3, 3]
        B = R.shape[0]
        D = V.shape[-1]

        # Convert 3x3 to 3x4 affine matrices by appending zero translation
        R_aff = torch.cat(
            (R, torch.zeros(B, 3, 1, device=V.device, dtype=R.dtype)),
            dim=2
        )  # [B,3,4]

        # Pytorch expects transform as (batch, 3, 4)
        # but affine_grid wants shape (B, C, D+1, D+1, D+1)
        grid = torch.nn.functional.affine_grid(
            R_aff,
            size=(B, 1, D, D, D),
            align_corners=align_corners
        )

        # Rotate volume
        V_rot = F.grid_sample(
            V,
            grid,
            mode='bilinear',
            align_corners=align_corners
        )

        return V_rot


def ctf_from_params(ctf_param, lattice):
    B = ctf_param.shape[0]
    D = lattice.D
    freqs = lattice.freqs2d.unsqueeze(0).expand(
        B, *lattice.freqs2d.shape
    ) / ctf_param[:, 0].view(B, 1, 1)
    c = ctf.compute_ctf(freqs, *torch.split(ctf_param[:, 1:], 1, 1)).view(
        B, D, D
    )
    return c
def circular_mask(D, R=None, device='cuda'):
    """
    Create a circular mask of diameter D and radius R (default D//2)
    """
    if R is None:
        R = D // 2

    coords = torch.arange(D, device=device) - (D-1)/2
    Y, X = torch.meshgrid(coords, coords, indexing='ij')
    dist2 = X*X + Y*Y
    mask = (dist2 <= R*R).float()
    return mask

def unsymmetrize_ht3(x: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1]
    assert D % 2 != 0
    return x[..., 0:-1, 0:-1, 0:-1] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################################################################""
# Inputs
###################################################################################################""
epoch=99
run_dir = "./data/cryofold/AKMD/snr1/run/"
config_file = "%s/config.yaml"%run_dir
weight_file = "%s/weights.%i.pkl"%(run_dir, epoch)
z_file =  "%s/z.%i.pkl"%(run_dir, epoch)
particles_file = "./data/cryofold/AKMD/snr1/Particles/particles.mrcs"
poses_file = "./data/cryofold/AKMD/snr1/particles.pkl"
ctf_file = "./data/cryofold/AKMD/snr1/ctf.pkl"


cfg = config.load(config_file)
D = cfg["lattice_args"]["D"]  # image size + 1
zdim = cfg["model_args"]["zdim"]
norm = [float(x) for x in cfg["dataset_args"]["norm"]]

model, lattice = HetOnlyVAE.load(cfg,weight_file, device=device)
model = model.to(device)
model.eval()
z = torch.tensor(load_pkl(z_file)).to(device)

###################################################################################################""
# Load Images and poses and CTF
###################################################################################################""
imageDataset = dataset.ImageDataset(mrcfile=particles_file,device=device)
D = imageDataset.D
N = imageDataset.N
Dr = D-1

# Lattice
lattice = Lattice(D, extent=0.5).to(device)
enc_mask = lattice.get_circular_mask(D // 2)

# Poses
posetracker = PoseTracker.load(poses_file, imageDataset.N, D, None, None,).to(device)

# CTF
ctf_params = torch.tensor(ctf.load_ctf_for_training(D - 1,ctf_file)).to(device)

# test
# particles_ft, particles, i = next(iter(imageDataset))


###################################################################################################""
# PDB_DCD
###################################################################################################""
# batch_size=16

# with torch.no_grad():

#     struct0 = model.decoder.structure_decoder(z.mean(dim=-2, keepdim=True))
#     struct0["final_atom_positions"] = struct0["final_atom_positions"] @ model.decoder.rot_init+ model.decoder.trans_init[..., None, :]
#     crd0,coefs = struct_to_crd(struct0, ca=False, coefs=model.decoder.atom_coefs)
#     struct0 = tensor_tree_map(lambda x: x[-1].detach().cpu().numpy(), struct0)
#     struct_to_pdb(struct0, run_dir+"reference.pdb")
#     torch.save(coefs.cpu(), run_dir+"coefs.pt")

#     N_atoms = crd0.shape[-2]

#     trajectory = np.zeros((N, N_atoms, 3))
#     for start in range(0, N, batch_size):
#         print(start)
#         end = min(start + batch_size, N)
#         B = end - start
#         z_val = z[range(start, end)]

#         struct = model.decoder.structure_decoder(z_val)
#         struct["final_atom_positions"] = struct["final_atom_positions"] @ model.decoder.rot_init+ model.decoder.trans_init[..., None, :]
#         crd,coefs = struct_to_crd(struct, ca=False, coefs=model.decoder.atom_coefs)
#         trajectory[start:end] = crd.cpu().numpy()

# numpyArr2dcd(np.concatenate((trajectory, crd0.detach().cpu().numpy()),  axis=0  ), run_dir+"reference.dcd")
# # ###################################################################################################""
# # Reconstruction
# ###################################################################################################""
# dcd = dcd2numpyArr( run_dir+"reference.dcd")

# batch_size=64
# volume = torch.zeros(Dr, Dr, Dr, device=device)
# volume_reg = torch.zeros(Dr, Dr, Dr, device=device)

# for start in range(0, N, batch_size):

#     # Batching
#     print(start)
#     end = min(start + batch_size, N)
#     B = end - start
#     particles_ft,_, i = imageDataset[range(start, end)]
#     particles_ft=particles_ft.to(device)
#     particles_ft*= -1
#     rot, tran = posetracker.get_pose(range(start, end))
#     c = ctf_from_params(ctf_params[start:end], lattice)

#     # Apply CTF
#     y=particles_ft * c

#     # Shift 
#     y = torch.view_as_complex(lattice.translate_ft(torch.view_as_real(y).view(B, D*D, 2), tran.unsqueeze(1)).view(B, D, D, 2))

#     # Backproject image  
#     y_real = ifft2_center(unsymmetrize_ht(y)).real
#     y_real_expand = y_real.unsqueeze(1).expand(B,Dr,Dr,Dr)
#     recon  = rotateVolume(y_real_expand.unsqueeze(1), rot).squeeze(1)
#     recon /= N
#     volume += torch.sum(recon, dim=0)

#     # Backproject CTF  
#     ctf_real = ifft2_center(unsymmetrize_ht(c)).real
#     ctf_real = ctf_real.unsqueeze(-3)
#     ctf_real_expand = ctf_real.expand(B, Dr, Dr,Dr)
#     ctf_recon = rotateVolume(ctf_real_expand.unsqueeze(1), rot).squeeze(1)
#     ctf_recon =  fft3center(ctf_recon)
#     ctf_recon /= N
#     volume_reg +=  torch.sum(torch.abs(ctf_recon) ** 2, dim=0).real

# volume_reg_floor = 1e-5 *volume_reg.mean()
# volume_reg = torch.maximum(volume_reg,volume_reg_floor * torch.ones_like(volume_reg))

# volume_ft = fft3center(volume)
# volume_ft_reg = volume_ft / (volume_reg)
# volume_reg = ifft3center(volume_ft_reg).real

# write_mrc("./data/cryofold/AKMD/snr1/particles_center_phase_flipped_backproj.mrc",volume_reg, is_vol=True)
# # write_mrc("./data/cryofold/AKMD/snr0.01/particles_center_phase_flipped.mrcs", out, is_vol=False)
# #cryodrgn backproject_voxel ./data/cryofold/AKMD/snr1/particles_center_phase_flipped.mrcs --poses ./data/cryofold/AKMD/snr1/particles.pkl --uninvert-data -o ./data/cryofold/AKMD/snr1/backproject_phase_flipped



from flexfold.core import get_voxel_mask, vol_real_mask

def warp_particle_volume_to_canonical(Vp, D_field, align_corners=True):
    """
    Vp: [B,D,D,D] particle volume (real-space)
    D_field: [B,D,D,D,3] deformation field mapping canonical->(r0 - r)  (voxel units)
    returns: Vw [B,D,D,D] warped volume in canonical grid
    """
    device = Vp.device
    B,_,_, D = Vp.shape

    # source (particle) coords per canonical voxel: x_p = x0 - D_field(x0)
    coords = torch.linspace(-(D-1)/2, (D-1)/2, D, device=device)
    Z, Y, X = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid0 = torch.stack([X, Y, Z], dim=-1).unsqueeze(0)  # [1, D, D, D, 3]
    grid = grid0 - D_field  # [B,D,D,D,3] in voxel units

    # normalized: (coord + (D-1)/2)/(D-1) * 2 - 1
    # [B,D,D,D,3]
    grid = (grid + (D - 1) / 2.0) * ( 2.0 / (D - 1)) - 1.0 

    # grid_sample expects input shape [N,C,D,H,W] and grid [N, outD, outH, outW, 3]
    Vw = F.grid_sample(Vp.unsqueeze(1), grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)

    return Vw.squeeze(1)  # [B,D,D,D]

def push_particle_volume_to_canonical(Vp, D_field):
    """
    Vp: [B, D, D, D] particle volume
    D_field: [B, D, D, D, 3] displacement field in voxel units
    Returns:
        V_canonical: [B, D, D, D]
    """
    B, D, _, _ = Vp.shape
    device = Vp.device

    # canonical voxel indices
    coords = torch.linspace(-(D-1)/2, (D-1)/2, D, device=device)
    Z, Y, X = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid = torch.stack([X, Y, Z], dim=-1).float()  # [D,D,D,3]
    grid = grid.unsqueeze(0).expand(B, D, D, D, 3)  # [B,D,D,D,3]

    # compute canonical positions
    canonical_pos = grid + D_field  # [B,D,D,D,3]
    canonical_pos = (canonical_pos + (D - 1) / 2.0) 
    

    # flatten
    Vp_flat = Vp.view(B, -1)  # [B, D^3]
    pos_flat = canonical_pos.view(B, -1, 3)  # [B, D^3, 3]

    # integer voxel indices and clamping
    ix = pos_flat[..., 2].long().clamp(0, D-1)
    iy = pos_flat[..., 1].long().clamp(0, D-1)
    iz = pos_flat[..., 0].long().clamp(0, D-1)

    # compute flat indices in canonical volume
    flat_idx = ix * D * D + iy * D + iz  # [B, D^3]

    # allocate output
    V_canonical = torch.zeros(B, D*D*D, device=device, dtype=Vp.dtype)

    # scatter-add
    for b in range(B):
        V_canonical[b].scatter_add_(0, flat_idx[b], Vp_flat[b])

    # reshape back
    V_canonical = V_canonical.view(B, D, D, D)
    return V_canonical



def deformation_field(crd_init, crd_target,vox_loc_init, vox_mask_init, grid_size, pixel_size, coef=None):
    """
    crd_init : [1, N_atoms, 3]
    crd_target : [B, N_atoms, 3]
    vox_loc_init : [1, N_atoms, P, 3]
    vox_mask_init : [1, N_atoms, P]
    """
    _, N_atoms, P = vox_mask_init.shape
    B = crd_target.shape[0]

    # Compute Gaussian contributions
    xyz = (vox_loc_init - grid_size / 2 + 0.5) * pixel_size  # (1, N_atoms, P, 3)
    sq_dist = torch.sum((xyz - crd_init[:, :, None, : ]) ** 2, dim=-1)    # (1, N_atoms, P)
    G = torch.exp(-0.5 * sq_dist / sigma**2) * vox_mask_init  # (1, N_atoms, P)

    if coef is not None : 
        G *= coef[..., None]

    # Wrap : sum(G*dr) / sum(G)
    delta_crd = crd_init-crd_target  # (B, N_atoms, P, 3)
    num = G[..., None] * delta_crd[..., None, :] # (B, N_atoms, P, 3)
    denom = G.unsqueeze(-1) # (1, N_atoms, P, 3)

    # Flatten for scatter
    flat_idx = vox_loc_init.reshape(1, -1, 3).long() # (1, N_atoms * P, 3)
    flat_idx = flat_idx[..., 0] * (grid_size**2) + flat_idx[..., 1] * grid_size + flat_idx[..., 2]  
    flat_idx = flat_idx.unsqueeze(-1)
    flat_idx_num =  flat_idx.expand(B, N_atoms*P, 3)

    # Scatter into flat vol buffer [B, N_atoms * P ] --> [B, D * D * D] 
    num_dense = torch.zeros(B, grid_size **3, 3 , device=crd_init.device, dtype=crd_init.dtype)
    denom_dense = torch.zeros(1, grid_size **3, 1, device=crd_init.device, dtype=crd_init.dtype)

    num_dense = num_dense.scatter_add(1, flat_idx_num, num.view(B,-1,3))
    num_dense = num_dense.view(B, grid_size, grid_size,grid_size, 3)

    denom_dense = denom_dense.scatter_add(1, flat_idx, denom.view(1,-1,1))
    denom_dense = denom_dense.view(1, grid_size, grid_size,grid_size, 1)

    D_field = num_dense/(denom_dense + eps)

    return torch.permute(D_field, (0,3,2,1, 4)) # permute to match volume data ordering Z, Y, X


def map_to_canonical(vol_target, crd_target, vox_loc_target, vox_mask_target, vox_loc_canonical, pixel_size, coef=None, eps=1e-8):
    """
    Vol_target : [B, N, N, N]

    crd_target : [B, N_atoms, 3]
    vox_loc_target : [B, N_atoms, P, 3]
    vox_mask_target : [B, N_atoms, P]

    vox_loc_canonical : [1, N_atoms, P, 3]
    """
    B, N_atoms, P, _ = vox_loc_target.shape
    grid_size = vol_target.shape[-1]

    # Compute Gaussian contributions of targets
    xyz = (vox_loc_target - grid_size / 2 + 0.5) * pixel_size  # (B, N_atoms, P, 3)
    sq_dist = torch.sum((xyz - crd_target[:, :, None, : ]) ** 2, dim=-1)    # (B, N_atoms, P)
    G = torch.exp(-0.5 * sq_dist / sigma**2) * vox_mask_target  # (B, N_atoms, P)

    if coef is not None : 
        G *= coef[None, :, None]

    G = G.view(B, -1)

    # linear indices in flattened volume (z * D^2 + y * D + x)
    def flat_idx(idx):
        return idx[...,0]*grid_size**2 + idx[...,1]*grid_size + idx[...,2]


    flat_idx_target = flat_idx(vox_loc_target.view(B, -1, 3).long())   # (B, N_atoms*P)
    flat_idx_canonical = flat_idx(vox_loc_canonical.view(1, -1, 3).long())   # (1, N_atoms*P)
    flat_idx_canonical = flat_idx_canonical.expand(B, N_atoms*P)

    # ---- sample from deformed ----
    # (B, N_atoms*P)
    vol_target = torch.permute(vol_target, (0,3,2,1))
    vol_flat = vol_target.reshape(B, -1)
    vox_vals = torch.gather(vol_flat , 1, flat_idx_target)

    # weighted contributions
    vox_vals = G * vox_vals                      # (B, N_atoms*P)

    # ---- scatter to canonical ----
    vol_canon =  torch.zeros(B, grid_size **3, device=vol_target.device, dtype=vol_target.dtype)
    norm      =  torch.zeros(B, grid_size **3, device=vol_target.device, dtype=vol_target.dtype)

    vol_canon= vol_canon.scatter_add(1, flat_idx_canonical, vox_vals)
    norm= norm.scatter_add(1, flat_idx_canonical, G)

    vol_canon = vol_canon / (norm + eps)

    vol_canon = vol_canon.view(B, grid_size, grid_size, grid_size)
    
    # permute to match volume data ordering Z, Y, X
    vol_canon = torch.permute(vol_canon, (0,3,2,1))

    return vol_canon



import torch
import torch.nn.functional as F

def compute_jacobian_det(disp):
    j = jacobian(disp)
    return torch.linalg.det(j)

def grad_central(u):
    # u: [B, 1, D, H, W]
    kx = torch.tensor([[-0.5, 0, 0.5]], device=u.device).reshape(1,1,1,1,3)
    ky = torch.tensor([[-0.5, 0, 0.5]], device=u.device).reshape(1,1,1,3,1)
    kz = torch.tensor([[-0.5, 0, 0.5]], device=u.device).reshape(1,1,3,1,1)

    dx = F.conv3d(u, kx, padding=(0,0,1))
    dy = F.conv3d(u, ky, padding=(0,1,0))
    dz = F.conv3d(u, kz, padding=(1,0,0))
    return dx, dy, dz

def jacobian(disp):  # disp: [B, D, H, W, 3]
    disp = disp.permute(0,4,1,2,3)
    ux, uy, uz = disp[:,[0]], disp[:,[1]], disp[:,[2]]

    ux_x, ux_y, ux_z = grad_central(ux)
    uy_x, uy_y, uy_z = grad_central(uy)
    uz_x, uz_y, uz_z = grad_central(uz)

    J = torch.stack([
        torch.stack([1 + ux_x,     ux_y,     ux_z], dim=-1),
        torch.stack([    uy_x, 1 + uy_y,     uy_z], dim=-1),
        torch.stack([    uz_x,     uz_y, 1 + uz_z], dim=-1),
    ], dim=-2) # [B,D,H,W,3,3]

    return J.squeeze(1)

dcd = dcd2numpyArr( run_dir+"reference.dcd")

# crd0 = torch.tensor(dcd[[-1]], device=device)
# crd1 = torch.tensor(dcd[0:10], device=device)
# dcrd = crd0 - crd1
# crd = crd0
# pixel_size=1.0
# sigma = 1.0
# grid_size = Dr
# eps = 1e-8
# n_pix_cutoff = 19


# vox_loc0, vox_mask0 = get_voxel_mask(crd0, grid_size, pixel_size,  n_pix_cutoff)
# vol0 = vol_real_mask(crd0, vox_loc0, vox_mask0, grid_size, sigma, pixel_size)
# vox_loc1, vox_mask1 = get_voxel_mask(crd1, grid_size, pixel_size,  n_pix_cutoff)
# vol1 = vol_real_mask(crd1, vox_loc1, vox_mask1, grid_size, sigma, pixel_size)


# vol_mask = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool, device=device)
# coords = vox_loc0.reshape(-1, 3 ).long()
# vol_mask[coords[:,2], coords[:,1], coords[:,0]] = True

# coefs =  torch.load(run_dir+"coefs.pt").to(device)


# deformation = deformation_field(crd0, crd1, vox_loc_init=vox_loc0, vox_mask_init=vox_mask0, pixel_size=pixel_size, grid_size=grid_size,coef=coefs)
# det = compute_jacobian_det(deformation)

# vol_wrapped = warp_particle_volume_to_canonical(vol1, deformation)
# det = (det/ (3**3)).clamp_min(1e-6) 


# vol_push = push_particle_volume_to_canonical(vol1.contiguous(), deformation)
# # vol_wrapped *= vol_mask.unsqueeze(0)

# vol_mapped = map_to_canonical(
#     vol_target=vol1, 
#     vox_loc_canonical= vox_loc0, 
#     crd_target = crd1, 
#     vox_loc_target = vox_loc1, 
#     vox_mask_target = vox_mask1, 
#     pixel_size=pixel_size, 
#     coef=coefs
# )

# write_mrc("./data/cryofold/AKMD/snr1/backproject_deformed/test_1.mrc",(vol_mapped)[0], is_vol=True)
# write_mrc("./data/cryofold/AKMD/snr1/backproject_deformed/test_gt.mrc",(vol0)[0], is_vol=True)


# ###################################################################################################""
# Reconstruction + Deformation
###################################################################################################""
dcd = dcd2numpyArr( run_dir+"reference.dcd")
coefs =  torch.load(run_dir+"coefs.pt").to(device)
output_prefix = "./data/cryofold/AKMD/snr1/backproject_deformed/"
pixel_size=1.0
sigma = 1.0
grid_size = Dr
eps = 1e-8
n_pix_cutoff = 29

batch_size=8
volume = torch.zeros(Dr, Dr, Dr, device=device)
volume_reg = torch.zeros(Dr, Dr, Dr, device=device)
volume_half1 = torch.zeros(Dr, Dr, Dr, device=device)
volume_reg_half1 = torch.zeros(Dr, Dr, Dr, device=device)
volume_half2 = torch.zeros(Dr, Dr, Dr, device=device)
volume_reg_half2 = torch.zeros(Dr, Dr, Dr, device=device)

crd_canon = torch.tensor(dcd[[-1]], device=device)
traj = torch.tensor(dcd[:-1], device=device)
print(traj.shape)

vox_mask_canon = get_voxel_mask(crd_canon, grid_size, pixel_size,  n_pix_cutoff)
vol_canon = vol_real_mask(crd_canon, vox_mask_canon[0], vox_mask_canon[1], grid_size, sigma, pixel_size)
vol_mask = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool, device=device)
coords = vox_mask_canon[0].reshape(-1, 3 ).long()
vol_mask[coords[:,2], coords[:,1], coords[:,0]] = True


vox_mask_canon_half = get_voxel_mask(crd_canon, grid_size, pixel_size,  n_pix_cutoff//2+1)
vol_mask_half = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool, device=device)
coords = vox_mask_canon_half[0].reshape(-1, 3 ).long()
vol_mask_half[coords[:,2], coords[:,1], coords[:,0]] = True

vox_mask_canon_tight = get_voxel_mask(crd_canon, grid_size, pixel_size,  5)
vol_mask_tight = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.bool, device=device)
coords = vox_mask_canon_tight[0].reshape(-1, 3 ).long()
vol_mask_tight[coords[:,2], coords[:,1], coords[:,0]] = True

for ii in tqdm.tqdm(range(N// batch_size)):

    # Batching
    start = ii*batch_size
    end = min(start + batch_size, N)
    B = end - start
    particles_ft,_, i = imageDataset[start:end]
    particles_ft=particles_ft.to(device)
    rot, tran = posetracker.get_pose(range(start, end))
    c = ctf_from_params(ctf_params[start:end], lattice)
    crd_traj = traj[start: end]

    # Inverse particle
    particles_ft*= -1


    # Apply CTF
    y=particles_ft * c

    # Shift 
    y = torch.view_as_complex(lattice.translate_ft(torch.view_as_real(y).view(B, D*D, 2), tran.unsqueeze(1)).view(B, D, D, 2))

    # Backproject image  
    y_real = ifft2_center(unsymmetrize_ht(y)).real
    recon = y_real.unsqueeze(1).expand(B,Dr,Dr,Dr)

    # deform
    # D_field = deformation_field(crd_canon, crd_traj, vox_loc_init=vox_mask_canon[0], vox_mask_init=vox_mask_canon[1], pixel_size=pixel_size, grid_size=grid_size)
    # D_field_det = compute_jacobian_det(D_field)
    # D_field_det = (D_field_det).clamp_min(1e-6) 
    # # Wrap and rotate
    # recon = warp_particle_volume_to_canonical(recon, D_field)
    # recon = recon / (D_field_det)
    recon  = rotateVolume(recon.unsqueeze(1), rot).squeeze(1)

    vox_mask_recon = get_voxel_mask(crd_traj, grid_size, pixel_size,  n_pix_cutoff)
    recon = map_to_canonical(
        vol_target=recon, 
        vox_loc_canonical= vox_mask_canon[0], 
        crd_target = crd_traj, 
        vox_loc_target = vox_mask_recon[0], 
        vox_mask_target = vox_mask_recon[1], 
        pixel_size=pixel_size, 
        coef=coefs
    )

    recon /= N
    recon = torch.sum(recon, dim=0)

    # CTF real space  
    ctf_real = ifft2_center(unsymmetrize_ht(c)).real
    ctf_real = ctf_real.unsqueeze(-3)
    ctf_recon = ctf_real.expand(B, Dr, Dr,Dr)

    # Wrap and rotate
    ctf_recon = rotateVolume(ctf_recon.unsqueeze(1), rot).squeeze(1)
    # ctf_recon = warp_particle_volume_to_canonical(ctf_recon, D_field)
    # ctf_recon = ctf_recon / (D_field_det )
    ctf_recon_tmp = map_to_canonical(
        vol_target=ctf_recon, 
        vox_loc_canonical= vox_mask_canon[0], 
        crd_target = crd_traj, 
        vox_loc_target = vox_mask_recon[0], 
        vox_mask_target = vox_mask_recon[1], 
        pixel_size=pixel_size, 
        coef=coefs
    )

    
    ctf_recon[:,vol_mask] = ctf_recon_tmp[:,vol_mask]


    # Wrap and rotate
    ctf_recon =  fft3center(ctf_recon)
    ctf_recon /= N
    ctf_recon =  torch.sum(torch.abs(ctf_recon) ** 2, dim=0).real

    volume += recon
    volume_reg += ctf_recon
    if ii%2 == 0:
        volume_half1 += recon
        volume_reg_half1 += ctf_recon
    else:
        volume_half2 += recon
        volume_reg_half2 += ctf_recon


def regularize_volume(volume, volume_reg):
    volume_reg_floor = 1e-5 *volume_reg.mean()
    volume_reg = torch.maximum(volume_reg,volume_reg_floor * torch.ones_like(volume_reg))

    volume_ft = fft3center(volume)
    volume_ft_reg = volume_ft / (volume_reg)
    return ifft3center(volume_ft_reg).real


volume = regularize_volume(volume, volume_reg)
volume_half1 = regularize_volume(volume_half1, volume_reg_half1)
volume_half2 = regularize_volume(volume_half2, volume_reg_half2)




write_mrc(output_prefix+"/backproject_unmasked.mrc",volume, is_vol=True)
write_mrc(output_prefix+"/backproject_half1_unmasked.mrc",volume_half1, is_vol=True)
write_mrc(output_prefix+"/backproject_half2_unmasked.mrc",volume_half2, is_vol=True)

volume *= vol_mask_tight.float()
volume_half1 *= vol_mask_tight.float()
volume_half2 *= vol_mask_tight.float()


write_mrc(output_prefix+"/backproject.mrc",volume, is_vol=True)
write_mrc(output_prefix+"/backproject_half1.mrc",volume_half1, is_vol=True)
write_mrc(output_prefix+"/backproject_half2.mrc",volume_half2, is_vol=True)
write_mrc(output_prefix+"/backproject_mask.mrc",vol_mask_tight.float(), is_vol=True)
# write_mrc("./data/cryofold/AKMD/snr0.01/particles_center_phase_flipped.mrcs", out, is_vol=False)
#cryodrgn backproject_voxel ./data/cryofold/AKMD/snr1/particles_center_phase_flipped.mrcs --poses ./data/cryofold/AKMD/snr1/particles.pkl --uninvert-data -o ./data/cryofold/AKMD/snr1/backproject_phase_flipped




# ind = 1

# write_mrc("./data/cryofold/AKMD/snr1/test0.mrc",vol0[-1].cpu())
# write_mrc("./data/cryofold/AKMD/snr1/test1.mrc",vol1[ind].cpu())
# write_mrc("./data/cryofold/AKMD/snr1/test2.mrc",vol_wrapped[ind].cpu())
# write_mrc("./data/cryofold/AKMD/snr1/test3.mrc",vol_mask.float().cpu())

# import matplotlib.pyplot as plt

# D_field = deformation[ind].cpu()
# D = D_field.shape[0]

# axis = 1
# D_field = D_field.sum(axis=-(1+axis))

# Ux = D_field[...,  0] # X displacements
# Uy = D_field[...,  1]  # Y displacements

# # Create grid for quiver
# x = torch.arange(D)
# y = torch.arange(D)
# X, Y = torch.meshgrid(x, y, indexing='xy')

# # Downsample to avoid clutter
# fig, ax = plt.subplots(1,4,figsize=(40,10), layout="constrained")
# ax[0].quiver(X, Y,
#            Ux, Uy, width = 0.001)

# ax[0].set_title("Deformation Field (XY sum)")
# ax[0].invert_yaxis()      # match image coordinate convention
# ax[0].set_xlabel("X")
# ax[0].set_ylabel("Y")
# ax[0].axis('equal')

# ax[1].imshow(vol0[-1].sum(dim=-axis).cpu())
# ax[2].imshow(vol1[ind].sum(dim=-axis).cpu())
# ax[3].imshow(vol_wrapped[ind].sum(dim=-axis).cpu())
# ax[1].set_title("Canonical")
# ax[2].set_title("Deformed")
# ax[2].set_title("Wrapped")

# fig.savefig("./data/cryofold/AKMD/snr1/test.png", dpi=300)