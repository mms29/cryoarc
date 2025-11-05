from scipy.spatial.transform import Rotation
from cryodrgn import fft
from cryodrgn.mrcfile import write_mrc, parse_mrc

import torch
import numpy as np

import tqdm

from openfold.np import residue_constants, protein
from openfold.utils.tensor_utils import tensor_tree_map
from cryodrgn.fft import fftshift
from torch.fft import ifft2
import  openfold.np.residue_constants as rc


import matplotlib.pyplot as plt
import pandas as pd


def parse_lightning_csv(path):
    df = pd.read_csv(path)

    # Always keep step + epoch columns if present
    base_cols = [c for c in ["step", "epoch"] if c in df.columns]

    # Classify metrics
    step_metrics  = [c for c in df.columns if c.endswith("_step")]
    epoch_metrics = [c for c in df.columns if c.endswith("_epoch")]

    # Separate val vs train
    val_step_cols   = [c for c in step_metrics if c.startswith("val")]
    train_step_cols = [c for c in step_metrics if c not in val_step_cols]

    val_epoch_cols   = [c for c in epoch_metrics if c.startswith("val")]
    train_epoch_cols = [c for c in epoch_metrics if c not in val_epoch_cols]

    # Build DataFrames (dropping rows where all relevant metrics are NaN)
    train_step_df  = df[base_cols + train_step_cols].dropna(how="all", subset=train_step_cols).reset_index(drop=True)
    train_epoch_df = df[base_cols + train_epoch_cols].dropna(how="all", subset=train_epoch_cols).reset_index(drop=True)
    val_step_df    = df[base_cols + val_step_cols].dropna(how="all", subset=val_step_cols).reset_index(drop=True)
    val_epoch_df   = df[base_cols + val_epoch_cols].dropna(how="all", subset=val_epoch_cols).reset_index(drop=True)

    return train_step_df, train_epoch_df, val_step_df, val_epoch_df

def plot_loss(infile, outfile):
    train_step, train_epoch, val_step, val_epoch = parse_lightning_csv(infile)

    movavg = lambda arr,w: np.convolve(
        np.nan_to_num(arr), np.ones(w), 'valid'
    ) / np.convolve(~np.isnan(arr), np.ones(w), 'valid')
    movavg_step = lambda arr,w: arr[:-(w-1)]*(len(arr)/(len(arr)-w))

    losses=["data_loss", "chi_loss", "viol_loss","pose_rot", "kld", "loss", "scale_loss"]
    col = "tab:blue"
    valcol = "tab:green"
    nrows = 2
    ncols=4
    fig, ax = plt.subplots(nrows,ncols, figsize=(20,10), layout="constrained")
    for x in range(nrows):
        for y in range(ncols):
            ii = x *ncols + y
            if ii>=len(losses):
                break

            print(losses[ii])

            if not losses[ii]+"_step" in train_step:
                break

            loss = train_step[losses[ii]+"_step"]
            step = train_step["step"] * (train_step["epoch"].max() - train_step["epoch"].min()) / (train_step["step"].max() - train_step["step"].min())
            if len(step)>50:
                w = min(len(step)//10,10)
                ax[x,y].plot(step, loss, alpha=0.5, c=col)
                ax[x,y].plot(movavg_step(step,w), movavg(loss,w), label = "training", c=col)
            else:
                ax[x,y].plot(step, loss, label = "training", c=col)

            ax[x,y].set_xlabel("epoch")
            ax[x,y].set_ylabel(losses[ii])

            if ("val_" + losses[ii]+"_epoch") in val_epoch:
                loss = val_epoch["val_" +losses[ii]+"_epoch"]
                step = val_epoch["epoch"]
                if len(step)>50:
                    w = min(len(step)//10,10)
                    ax[x,y].plot(step, loss, alpha=0.5, c=valcol)
                    ax[x,y].plot(movavg_step(step,w), movavg(loss,w), label = "validation", c=valcol)
                else:
                    ax[x,y].plot(step, loss, label = "validation", c=valcol)

    ax[0,0].legend()
    fig.savefig(outfile, dpi=300)

    plt.close(fig)


def output_single_pdb(all_atom_positions, aatype, all_atom_mask, file, chain_index=None, residue_index=None, b_factors=None):

    if chain_index is None:
        chain_index = np.zeros_like(aatype)
    if b_factors is None:
        b_factors = np.zeros_like(all_atom_mask)
    if residue_index is None:
        residue_index = np.arange(len(aatype))+1

    pdb_elem = protein.Protein(
        aatype=aatype,
        atom_positions=all_atom_positions,
        atom_mask=all_atom_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index,
        remark="",
        parents=None,
        parents_chain_index=None,
    )
    outstring = protein.to_pdb(pdb_elem)
    with open(file, 'w') as fp:
        fp.write(outstring)

def struct_to_pdb(struct, file, return_string=False):
    aatype=struct["aatype"]

    atom_positions=struct["final_atom_positions"]
    atom_mask=struct["final_atom_mask"]
    residue_index=struct["residue_index"] if "residue_index" in struct else   np.arange(len(aatype))+1
    b_factors=struct["plddt"].repeat(37).reshape(-1,37) if "plddt" in struct else np.zeros_like(atom_mask)
    chain_index=struct["asym_id"] if "asym_id" in struct else  np.zeros_like(aatype)

    pdb_elem = protein.Protein(
        aatype=aatype,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index,
        remark="",
        parents=None,
        parents_chain_index=None,
    )
    outstring = protein.to_pdb(pdb_elem)
    if return_string:
        return outstring
    with open(file, 'w') as fp:
        fp.write(outstring)

def rotmat_angle_deg(Ra, Rb):
    # Ra, Rb: (..., 3, 3) rotation matrices (torch tensors)
    R = Ra.transpose(-2, -1).matmul(Rb)          # relative rotation
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)               # radians in [0, pi]
    return theta * (180.0 / torch.pi)

def ifft2_center(img: torch.Tensor) -> torch.Tensor:
    """2-dimensional discrete inverse Fourier transform reordered with origin at center."""
    if img.dtype == torch.float16:
        img = img.type(torch.float32)

    return fftshift(ifft2(fftshift(img, dim=(-1, -2))), dim=(-1, -2))

def unsymmetrize_ht(x: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1]
    assert D % 2 != 0
    return x[..., 0:-1, 0:-1] 


def weighted_normalized_l2(F_pred, F_target, weights, eps=1e-8):
    if F_pred.shape[-1] == 2:
        F_pred = torch.view_as_complex(F_pred)
        F_target = torch.view_as_complex(F_target)

    diff = F_pred - F_target
    weighted_sq_diff = weights * torch.abs(diff)**2
    numerator = torch.sum(weighted_sq_diff, dim=-1)

    weighted_ref_norm2 = torch.sum(weights * torch.abs(F_target)**2, dim=-1)
    loss = torch.sqrt(numerator / (weighted_ref_norm2 + eps))

    return loss.mean()

def gaussian_weight(size, sigma=0.5):
    y = torch.arange(size).float() - size//2
    x = torch.arange(size).float() - size//2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    R /= R.max()
    weights = torch.exp(-0.5 * (R/sigma)**2)
    return weights

def frequency_weights(size, p=1.0):
    y = torch.arange(size).float() - size//2
    x = torch.arange(size).float() - size//2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)  # radial distance
    R /= R.max()                  # normalize 0..1
    weights = (1 - R)**p          # 1 at center, 0 at edges
    return weights

def fourier_corr(A: torch.Tensor, B: torch.Tensor,  eps:float=1e-8) -> torch.Tensor:
    num = torch.sum(A * torch.conj(B), dim=(-1,-2))
    denom = torch.sqrt(torch.sum(torch.abs(A) ** 2, dim=(-1,-2)) * torch.sum(torch.abs(B) ** 2, dim=(-1,-2)))
    return (num / (denom+ eps)).real


def get_cc(A: torch.Tensor, B: torch.Tensor, eps:float=1e-8) -> torch.Tensor:
    num = torch.sum(A * B, dim=(-1,-2))
    denom = torch.sqrt(torch.sum(A**2, dim=(-1,-2)) * torch.sum(B**2, dim=(-1,-2)))
    return num / (denom + eps)


def euler2matrix(angles):
    return Rotation.from_euler("zyz", np.array(angles), degrees=True).as_matrix()

def matrix2euler(A):
    return Rotation.from_matrix(A).as_euler("zyz", degrees=True)

def get_sphere(angular_dist):
    num_pts = int(np.pi * 10000 * 1 / (angular_dist ** 2))
    angles = np.zeros((num_pts, 3))
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = (np.arccos(1 - 2.0 * indices / num_pts))
    theta = (np.pi * (1 + 5 ** 0.5) * indices)
    for i in range(num_pts):
        R = Rotation.from_euler("yz", np.array([phi[i], theta[i]]), degrees=False).as_matrix()
        angles[i] = matrix2euler(R)
    return angles

def get_sphere_near(angular_dist, near_angle, near_angle_cutoff):
    angles = get_sphere(angular_dist)
    angles_final = [near_angle]

    for i in range(len(angles)):
        if get_angular_distance(near_angle, angles[i]) < near_angle_cutoff:
            angles_final.append(angles[i])
    return np.array(angles_final)

def get_sphere_full(angular_dist, near_angle=None, near_angle_cutoff=None):
    n_zviews = 360 // angular_dist
    if near_angle is not None:
        angles = get_sphere_near(angular_dist, near_angle, near_angle_cutoff)
    else:
        angles = get_sphere(angular_dist)
    num_pts = len(angles)
    new_angles = np.zeros((num_pts * n_zviews, 3))
    for i in range(num_pts):
        for j in range(n_zviews):
            new_angles[i*n_zviews+ j, 0] =j * angular_dist
            new_angles[i*n_zviews+ j, 1] =angles[i,1]
            new_angles[i*n_zviews+ j, 2] =angles[i,2]
    return new_angles

def get_angular_distance(a1, a2):
    R1 = euler2matrix(a1)
    R2 = euler2matrix(a2)
    R = np.dot(R1, R2.T)
    cosTheta = (np.trace(R) - 1) / 2
    cosTheta = np.clip(cosTheta, -1.0, 1.0) 
    return    np.rad2deg(np.arccos(cosTheta))

def get_corr_ft(v1, v2,v2_sum2):
    corr = ifft3center(v1 * v2.conj()).real
    denom = torch.sqrt(torch.sum(torch.abs(v1) ** 2, dim=(-1,-2,-3))* v2_sum2)
    corr /= denom[..., None, None, None]

    B = corr.shape[0]
    corr_flat = corr.view(B, -1)
    corr_max, flat_index = corr_flat.max(dim=1)
    max_idx = torch.stack(torch.unravel_index(flat_index, corr.shape[1:]), dim=1)
    return corr_max, max_idx


def fft3center(x):
    dim =  (-1,-2,-3)
    return fft.fftshift(fft.fftn(fft.fftshift(x, dim=dim), dim=dim), dim=dim)

def ifft3center(x):
    dim =  (-1,-2,-3)
    return fft.fftshift(fft.ifftn(fft.fftshift(x, dim=dim), dim=dim), dim=dim)

def register_crd_to_vol_iter(crd, vol, angles, chunksize, quality_ratio, sigma, pixel_size, grid_size, real_space):
    device = crd.device
    n_pix_cutoff=int(np.ceil(quality_ratio * sigma / pixel_size) * 2 + 1)    
    n_angles = angles.shape[0]
    n_chunk = int(np.ceil(n_angles/chunksize))

    cc=torch.zeros(len(angles), dtype=crd.dtype, device=device)
    shifts=torch.zeros((len(angles),3), dtype=crd.dtype, device=device)

    vol_sum2 = torch.sum(torch.abs(vol) ** 2, dim=(-1,-2,-3))

    for i in tqdm.tqdm(range(n_chunk), "Angular search", n_chunk) :
        min_indice= (i)*chunksize 
        max_indice= (i+1)*chunksize if (i+1)*chunksize <n_angles else n_angles
        a = angles[min_indice:max_indice]
        R = torch.cat([torch.as_tensor(euler2matrix(_a), device=device, dtype=crd.dtype).unsqueeze(0) for _a in a], dim=0)

        # crd : [N,3]
        # R : [B,3,3]
        c_search = (crd.unsqueeze(0) @ R.unsqueeze(1)).squeeze(1)
        if real_space:
            vox_loc, vox_mask = get_voxel_mask(c_search, grid_size, pixel_size,  n_pix_cutoff)
            vol_search = fft3center(
                vol_real_mask(
                    c_search, 
                    vox_loc, 
                    vox_mask, 
                    grid_size, 
                    sigma, 
                    pixel_size
                )
            )
        else:
            raise NotImplementedError()
            # vol_search = vol_ft(c_search, grid_size, sigma, pixel_size)

        corr, shift = get_corr_ft(vol_search, vol, vol_sum2)
        shifts[min_indice:max_indice] = shift
        cc[min_indice:max_indice] = corr

    best_indice= cc.argmax()
    angles_final = angles[best_indice]
    shift_final = (shifts[best_indice] - grid_size/2 + .5)* pixel_size
    R_final = torch.tensor(euler2matrix(angles_final), device=crd.device, dtype=crd.dtype)

    return angles_final, R_final, shift_final

def register_crd_to_vol(vol, crd, grid_size, sigma, pixel_size, dist_search, real_space=True, quality_ratio=5.0, chunksize="auto"):
    device = crd.device

    if device.type=="cuda":
        if chunksize =="auto":
            free_mem = get_free_mem(crd.device)
            print("Tuning chunksize ..." )
            print("\tFREE MEMORY %.2f GB" %(free_mem/1e9))
            estimated_mem = vol.numel() * vol.element_size() * 10
            print("\tBATCH MEMORY %.2f MB" %(estimated_mem/1e6))
            chunksize = int(free_mem/estimated_mem)
            print("\tCHUNK SIZE %i" %chunksize)
    else:
        chunksize=1

    if not (isinstance(dist_search, tuple) or isinstance(dist_search, list)) : 
        dist_search = (dist_search,)

    angles_final = None
    for i,d in enumerate(dist_search):
        print("Angular search %i (distance=%.2f degrees)"%(i,d))
        angles = get_sphere_full(int(np.floor(d)), near_angle=angles_final, near_angle_cutoff=d*3)
        angles_final, R_final, shift_final = register_crd_to_vol_iter(
            crd, 
            vol, 
            angles, 
            chunksize, 
            quality_ratio, 
            sigma,
            pixel_size, 
            grid_size, 
            real_space
        )
        print("Best angle = %s"%str(angles_final))
        print("Best shift = %s"%str(shift_final))
        
    return angles_final, R_final, shift_final

def get_free_mem(device,memory_fraction=0.9,extra_overhead_gb=1.0):
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory
    reserved = torch.cuda.memory_reserved(device) 
    allocated = torch.cuda.memory_allocated(device)
    free_mem = total_mem - max(reserved, allocated) - extra_overhead_gb
    free_mem *= memory_fraction
    return free_mem

def lattice_ft_2D(device, grid_size = 128, sigma = 1.0, pixel_size=1.0):
    freqs = torch.fft.fftfreq(grid_size, d=pixel_size, device=device)
    u, v = torch.meshgrid(freqs, freqs)
    u = torch.fft.fftshift(u).reshape(grid_size**2)
    v = torch.fft.fftshift(v).reshape(grid_size**2)
    return torch.stack((u,v)).T

def lattice_ft_3D(device, grid_size = 128, sigma = 1.0, pixel_size=1.0):
    freqs = torch.fft.fftfreq(grid_size, d=pixel_size, device=device)
    u, v, w = torch.meshgrid(freqs, freqs, freqs)
    u = torch.fft.fftshift(u).reshape(grid_size**3)
    v = torch.fft.fftshift(v).reshape(grid_size**3)
    w = torch.fft.fftshift(w).reshape(grid_size**3)
    return torch.stack((u,v, w)).T


def img_ft_lattice(crd, crd_lattice, sigma = 1.0, pixel_size=1.0, crd_mask=None, coef=None):
    # crd 
    #   [batch_dim, N_atoms, 3]
    # crd_mask  
    #   [batch_dim, N_atoms]
    # crd_lattice  
    #   [lattice_size, 3]
    # -> Output  
    #   [batch_dim, crd_lattice]

    batch_dim = crd.shape[:-2]
    lattice_size = crd_lattice.shape[-2]

    crd_lattice /= pixel_size

    gaussian_envelope = torch.exp(-2 * (torch.pi**2) * sigma**2 * torch.sum(crd_lattice**2, dim=-1))

    F = torch.exp(-2j * torch.pi * torch.sum(crd[...,None, :] * crd_lattice[..., None, :,:], dim=-1))
    if crd_mask is not None:
        F = F * crd_mask[..., None]
    if coef is not None:
        F = F * coef[..., None]
        
    F = torch.sum(F, dim=-2)

    F /= 2* torch.pi

    return (gaussian_envelope[None] * F).reshape(batch_dim+(lattice_size, ))

def img_ht_lattice(crd, crd_lattice, sigma = 1.0, pixel_size=1.0):
    I = img_ft_lattice(crd, crd_lattice, sigma, pixel_size)
    return  I.real- I.imag

def get_circle(radius,):
    s = radius
    x = torch.arange(s)
    y = torch.arange(s)
    xx, yy = torch.meshgrid(x, y)
    d = (xx - s // 2) ** 2 + (yy - s // 2) ** 2
    mask = d <= (s / 2) ** 2
    circle = torch.stack((xx[mask], yy[mask]), dim=-1)
    return circle


def get_circle_3D(radius):
    s = radius
    x = torch.arange(s)
    y = torch.arange(s)
    z = torch.arange(s)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')  
    center = s // 2
    d = (xx - center) ** 2 + (yy - center) ** 2 + (zz - center) ** 2
    mask = d <= (s / 2) ** 2

    sphere = torch.stack((xx[mask], yy[mask], zz[mask]), dim=-1)
    return sphere


def get_pixel_mask(coord, grid_size, pixel_size, n_pix_cutoff):
    threshold = (n_pix_cutoff - 1) // 2
    circle = get_circle(n_pix_cutoff)  # shape (n_pix, 2)
    circle= circle.to(coord.device)
    
    # Compute base pixel positions for all atoms (broadcasting)
    base_pix = torch.floor(coord[..., :2] / pixel_size - threshold + grid_size / 2)  # shape (n_atoms, 2)
    
    # Add circle offsets to each base pixel position
    pix = base_pix[..., None, :] + circle[None, :, :]  # shape (n_atoms, n_pix, 2)

    # Validity mask: check if x and y are in bounds
    valid_x = (pix[..., 0] >= 0) & (pix[..., 0] < grid_size)
    valid_y = (pix[..., 1] >= 0) & (pix[..., 1] < grid_size)
    valid_mask = valid_x & valid_y

    pix[valid_mask==0.0] = 0.0

    return pix, valid_mask


def get_voxel_mask(coord, grid_size, pixel_size, n_pix_cutoff):
    threshold = (n_pix_cutoff - 1) // 2
    circle = get_circle_3D(n_pix_cutoff)  # shape (n_pix, 3)
    circle= circle.to(coord.device)

    if len(coord.shape) ==3:
        circle = circle[None]
    
    # Compute base pixel positions for all atoms (broadcasting)
    base_pix = torch.floor(coord / pixel_size - threshold + grid_size / 2)  # shape (n_atoms, 3)
    
    # Add circle offsets to each base pixel position
    pix = base_pix[..., None, :] + circle[..., None, :, :]  # shape (n_atoms, n_pix, 2)

    # Validity mask: check if x and y are in bounds
    valid_x = (pix[..., 0] >= 0) & (pix[..., 0] < grid_size)
    valid_y = (pix[..., 1] >= 0) & (pix[..., 1] < grid_size)
    valid_z = (pix[..., 2] >= 0) & (pix[..., 2] < grid_size)
    valid_mask = valid_x & valid_y & valid_z

    pix[valid_mask==0.0] = 0.0

    return pix, valid_mask


def img_real(crd, grid_size = 128, sigma = 1.0, pixel_size=1.0):
    # crd 
    #   [batch_dim, N_atoms, 3]
    # crd_mask  
    #   [batch_dim, N_atoms]

    batch_dim = crd.shape[:-2]

    xx, yy = torch.meshgrid(
        torch.arange(grid_size, device=crd.device), torch.arange(grid_size, device=crd.device),
    )
    xx = (xx.reshape(grid_size**2) - grid_size/2 +.5) * pixel_size
    yy = (yy.reshape(grid_size**2) - grid_size/2 +.5) * pixel_size

    I = torch.exp(-1 / (2 * sigma**2)  * ( 
        (xx - crd[..., 1, None])**2 + \
        (yy - crd[..., 0, None])**2 ) \
    )
    I = torch.sum(I, dim=-2)
    I /= (2 * torch.pi * sigma**2)
    return  I.reshape(batch_dim+(grid_size, grid_size))


def img_real_mask(crd, pix_loc, pix_mask, grid_size=128, sigma=1.0, pixel_size=1.0, coef=None):
    B, N_atoms, P = pix_mask.shape

    # Compute (x, y) positions in real space
    xy = (pix_loc - grid_size / 2 + 0.5) * pixel_size  # (B, N_atoms, P, 2)
    crd_xy = crd[:, :, None, :2]                       # (B, N_atoms, 1, 2)

    # Compute Gaussian contributions
    sq_dist = torch.sum((xy - crd_xy) ** 2, dim=-1)    # (B, N_atoms, P)
    I = torch.exp(-0.5 * sq_dist / sigma**2) * pix_mask  # (B, N_atoms, P)
    if coef is not None:
        I = I *  coef[..., None] 

    # Flatten for scatter
    flat_I = I.reshape(B, -1)                                # (B, N_atoms * P)
    flat_idx = pix_loc.reshape(B, -1, 2).long()              # (B, N_atoms * P, 2)

    # Build linear indices for scatter_add
    x = flat_idx[..., 0]
    y = flat_idx[..., 1]
    idx = x * grid_size + y                                  # flatten 2D index to 1D

    # Scatter into flat image buffer
    I_out = torch.zeros(B, grid_size * grid_size, device=crd.device, dtype=I.dtype)
    I_out = I_out.scatter_add(1, idx, flat_I)

    # Reshape back to 2D grid
    I_out = I_out.view(B, grid_size, grid_size)

    # Normalize Gaussian
    I_out /= (2 * torch.pi * sigma**2)

    return I_out.transpose(-2,-1)

def vol_real_mask(crd, pix_loc, pix_mask, grid_size=128, sigma=1.0, pixel_size=1.0, coef=None):
    B, N_atoms, P = pix_mask.shape

    # Compute (x, y) positions in real space
    xyz = (pix_loc - grid_size / 2 + 0.5) * pixel_size  # (B, N_atoms, P, 3)
    crd_xyz = crd[:, :, None, : ]                       # (B, N_atoms, 1, 3)

    # Compute Gaussian contributions
    sq_dist = torch.sum((xyz - crd_xyz) ** 2, dim=-1)    # (B, N_atoms, P)
    I = torch.exp(-0.5 * sq_dist / sigma**2) * pix_mask  # (B, N_atoms, P)

    if coef is not None : 
        I *= coef[..., None]

    # Flatten for scatter
    flat_I = I.reshape(B, -1)                                # (B, N_atoms * P)
    flat_idx = pix_loc.reshape(B, -1, 3).long()              # (B, N_atoms * P, 2)

    # Build linear indices for scatter_add
    x = flat_idx[..., 0]
    y = flat_idx[..., 1]
    z = flat_idx[..., 2]
    idx = x * (grid_size**2) + y * grid_size + z             # flatten 2D index to 1D


    # Scatter into flat vol buffer
    I_out = torch.zeros(B, grid_size **3, device=crd.device, dtype=I.dtype)
    I_out = I_out.scatter_add(1, idx, flat_I)

    # Reshape back to 3D grid
    I_out = I_out.view(B, grid_size, grid_size,grid_size)

    # Normalize Gaussian
    I_out /= (2 * torch.pi * sigma**2)

    return torch.permute(I_out, (0,3,2,1))



def vol_ft(crd, grid_size = 128, sigma = 1.0, pixel_size=1.0, crd_mask=None):
    # crd 
    #   [batch_dim, N_atoms, 3]
    # crd_mask  
    #   [batch_dim, N_atoms]

    batch_dim = crd.shape[:-2]

    freqs = torch.fft.fftfreq(grid_size, d=pixel_size, device=crd.device)
    u, v, w = torch.meshgrid(freqs, freqs, freqs)
    u = torch.fft.fftshift(u).reshape(grid_size**3)
    v = torch.fft.fftshift(v).reshape(grid_size**3)
    w = torch.fft.fftshift(w).reshape(grid_size**3)

    gaussian_envelope = torch.exp(-2 * (torch.pi**2) * sigma**2 * (u**2 + v**2 + w**2))

    F = torch.exp(-2j * torch.pi * (
        crd[..., 2, None] * u + \
        crd[..., 1, None] * v + \
        crd[..., 0, None] * w )
    )

    if crd_mask is not None:
        F = torch.sum(crd_mask[..., None] *  F, dim=-2)
    else:
        F = torch.sum(F, dim=-2)

    return (gaussian_envelope[None] * F).reshape(batch_dim + (grid_size, grid_size, grid_size))



def vol_real(crd, grid_size = 128, sigma = 1.0, pixel_size=1.0):
    # crd 
    #   [batch_dim, N_atoms, 3]
    # crd_mask  
    #   [batch_dim, N_atoms]

    batch_dim = crd.shape[:-2]

    grid = torch.arange(grid_size, device=crd.device)
    xx, yy, zz = torch.meshgrid(grid,grid,grid)
    xx = (xx.reshape(grid_size**3) - grid_size/2 +.5) * pixel_size
    yy = (yy.reshape(grid_size**3) - grid_size/2 +.5) * pixel_size
    zz = (zz.reshape(grid_size**3) - grid_size/2 +.5) * pixel_size

    I = torch.exp(-1 / (2 * sigma**2)  * ( 
        (xx - crd[..., 2, None])**2 + \
        (yy - crd[..., 1, None])**2 + \
        (zz - crd[..., 0, None])**2 ) 
    )
    I = torch.sum(I, dim=-2)
    I /= (2 * torch.pi * sigma**2)
    return  I.reshape(batch_dim + (grid_size, grid_size, grid_size))


atomdefs={'H':(1.0,1.00794),'HO':(1.0,1.00794),'C':(6.0,12.0107),'A':(7.0,14.00674),'N':(7.0,14.00674),'O':(8.0,15.9994),'P':(15.0,30.973761),'K':(19.0,39.0983),
    'S':(16.0,32.066),'W':(18.0,1.00794*2.0+15.9994),'AU':(79.0,196.96655) }
def aatype_to_coefs(aatype):


    def atom_name_to_coef(atomlist):
        return [atomdefs[c[:1]][0] if c != "" else 0.0 for c in atomlist]

    def aatype_to_14_names(a):
        return rc.restype_name_to_atom14_names[rc.restype_1to3[rc.restypes[a]]]

    n = aatype.shape[-1]
    coef37 = aatype.new_zeros((n, 37))

    for a in range(aatype.shape[-1]):
        coef_i = atom_name_to_coef(aatype_to_14_names(aatype[a]))
        ind_i =  rc.RESTYPE_ATOM14_TO_ATOM37[aatype[a]]
        zero_ind=False
        for c, i in zip(coef_i, ind_i):
            if i==0 and zero_ind:
                break
            elif i==0:
                zero_ind=True
            coef37[a,i] = c 
            
    return coef37



def dcd2numpyArr(filename):
    """
    Read coordinate file (.DCD)
    :param filename: DCD file
    :return: coordinates ncoord * n_atoms * 3
    """
    print("> Reading dcd file %s" % filename)
    BYTESIZE = 4
    with open(filename, 'rb') as f:

        # Header
        # ---------------- INIT

        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        crd_type = f.read(BYTESIZE).decode('ascii')
        nframe = int.from_bytes((f.read(BYTESIZE)), "little")
        start_frame = int.from_bytes((f.read(BYTESIZE)), "little")
        len_frame = int.from_bytes((f.read(BYTESIZE)), "little")
        len_total = int.from_bytes((f.read(BYTESIZE)), "little")
        for i in range(5):
            f.read(BYTESIZE)
        time_step = np.frombuffer(f.read(BYTESIZE), dtype=np.float32)
        for i in range(9):
            f.read(BYTESIZE)
        charmm_version = int.from_bytes((f.read(BYTESIZE)), "little")

        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- TITLE
        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        ntitle = int.from_bytes((f.read(BYTESIZE)), "little")
        tilte_rd = f.read(BYTESIZE * 20 * ntitle)
        try:
            title = tilte_rd.encode("ascii")
        except AttributeError:
            title = str(tilte_rd)
        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- NATOM
        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        natom = int.from_bytes((f.read(BYTESIZE)), "little")
        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ----------------- DCD COORD
        dcd_arr = np.zeros((nframe, natom, 3), dtype=np.float32)
        for i in range(nframe):
            for j in range(3):

                start_size = int.from_bytes((f.read(BYTESIZE)), "little")
                while (start_size != BYTESIZE * natom and start_size != 0):
                    # print("\n-- UNKNOWN %s -- " % start_size)

                    f.read(start_size)
                    end_size = int.from_bytes((f.read(BYTESIZE)), "little")
                    if end_size != start_size:
                        raise RuntimeError("Can not read dcd file")
                    start_size = int.from_bytes((f.read(BYTESIZE)), "little")

                bin_arr = f.read(BYTESIZE * natom)
                if len(bin_arr) == BYTESIZE * natom:
                    dcd_arr[i, :, j] = np.frombuffer(bin_arr, dtype=np.float32)
                else:
                    break
                end_size = int.from_bytes((f.read(BYTESIZE)), "little")
                if end_size != start_size:
                    if i > 1:
                        break
                    else:
                        # pass
                        raise RuntimeError("Can not read dcd file %i %i " % (start_size, end_size))

        print("\t -- Summary of DCD file -- ")
        print("\t\t crd_type  : %s" % crd_type)
        print("\t\t nframe  : %s" % nframe)
        print("\t\t len_frame  : %s" % len_frame)
        print("\t\t len_total  : %s" % len_total)
        print("\t\t time_step  : %s" % time_step)
        print("\t\t charmm_version  : %s" % charmm_version)
        print("\t\t title  : %s" % title)
        print("\t\t natom  : %s" % natom)
    print("\t Done \n")

    return dcd_arr

def numpyArr2dcd(arr, filename, start_frame=1, len_frame=1, time_step=1.0, title=None):
    """
    Write coordinate file (.DCD)
    :param arr: coordinates ncoord * natoms * 3
    :param filename: DCD file
    :param start_frame:
    :param len_frame:
    :param time_step:
    :param title:
    """
    print("> Wrinting dcd file %s" % filename)
    BYTESIZE = 4
    nframe, natom, _ = arr.shape
    len_total = nframe * len_frame
    charmm_version = 24
    if title is None:
        title = "DCD file generated by AFMfit"
    ntitle = (len(title) // (20 * BYTESIZE)) + 1
    with open(filename, 'wb') as f:
        zeroByte = int.to_bytes(0, BYTESIZE, "little")

        # Header
        # ---------------- INIT
        f.write(int.to_bytes(21 * BYTESIZE, BYTESIZE, "little"))
        f.write(b'CORD')
        f.write(int.to_bytes(nframe, BYTESIZE, "little"))
        f.write(int.to_bytes(start_frame, BYTESIZE, "little"))
        f.write(int.to_bytes(len_frame, BYTESIZE, "little"))
        f.write(int.to_bytes(len_total, BYTESIZE, "little"))
        for i in range(5):
            f.write(zeroByte)
        f.write(np.float32(time_step).tobytes())
        for i in range(9):
            f.write(zeroByte)
        f.write(int.to_bytes(charmm_version, BYTESIZE, "little"))

        f.write(int.to_bytes(21 * BYTESIZE, BYTESIZE, "little"))

        # ---------------- TITLE
        f.write(int.to_bytes((ntitle * 20 + 1) * BYTESIZE, BYTESIZE, "little"))
        f.write(int.to_bytes(ntitle, BYTESIZE, "little"))
        f.write(title.ljust(20 * BYTESIZE).encode("ascii"))
        f.write(int.to_bytes((ntitle * 20 + 1) * BYTESIZE, BYTESIZE, "little"))

        # ---------------- NATOM
        f.write(int.to_bytes(BYTESIZE, BYTESIZE, "little"))
        f.write(int.to_bytes(natom, BYTESIZE, "little"))
        f.write(int.to_bytes(BYTESIZE, BYTESIZE, "little"))

        # ----------------- DCD COORD
        for i in range(nframe):
            for j in range(3):
                f.write(int.to_bytes(BYTESIZE * natom, BYTESIZE, "little"))
                f.write(np.float32(arr[i, :, j]).tobytes())
                f.write(int.to_bytes(BYTESIZE * natom, BYTESIZE, "little"))
    print("\t Done \n")