




import torch 
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt
from flexfold.fsc import fourier_shell_correlation, fsc_auc,fsc_thresh, spherical_soft_mask, fourier_mask
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from flexfold import dataset
from flexfold.pose import PoseTracker
from flexfold.lattice import Lattice
from cryodrgn import __version__, ctf
from flexfold.core import unsymmetrize_ht,  dcd2numpyArr, numpyArr2dcd, get_voxel_mask, vol_real_mask
from cryodrgn.mrcfile import parse_mrc, write_mrc
import torch.nn.functional as F
import argparse
import math
from mpmath import erfinv
import tqdm
import os
from cryodrgn.commands_utils.fsc import calculate_cryosparc_fscs
import numpy as np
from torch import fft
import glob 
from cryodrgn import lie_tools, utils
from torch.utils.data import DataLoader

def fft3center(x):
    dim =  (-1,-2,-3)
    return fft.fftshift(fft.fftn(fft.fftshift(x, dim=dim), dim=dim, norm='ortho'), dim=dim)

def ifft3center(x):
    dim =  (-1,-2,-3)
    return fft.fftshift(fft.ifftn(fft.fftshift(x, dim=dim), dim=dim, norm='ortho'), dim=dim)

def fft2center(x):
    dim =  (-1,-2)
    return fft.fftshift(fft.fftn(fft.fftshift(x, dim=dim), dim=dim, norm='ortho'), dim=dim)

def ifft2center(x):
    dim =  (-1,-2)
    return fft.fftshift(fft.ifftn(fft.fftshift(x, dim=dim), dim=dim, norm='ortho'), dim=dim)

def read_coords(pdb_file):
    """
    Read coords in PDB file
    :param pdb_file: PDB file
    :return: array of n_atoms*3
    """
    coords = []
    with open(pdb_file, "r") as f:
        for line in f:
            if 'ATOM' in line:
                coords.append([
                    line[30:38], line[38:46], line[46:54]
                ])
    return np.array(coords).astype(float)

def gaussian_kernel3d(kernel_size: int, sigma: float, device=None, dtype=None):
    """
    Create a normalized 3D Gaussian kernel.

    Args:
        kernel_size (int): Size of the kernel (should be odd, e.g. 3, 5, 7, ...).
        sigma (float): Standard deviation of the Gaussian.
        device, dtype: Optional, to match target tensor.

    Returns:
        torch.Tensor: 3D Gaussian kernel of shape (1, 1, k, k, k)
    """
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    # 1D coordinates
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')

    # Gaussian formula
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Reshape for conv3d (out_channels=1, in_channels=1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel


def gaussian_filter3d(x: torch.Tensor, kernel_size: int, sigma: float):
    """
    Apply a 3D Gaussian filter to a tensor using conv3d.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, D, H, W)
        kernel_size (int): Gaussian kernel size (odd)
        sigma (float): Gaussian standard deviation

    Returns:
        torch.Tensor: Smoothed tensor
    """
    device, dtype = x.device, x.dtype
    kernel = gaussian_kernel3d(kernel_size, sigma, device=device, dtype=dtype)
    padding = kernel_size // 2
    x_smoothed = F.conv3d(x, kernel, padding=padding, groups=x.shape[1])
    return x_smoothed

def make_soft_circular_mask(D, cutoff_radius_frac=0.9, transition_frac=0.05, device='cpu'):
    """
    D: image size (assume square D x D)
    cutoff_radius_frac: fraction of Nyquist (0..1) where mask ~1 inside
    transition_frac: fraction of Nyquist for cosine falloff
    returns: mask [D,D] real in [0,1] centered (for FFT grid with fftshift)
    """
    # Create normalized freq coords [-1,1]
    coords = torch.linspace(-1.0, 1.0, steps=D, device=device)
    Y, X = torch.meshgrid(coords, coords, indexing='ij')
    r = torch.sqrt(X**2 + Y**2)  # 0..sqrt(2)
    # normalize radius so Nyquist corresponds to 1.0
    r = r / math.sqrt(2.0)

    r0 = cutoff_radius_frac
    trans = transition_frac
    mask = torch.zeros_like(r)

    inner = r <= r0
    trans_zone = (r > r0) & (r <= r0 + trans)
    mask[inner] = 1.0
    # cosine rolloff in transition zone
    mask[trans_zone] = 0.5 * (1 + torch.cos(math.pi * (r[trans_zone] - r0) / trans))
    # outside stays 0
    return mask

def rotateVolume( V, R, align_corners=True):
        # Input shape: R [B, 3, 3]
        B = R.shape[0]
        D = V.shape[-1]

        # Convert 3x3 to 3x4 affine matrices by appending zero translation
        R_aff = torch.cat(
            (R, torch.zeros(B, 3, 1, device=V.device, dtype=R.dtype)),
            dim=2
        )  # [B,3,4]

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

def flat_idx(idx, grid_size):
    return idx[...,0]*grid_size**2 + idx[...,1]*grid_size + idx[...,2]

def map_to_canonical(
        vol_target, 
        vox_loc_target, 
        vox_loc_canonical, 
        gaussian_weights,
        gaussian_norm,
        eps=1e-5,
        inverse=False
    ):
    """
    Vol_target : [B, N, N, N]
    vox_loc_target : [B, N_atoms, P, 3]
    vox_loc_canonical : [1, N_atoms, P, 3]
    gaussian_weights : [1, N_atoms, P]
    gaussian_norm : [1, N, N, N]
    """
    B, N_atoms, P, _ = vox_loc_target.shape
    grid_size = vol_target.shape[-1]



    flat_idx_target = flat_idx(vox_loc_target.view(B, -1, 3).long(), grid_size)   # (B, N_atoms*P)
    flat_idx_canonical = flat_idx(vox_loc_canonical.view(1, -1, 3).long(),  grid_size)   # (1, N_atoms*P)
    flat_idx_canonical = flat_idx_canonical.expand(B, N_atoms*P)

    if inverse:
        tmp = flat_idx_target 
        flat_idx_target = flat_idx_canonical
        flat_idx_canonical = tmp

    # ---- sample from deformed ----
    # (B, N_atoms*P)
    vol_target = torch.permute(vol_target, (0,3,2,1))
    vol_flat = vol_target.reshape(B, -1)
    vox_vals = torch.gather(vol_flat , 1, flat_idx_target)

    G = gaussian_weights.expand(B, N_atoms, P)
    G = G.view(B, -1)

    # weighted contributions
    vox_vals = G * vox_vals                     # (B, N_atoms*P)

    # ---- scatter to canonical ----
    vol_canon =  torch.zeros(B, grid_size **3, device=vol_target.device, dtype=vol_target.dtype)

    vol_canon= vol_canon.scatter_add(1, flat_idx_canonical, vox_vals)

    vol_canon = vol_canon / (gaussian_norm + eps)

    vol_canon = vol_canon.view(B, grid_size, grid_size, grid_size)
    
    # permute to match volume data ordering Z, Y, X
    vol_canon = torch.permute(vol_canon, (0,3,2,1))

    return vol_canon


def gaussian_kernel_norm(gaussian_weigths, vox_loc_canonical, grid_size):
    B, N_atoms, P, _ = vox_loc_canonical.shape
    flat_idx_canonical = flat_idx(vox_loc_canonical.view(B, -1, 3).long(),  grid_size)   # (1, N_atoms*P)
    norm      =  torch.zeros(B, grid_size **3, device=vox_loc_canonical.device)
    norm= norm.scatter_add(1, flat_idx_canonical, gaussian_weigths.view(1,-1).expand(B,-1))
    return norm

def gaussian_kernel_weigths(crd, vox_loc,vox_mask, coef, grid_size, pixel_size, sigma):
    xyz = (vox_loc - grid_size / 2 + 0.5) * pixel_size  # (1, N_atoms, P, 3)
    sq_dist = torch.sum((xyz - crd[:, :, None, : ]) ** 2, dim=-1)    # (1, N_atoms, P)
    G = torch.exp(-0.5 * sq_dist / sigma**2) * vox_mask  # (1, N_atoms, P)
    if coef is not None : 
        G *= coef.unsqueeze(-1)
    return G

def gaussian_halfwidth_voxels(sigma, spacing, p=0.99):
    # a = sqrt(2)*sigma * erfinv(p^(1/3))
    a_over_sigma = math.sqrt(2.0) * float(erfinv(p ** (1.0/3.0)))
    cutoff = a_over_sigma * sigma

    half_pts = math.ceil(cutoff / spacing)
    patch_size = 2*half_pts + 1
    return patch_size

def gaussian_kernel_mask(vox_mask, grid_size, device):
    B, N_atoms, P, _ = vox_mask.shape
    vol_mask = torch.zeros((B, grid_size, grid_size, grid_size), dtype=torch.bool, device=device)
    mask_coords = vox_mask.reshape(B, -1, 3 ).long()
    for b in range(B):
        vol_mask[b, mask_coords[b,...,2], mask_coords[b,...,1], mask_coords[b,...,0]] = True
    return vol_mask



def flexible_backprojection(
        imageDataset,
        posetracker,
        lattice, 
        crd_canon : torch.Tensor, 
        coordinates: torch.Tensor,
        ctf_params : torch.Tensor, 
        coefs : torch.Tensor, 
        pixel_size : float,  
        sigma : float, 
        n_pix_cutoff : int,
        batch_size: int = 4, 
        rigid : bool = False, 
        eps : float = 1e-8, 
        num_workers=4,
        ) :
    """
    
    """
    device = crd_canon.device

    D= imageDataset.D
    Dr = grid_size = D-1
    N = imageDataset.N

    # Initialize volumes
    volume = torch.zeros(Dr, Dr, Dr, device=device)
    volume_half1 = torch.zeros(Dr, Dr, Dr, device=device)
    volume_half2 = torch.zeros(Dr, Dr, Dr, device=device)

    volume_reg = torch.zeros(Dr, Dr, Dr, device=device)
    volume_reg_half1 = torch.zeros(Dr, Dr, Dr, device=device)
    volume_reg_half2 = torch.zeros(Dr, Dr, Dr, device=device)

    # soft_mask = make_soft_circular_mask(D, cutoff_radius_frac=0.95, transition_frac=0.05, device=device)
    soft_mask = None

    if not rigid : 
        vox_mask_canon = get_voxel_mask(crd_canon, grid_size, pixel_size,  n_pix_cutoff)
        gaussian_weights = gaussian_kernel_weigths(crd_canon, vox_mask_canon[0],vox_mask_canon[1], coefs, grid_size, pixel_size, sigma)
        gaussian_norm = gaussian_kernel_norm(gaussian_weights, vox_mask_canon[0], grid_size)

        vol_smooth_mask = vol_real_mask(crd_canon, vox_mask_canon[0], vox_mask_canon[1], grid_size, sigma, pixel_size, coef=coefs)
        vol_smooth_mask/= vol_smooth_mask.max()
        vol_smooth_inv_mask = 1.0-vol_smooth_mask

    loader = DataLoader(
        imageDataset,
        batch_size=batch_size,
        shuffle=False,   
        num_workers=num_workers,   
        pin_memory=True,  
    )


    for ii, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):

        # Get batch particle ctf and pose
        particles_ft,_, idx = batch
        particles_ft=particles_ft.to(device)

        # Batching
        start = int(idx[0])
        end   = int(idx[-1]) + 1
        B     = len(idx)

        rot, tran = posetracker.get_pose(torch.arange(start,end, device=device))
        c = ctf_from_params(ctf_params[start:end], lattice)

        if not args.rigid:
            crd_traj = coordinates[start: end]

        # Inverse particle
        particles_ft*= -1

        # Apply CTF
        y=particles_ft * c

        # Phase Shift 
        y = torch.view_as_complex(lattice.translate_ft(torch.view_as_real(y).view(B, D*D, 2), tran.unsqueeze(1)).view(B, D, D, 2))

        # # Apply mask
        if soft_mask is not None:
            y*= soft_mask

        # Backproject image (expand + rotate)
        y_real = ifft2center(unsymmetrize_ht(y)).real
        recon = y_real.unsqueeze(1).repeat(1,Dr,1, 1)
        recon  = rotateVolume(recon.unsqueeze(1), rot).squeeze(1)

        if not rigid:
            # Get voxels that gets warped
            vox_mask_recon = get_voxel_mask(crd_traj, grid_size, pixel_size,  n_pix_cutoff)

            # Map deformed voxels to canonical
            recon_deformed = map_to_canonical(
                vol_target=recon, 
                vox_loc_target =vox_mask_recon[0], 
                vox_loc_canonical = vox_mask_canon[0], 
                gaussian_weights=gaussian_weights,
                gaussian_norm=gaussian_norm,
                eps=eps
            )
            gaussian_norm_recon = gaussian_kernel_norm(gaussian_weights, vox_mask_recon[0], grid_size)

            recon_invdeformed = map_to_canonical(
                vol_target=recon, 
                vox_loc_target =vox_mask_recon[0], 
                vox_loc_canonical = vox_mask_canon[0], 
                gaussian_weights=gaussian_weights,
                gaussian_norm=gaussian_norm_recon,
                eps=eps,
                inverse=True
            )

            # filter
            vol_recon_smooth_mask = vol_real_mask(crd_traj, vox_mask_recon[0], vox_mask_recon[1], grid_size, sigma, pixel_size, coef=coefs)
            vol_recon_smooth_mask/= vol_recon_smooth_mask.max()
            vol_recon_smooth_inv_mask = 1.0-vol_recon_smooth_mask
            vol_smooth_inv_mask_merged = torch.min(vol_recon_smooth_inv_mask, vol_smooth_inv_mask)
            vol_smooth_inv_mask_intersect = (vol_recon_smooth_mask * vol_smooth_inv_mask)

            # print(torch.sum(vol_smooth_inv_mask_intersect[0]))
            # print(torch.sum(vol_recon_smooth_inv_mask[0]))
            # print(torch.sum(vol_smooth_inv_mask[0]))
            # print(torch.sum(vol_smooth_inv_mask_merged[0]))

            # print(torch.norm(recon[0]))
            # print(torch.norm(recon_invdeformed[0]))
            # raise

            recon = ( recon_deformed*vol_smooth_mask )+ (vol_smooth_inv_mask_merged * recon) + (vol_smooth_inv_mask_intersect*recon_invdeformed)

        # sum
        # recon /= N
        recon = torch.sum(recon, dim=0)

        # Same for CTF (expand + rotate)
        if soft_mask is not None:
            c*= soft_mask
        ctf_real = ifft2center(unsymmetrize_ht(c)).real
        ctf_real = ctf_real.unsqueeze(-3)
        ctf_recon = ctf_real.repeat(1, Dr, 1,1)

        ctf_recon = rotateVolume(ctf_recon.unsqueeze(1), rot).squeeze(1)


        if not rigid:
            ctf_recon_deformed = map_to_canonical(
                vol_target=ctf_recon, 
                vox_loc_target =vox_mask_recon[0], 
                vox_loc_canonical = vox_mask_canon[0], 
                gaussian_weights=gaussian_weights,
                gaussian_norm=gaussian_norm,
                eps=eps
            )   
            ctf_recon_invdeformed = map_to_canonical(
                vol_target=ctf_recon, 
                vox_loc_target =vox_mask_recon[0], 
                vox_loc_canonical = vox_mask_canon[0], 
                gaussian_weights=gaussian_weights,
                gaussian_norm=gaussian_norm_recon,
                eps=eps,
                inverse=True
            )   
            # ctf_recon = ctf_recon_deformed 
            ctf_recon = ( ctf_recon_deformed*vol_smooth_mask )+ (vol_smooth_inv_mask_merged * ctf_recon)+ (vol_smooth_inv_mask_intersect*ctf_recon_invdeformed)
        # Warp and rotate
        ctf_recon =  fft3center(ctf_recon)
        ctf_recon = torch.abs(ctf_recon) ** 2
        # ctf_recon /= N

        ctf_recon =  torch.sum(ctf_recon, dim=0).real

        volume += recon
        volume_reg += ctf_recon

        if ii%2 == 0:
            volume_half1 += recon
            volume_reg_half1 += ctf_recon
        else:
            volume_half2 += recon
            volume_reg_half2 += ctf_recon

    return volume, volume_half1, volume_half2,  volume_reg, volume_reg_half1, volume_reg_half2

def regularize_volume(volume, volume_reg, wiener_constant):

    volume_ft = fft3center(volume)
    # volume_ft=torch.conj(volume_ft)

    regularized_counts = volume_reg + wiener_constant * volume_reg.mean()
    regularized_counts *= volume_reg.mean() / regularized_counts.mean()

    volume_ft_reg = volume_ft / (regularized_counts)

    return ifft3center(volume_ft_reg).real

def normalize_volume(volume, crd_canon, vox_mask_canon, mask, rmsf, grid_size, sigma, pixel_size, coefs):

    vol_canon_smooth_binary_mask = mask / mask.max()
    vol_canon_smooth_binary_mask_inv = 1.0-vol_canon_smooth_binary_mask

    c = coefs/rmsf
    vol_invvar = vol_real_mask(crd_canon, vox_mask_canon[0], vox_mask_canon[1], grid_size, sigma, pixel_size, coef=c)
    vol_invvar  = vol_canon_smooth_binary_mask_inv + vol_canon_smooth_binary_mask * vol_invvar
    
    return volume * vol_invvar.squeeze(0) 


class FlexibleBackprojection:
    
    def __init__(
            self,
            crd_canon,
            indices_file,
            particles_file,
            poses_file,
            ctf_file,
            coordinates_file,
            coefs_file,
            lazy,
            rigid,
            pixel_size,
            gaussian_threshold,
            sigma,
    ):
        device = crd_canon.device

        self.crd_canon = crd_canon
        self.sigma = sigma
        self.pixel_size = pixel_size
        self.rigid=rigid
        self.gaussian_threshold= gaussian_threshold

        if indices_file is not None:
            self.indices = np.loadtxt(indices_file).astype(int)
        else:
            self.indices=None
        print("Reading particle file %s ..."%particles_file)
        self.imageDataset = dataset.ImageDataset(mrcfile=particles_file,device=device, ind=self.indices, lazy=lazy)
        self.N = self.imageDataset.N
        self.D = self.imageDataset.D
        self.Dr = self.D-1
        if self.pixel_size is None:
            self.pixel_size = self.imageDataset.src.apix
            if self.pixel_size is None:
                raise RuntimeError("Pixel size missing in metadata, must provide pixel size")
        print("Image data contains %s particles with pixel size %.2f ang"%(self.N, self.pixel_size))
        if self.indices is None :
            self.indices = np.arange(self.N)

        # Lattice
        self.lattice = Lattice(self.D, extent=0.5).to(device)

        # Poses
        if os.path.splitext(poses_file)[1] == ".ckpt":
            dummy_rot_np = np.zeros((self.indices.shape[0],3,3))
            dummy_trans_np = np.zeros((self.indices.shape[0],2))
            self.posetracker = PoseTracker(rots_np=dummy_rot_np, trans_np=dummy_trans_np,D=self.D, emb_type="quat").to(device)
            weights = torch.load(poses_file, map_location="cpu")["state_dict"]
            rots_quat =  weights["posetracker.rots"].to(device)[self.indices]
            self.posetracker.rots_emb.weight.data.copy_(lie_tools.SO3_to_quaternions(rots_quat))
            self.posetracker.trans_emb.weight.data.copy_(weights["posetracker.trans"].to(device)[self.indices])
        else:
            self.posetracker = PoseTracker.load(poses_file, self.N, self.D, None, ind=self.indices).to(device)

        # CTF
        self.ctf_params = torch.tensor(ctf.load_ctf_for_training(self.D - 1,ctf_file)).to(device)
        self.ctf_params = self.ctf_params[self.indices]

        if not self.rigid:
            self.coordinates = torch.tensor(dcd2numpyArr(coordinates_file), device=device)

            assert (self.coordinates.shape[0] == self.N or self.coordinates.shape[0] == (self.N+1) )
            assert self.crd_canon.shape[-2] == self.coordinates.shape[-2]

            self.coefs =  torch.load(coefs_file).to(device).unsqueeze(0)
            assert self.coordinates.shape[1] == self.coefs.shape[-1]

            self.n_pix_cutoff = gaussian_halfwidth_voxels(self.sigma, self.pixel_size, p=gaussian_threshold) # number of pixel for thrshold of 99% of the gaussian kernels

            crd_avg = self.coordinates.mean(dim=0, keepdim=True)
            rmsd = torch.sqrt(torch.mean(torch.square(torch.norm(self.crd_canon - crd_avg, dim=-1))))
            print("Canonical is %.2f ang away from average trajectory"%rmsd)
            print("Using flexible deformations from coordinates %s "%(coordinates_file))
            print("Number of voxels in Gaussian interpolation  = %i ** 3"%self.n_pix_cutoff)

    def run_backproj(self, batch_size, eps):

        with torch.no_grad():
            return flexible_backprojection(
                    imageDataset=self.imageDataset,
                    lattice=self.lattice, 
                    posetracker=self.posetracker,
                    crd_canon=self.crd_canon,
                    coordinates=self.coordinates, 
                    ctf_params=self.ctf_params , 
                    coefs=self.coefs, 
                    pixel_size=self.pixel_size ,  
                    sigma=self.sigma , 
                    n_pix_cutoff=self.n_pix_cutoff,
                    rigid=self.rigid, 
                    eps=eps, 
                    batch_size=batch_size, 
            ) 


def update_chunk_volumes(volumes, v):
    if volumes is None:
        return v
    else:
        return tuple(v1 + v2 for v1, v2 in zip(volumes, v))

def main(args):
    ###################################################################################################""
    # Inputs
    ###################################################################################################""

    eps = args.eps
    batch_size=args.batch_size
    wiener_constant = args.wiener_constant
    particles_file = args.particles
    poses_file = args.poses
    ctf_file = args.ctf
    output_dir = args.output_dir
    indices_file = args.indices
    coordinates_file = args.coordinates
    pixel_size = args.pixel_size
    lazy = args.lazy
    reference_file=args.reference
    coefs_file=args.coefs
    gaussian_threshold = args.gaussian_threshold
    rigid = args.rigid
    sigma = args.sigma

    reg_volumes_prefix = args.volumes

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if reg_volumes_prefix is None:

        if not rigid:
            crd_canon = torch.tensor(read_coords(reference_file), device=device, dtype=torch.float32).unsqueeze(0)
            
        if not args.chunks:
            backprojector= FlexibleBackprojection(
                crd_canon=crd_canon,
                indices_file=indices_file,
                particles_file=particles_file,
                poses_file=poses_file,
                ctf_file=ctf_file,
                coordinates_file=coordinates_file,
                coefs_file=coefs_file,
                lazy=lazy,
                rigid=rigid,
                pixel_size=pixel_size,
                gaussian_threshold=gaussian_threshold,
                sigma=sigma,
            )

            (       
                volume, 
                volume_half1, 
                volume_half2,  
                volume_reg, 
                volume_reg_half1, 
                volume_reg_half2
            
            ) = backprojector.run_backproj(batch_size=batch_size, eps=eps)
        else:

            coordinates_files = glob.glob(coordinates_file)
            indices_files = glob.glob(indices_file)
            coordinates_files.sort()
            indices_files.sort()
            N_chunks = len(coordinates_files)

            print("Running flexbile backprojections for %i chunks"%N_chunks)
            assert N_chunks == len(indices_files)

            volumes = None

            for (coordinates_file, indices_file) in zip(coordinates_files, indices_files):

                backprojector= FlexibleBackprojection(
                    crd_canon=crd_canon,
                    indices_file=indices_file,
                    particles_file=particles_file,
                    poses_file=poses_file,
                    ctf_file=ctf_file,
                    coordinates_file=coordinates_file,
                    coefs_file=coefs_file,
                    lazy=lazy,
                    rigid=rigid,
                    pixel_size=pixel_size,
                    gaussian_threshold=gaussian_threshold,
                    sigma=sigma,
                )

                v = backprojector.run_backproj(batch_size=batch_size, eps=eps)
                volumes = update_chunk_volumes(volumes, v)

            (       
                volume, 
                volume_half1, 
                volume_half2,  
                volume_reg, 
                volume_reg_half1, 
                volume_reg_half2
            
            ) = volumes


        write_mrc(output_dir+"/volume_unreg.mrc",volume, is_vol=True)
        write_mrc(output_dir+"/volume_unreg_half1.mrc",volume_half1, is_vol=True)
        write_mrc(output_dir+"/volume_unreg_half2.mrc",volume_half2, is_vol=True)
        write_mrc(output_dir+"/volume_counts.mrc",volume_reg, is_vol=True)
        write_mrc(output_dir+"/volume_counts_half1.mrc",volume_reg_half1, is_vol=True)
        write_mrc(output_dir+"/volume_counts_half2.mrc",volume_reg_half2, is_vol=True)

        pixel_size = backprojector.pixel_size

    else:
        print("Skipping backprojection, starting regularization from volumes %s ..."%reg_volumes_prefix)
        volume, _ = parse_mrc(reg_volumes_prefix+"/volume_unreg.mrc")
        volume_half1, _ = parse_mrc(reg_volumes_prefix+"/volume_unreg_half1.mrc")
        volume_half2, _ = parse_mrc(reg_volumes_prefix+"/volume_unreg_half2.mrc")
        volume_reg, _ = parse_mrc(reg_volumes_prefix+"/volume_counts.mrc")
        volume_reg_half1, _ = parse_mrc(reg_volumes_prefix+"/volume_counts_half1.mrc")
        volume_reg_half2, _ = parse_mrc(reg_volumes_prefix+"/volume_counts_half2.mrc")

        volume = torch.tensor(volume, device=device)
        volume_half1 = torch.tensor(volume_half1, device=device)
        volume_half2 = torch.tensor(volume_half2, device=device)
        volume_reg = torch.tensor(volume_reg, device=device)
        volume_reg_half1 = torch.tensor(volume_reg_half1, device=device)
        volume_reg_half2 = torch.tensor(volume_reg_half2, device=device)

    volume = regularize_volume(volume, volume_reg, wiener_constant)
    volume_half1 = regularize_volume(volume_half1, volume_reg_half1, wiener_constant)
    volume_half2 = regularize_volume(volume_half2, volume_reg_half2, wiener_constant)

    write_mrc(output_dir+"/backproject.mrc",volume, is_vol=True)
    write_mrc(output_dir+"/backproject_half1.mrc",volume_half1, is_vol=True)
    write_mrc(output_dir+"/backproject_half2.mrc",volume_half2, is_vol=True)


    # if not rigid:
    #     truncation = gaussian_halfwidth_voxels(backprojector.sigma, backprojector.pixel_size, p=backprojector.gaussian_threshold*4) # 4*sigma truncation 
    #     vox_mask_canon = get_voxel_mask(backprojector.crd_canon, backprojector.Dr, backprojector.pixel_size,  truncation)
    #     vol_mask = gaussian_kernel_mask(vox_mask_canon[0], backprojector.Dr, device)[-1]
    #     binary_vol_mask = vol_mask.clone()
    #     binary_vol_mask[binary_vol_mask>eps] = 1.0
    #     binary_vol_mask[binary_vol_mask<=eps] = 0.0

    #     volume *= binary_vol_mask.float()
    #     volume_half1 *= binary_vol_mask.float()
    #     volume_half2 *= binary_vol_mask.float()


    #     write_mrc(output_dir+"/backproject_masked.mrc",volume, is_vol=True)
    #     write_mrc(output_dir+"/backproject_masked_half1.mrc",volume_half1, is_vol=True)
    #     write_mrc(output_dir+"/backproject_masked_half2.mrc",volume_half2, is_vol=True)
    #     write_mrc(output_dir+"/backproject_mask.mrc",binary_vol_mask.float(), is_vol=True)

    # # Output filter to mimic the trilinear voxel interpolation in Fourier backprojection
    # output_filter_sigma = (3**0.5) / (6**0.5) # 3x 1D trilinear blur 1/6**0.5 = 0.7 subvoxel
    # volume = gaussian_filter3d(volume[None, None], 5, output_filter_sigma)[-1,-1]
    # volume_half1 = gaussian_filter3d(volume_half1[None, None], 5, output_filter_sigma)[-1,-1]
    # volume_half2 = gaussian_filter3d(volume_half2[None, None], 5, output_filter_sigma)[-1,-1]

    # write_mrc(output_dir+"/backproject.mrc",volume, is_vol=True)
    # write_mrc(output_dir+"/backproject_half1.mrc",volume_half1, is_vol=True)
    # write_mrc(output_dir+"/backproject_half2.mrc",volume_half2, is_vol=True)        

    # if not args.rigid:

    #     mean_coords = traj.mean(dim=0, keepdim=True)         # [1, N, 3]
    #     fluctuations = traj - mean_coords                    # [T, N, 3]
    #     rmsf = torch.sqrt((fluctuations ** 2).sum(dim=-1).mean(dim=0)) # [T, N, 3]
    #     volume = normalize_volume(volume, crd_canon, vox_mask_canon, vol_mask, rmsf, Dr, sigma, pixel_size, coefs)
    #     volume_half1 = normalize_volume(volume, crd_canon, vox_mask_canon, vol_mask, rmsf, Dr, sigma, pixel_size, coefs)
    #     volume_half2 = normalize_volume(volume, crd_canon, vox_mask_canon, vol_mask, rmsf, Dr, sigma, pixel_size, coefs)

    #     write_mrc(output_dir+"/backproject_normalized.mrc",volume, is_vol=True)
    #     write_mrc(output_dir+"/backproject_normalized_half1.mrc",volume_half1, is_vol=True)
    #     write_mrc(output_dir+"/backproject_normalized_half2.mrc",volume_half2, is_vol=True)


    fsc_curve,freqs = fourier_shell_correlation(volume_half1,volume_half2,None, apix=pixel_size)
    res_05, res_0143 = fsc_thresh(fsc_curve,freqs )

    fig, ax = plt.subplots(1,1)
    ax.plot(freqs.cpu().numpy(), fsc_curve.cpu().numpy())
    def fraction_formatter(x, pos):
        if x == 0:
            return "0"
        return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places
    ax.xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
    ax.axhline(0.143, c="red")
    ax.axhline(0.5, c="green")
    ax.axvline(1/res_05, c="red")
    ax.axvline(1/res_0143, c="green")
    ax.set_xlabel("Resolution ($1/\AA$)")
    ax.set_ylabel("Fourier Shell Correlation")
    ax.set_title("AVG FSC Resolution %.2f $\AA$ (%.2f $\AA$)"%(res_0143, res_05))
    fig.savefig(output_dir+"/fsc.png")
    plt.close(fig)


    _ = calculate_cryosparc_fscs(
            volume.cpu(),
            volume_half1.cpu(),
            volume_half2.cpu(),
            apix=pixel_size,
            out_file=output_dir+"/fsc_cs.txt",
            plot_file=output_dir+"/fsc_cs.png",
        )


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument("--coordinates",type=os.path.abspath,default=None,help="TODO",)
    parser.add_argument("--reference",type=os.path.abspath,default=None,help="TODO",)
    parser.add_argument("-o","--output_dir", type=os.path.abspath,help="TODO",)
    parser.add_argument("--pixel_size",type=float, help="TODO")

    parser.add_argument("--indices",type=os.path.abspath,default=None,help="TODO",)
    parser.add_argument("--coefs",type=os.path.abspath,default=None,help="TODO",)

    parser.add_argument( "--batch_size",type=int,help="TODO",)
    parser.add_argument("--rigid",action="store_true", help="TODO")
    parser.add_argument("--lazy",action="store_true", help="TODO")
    parser.add_argument("--chunks",action="store_true", help="TODO")
    parser.add_argument("--wiener_constant",type=float,default=1.0, help="TODO")
    parser.add_argument("--sigma",type=float,default=5.0, help="TODO")
    parser.add_argument("--gaussian_threshold",type=float,default=0.9, help="TODO")
    parser.add_argument("--particles",type=os.path.abspath,default=None, help="TODO")
    parser.add_argument("--poses",type=os.path.abspath,default=None, help="TODO")
    parser.add_argument("--ctf",type=os.path.abspath,default=None, help="TODO")
    parser.add_argument("--eps",type=float,default=1e-6, help="TODO")


    parser.add_argument("--volumes",type=os.path.abspath,default=None, help="TODO")
    
    return parser



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)