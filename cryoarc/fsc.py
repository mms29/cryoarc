# -----------------------------------------------------------------------------
# This file is derived from code in the cryoDRGN project:
#     https://github.com/zhonge/cryodrgn
#
# Original authors: cryoDRGN contributors
# License: GNU General Public License v3.0 (GPL-3.0)
#
# Modifications:
#   - Adapted and extended for CryoARC
#   - Modifications by Rémi Vuillemot, 2025
#
# This file remains subject to the GPL-3.0 license.
# See the LICENSE file in the original repository for details.
# -----------------------------------------------------------------------------




from typing import Sequence
import torch
from torch_grid_utils import fftfreq_grid
import einops
import torch

def _prepare_fft_data(
    a_fft: torch.Tensor,
    b_fft: torch.Tensor,
    frequency_grid: torch.Tensor,
    fft_mask: torch.Tensor | None,
    ndim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare FFT data by applying mask and flattening spatial dimensions.

    Args:
        a_fft: FFT of first tensor
        b_fft: FFT of second tensor
        frequency_grid: Frequency grid
        fft_mask: Optional mask for fft indices
        ndim: Number of spatial dimensions

    Returns
    -------
        Tuple of (a_fft_flat, b_fft_flat, frequencies)
    """
    if fft_mask is not None:
        # Apply mask to FFTs and frequency grid
        a_fft_flat = a_fft[..., fft_mask]  # (..., n_masked_freqs)
        b_fft_flat = b_fft[..., fft_mask]
        frequencies = frequency_grid[fft_mask]
    else:
        # Flatten spatial dimensions while preserving batch dimensions using einops
        if ndim == 2:
            # 2D case: (..., h, w) -> (..., h*w)
            a_fft_flat = einops.rearrange(a_fft, "... h w -> ... (h w)")
            b_fft_flat = einops.rearrange(b_fft, "... h w -> ... (h w)")
        elif ndim == 3:
            # 3D case: (..., d, h, w) -> (..., d*h*w)
            a_fft_flat = einops.rearrange(a_fft, "... d h w -> ... (d h w)")
            b_fft_flat = einops.rearrange(b_fft, "... d h w -> ... (d h w)")
        else:
            raise ValueError(f"Unsupported ndim: {ndim}. Only 2D and 3D are supported.")

        # Flatten frequency grid
        frequencies = torch.flatten(frequency_grid)

    return a_fft_flat, b_fft_flat, frequencies


def _compute_frequency_bins_weighted(
    image_shape: Sequence[int], device: torch.device
) -> torch.Tensor:
    """Compute frequency bin centers for weighted shell/ring correlation.

    Args:
        image_shape: Shape of the spatial dimensions
        device: Device to create tensors on

    Returns
    -------
        Frequency bin centers with shape (n_bins,)
    """
    # Use minimum dimension to define number of shells
    # This ensures all frequency components can contribute via interpolation
    min_dim = min(image_shape)
    bin_centers = torch.fft.rfftfreq(min_dim, device=device)

    return bin_centers


def fourier_shell_correlation(
    a: torch.Tensor, b: torch.Tensor, fft_mask: torch.Tensor | None = None, apix:float=1.0
) -> torch.Tensor:
    """Fourier shell correlation between two 3D images with batching.

    Supports both cubic and rectangular volumes using weighted interpolation.

    Args:
        a: Input tensor of shape (..., d, h, w)
        b: Input tensor of shape (..., d, h, w)
        fft_mask: Optional mask for fft, shape should match fft output

    Returns
    -------
        Correlation values of shape (broadcast(...), min(d, h, w) // 2 + 1)
    """
    # Input validation
    if a.ndim < 3:
        raise ValueError("Input tensors must have at least 3 dimensions.")
    if b.ndim < 3:
        raise ValueError("Input tensors must have at least 3 dimensions.")

    # Enforce that spatial dimensions match
    if a.shape[-3:] != b.shape[-3:]:
        raise ValueError(
            f"Spatial dimensions must match: a.shape[-3:] = {a.shape[-3:]} "
            f"vs b.shape[-3:] = {b.shape[-3:]}"
        )

    # Compute FFT
    spatial_dims_list = [-3, -2, -1]  # Last 3 dimensions
    a_fft = torch.fft.rfftn(a, dim=spatial_dims_list)
    b_fft = torch.fft.rfftn(b, dim=spatial_dims_list)
    
    # Get image shape (spatial dimensions)
    image_shape = a.shape[-3:]

    # Freqs 
    N =  a.shape[-1]//2 + 1
    freqs = torch.linspace(0,0.5,N, device=a.device) / apix

    return fourier_correlation(a_fft, b_fft, image_shape, fft_mask, rfft=True), freqs



def _weighted_normalized_cc_complex(
    a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Calculate weighted normalized cross correlation for batched tensors.

    Args:
        a: Complex tensor (..., n_freqs)
        b: Complex tensor (..., n_freqs)
        weights: Weight values (n_freqs,)

    Returns
    -------
        Weighted normalized correlation for each batch item
    """
    # Weighted correlation (sum over frequency dimension)
    correlation = torch.sum(weights * a * torch.conj(b), dim=-1)

    # Weighted norms
    norm_a = torch.sqrt(torch.sum(weights * torch.abs(a) ** 2, dim=-1))
    norm_b = torch.sqrt(torch.sum(weights * torch.abs(b) ** 2, dim=-1))

    # Normalize, handling potential division by zero
    denominator = norm_a * norm_b
    correlation = torch.where(
        denominator > 0, correlation / denominator, torch.zeros_like(correlation)
    )

    return correlation

def _frequency_to_bin_coordinates(
    frequencies: torch.Tensor, bin_centers: torch.Tensor
) -> torch.Tensor:
    """Convert frequencies to continuous bin coordinates using interpolation.

    Args:
        frequencies: Flattened frequency magnitudes
        bin_centers: Frequency bin centers

    Returns
    -------
        Continuous bin coordinates for each frequency
    """
    # Use searchsorted to find the bin each frequency would fall into
    bin_indices = torch.searchsorted(bin_centers, frequencies, right=True) - 1

    # Handle edge cases
    bin_indices = torch.clamp(bin_indices, 0, len(bin_centers) - 2)

    # Calculate fractional position within the bin
    left_centers = bin_centers[bin_indices]
    right_centers = bin_centers[bin_indices + 1]

    # Avoid division by zero for the last bin
    bin_width = right_centers - left_centers
    bin_width = torch.where(bin_width == 0, torch.ones_like(bin_width), bin_width)

    # Linear interpolation coordinate
    bin_coords = bin_indices.float() + (frequencies - left_centers) / bin_width

    return bin_coords

def _calculate_interpolation_weights(
    bin_coords: torch.Tensor, target_bin: int
) -> torch.Tensor:
    """Calculate linear interpolation weights for a target bin.

    Args:
        bin_coords: Continuous bin coordinates for each frequency
        target_bin: Target bin index

    Returns
    -------
        Interpolation weights for the target bin
    """
    # Distance from each frequency to the target bin
    distances = torch.abs(bin_coords - target_bin)

    # Linear interpolation: weight = max(0, 1 - distance)
    weights = torch.clamp(1.0 - distances, min=0.0)

    return weights

def _compute_shell_correlations_weighted(
    a_fft: torch.Tensor,
    b_fft: torch.Tensor,
    frequencies: torch.Tensor,
    bin_centers: torch.Tensor,
    batch_dims: Sequence[int],
) -> torch.Tensor:
    """Compute weighted normalized cross correlation for each shell using interpolation.

    Args:
        a_fft: Flattened FFT data for first tensor
        b_fft: Flattened FFT data for second tensor
        frequencies: Flattened frequency magnitudes
        bin_centers: Frequency bin centers
        batch_dims: Batch dimensions shape

    Returns
    -------
        Correlation values for each shell with shape (*batch_dims, n_shells)
    """
    n_bins = len(bin_centers)
    correlation_results = []

    # Convert frequencies to continuous bin coordinates
    bin_coords = _frequency_to_bin_coordinates(frequencies, bin_centers)

    # Process each shell using weighted interpolation
    for bin_idx in range(n_bins):
        if bin_idx == 0:
            # DC component always has FSC = 1.0
            correlation_results.append(torch.ones(batch_dims, device=a_fft.device))
        else:
            # Calculate weights for this bin
            weights = _calculate_interpolation_weights(bin_coords, bin_idx)

            # Only include components with non-zero weights
            mask = weights > 0
            if mask.sum() > 0:
                # Get weighted FFT values for all batch items
                a_masked = a_fft[..., mask]  # (..., n_nonzero_weights)
                b_masked = b_fft[..., mask]
                weights_nonzero = weights[mask]

                # Weighted normalized cross correlation
                correlation = _weighted_normalized_cc_complex(
                    a_masked, b_masked, weights_nonzero
                )
                correlation_results.append(correlation)
            else:
                # Empty shell
                correlation_results.append(torch.zeros(batch_dims, device=a_fft.device))

    # Stack results along last dimension
    return torch.stack(correlation_results, dim=-1)  # (..., n_shells)

def fourier_correlation(
    a_fft: torch.Tensor,
    b_fft: torch.Tensor,
    image_shape: Sequence[int],
    fft_mask: torch.Tensor | None = None,
    rfft: bool = True,
) -> torch.Tensor:
    """Compute fourier correlation from FFT data supporting rectangular shapes.

    Args:
        a_fft: (..., *fft_shape) tensor containing FFT of a (..., *image_shape) tensor
        b_fft: (..., *fft_shape) tensor containing FFT of a (..., *image_shape) tensor
        image_shape: Size of spatial dimensions before FFT (e.g. (h, w) or (d, h, w))
            Note: For real FFTs, fft_shape = (*image_shape[:-1], image_shape[-1]//2 + 1)
        fft_mask: Optional mask for fft indices
        rfft: Whether the FFT data is from rfft (True) or fft (False)

    Returns
    -------
        Fourier correlation values with shape (..., n_shells)
    """
    # Input validation - check that FFT tensors can be broadcast together
    try:
        # This will raise an error if tensors can't be broadcast
        torch.broadcast_shapes(a_fft.shape, b_fft.shape)
    except RuntimeError as e:
        raise ValueError(f"FFT tensors must be broadcastable: {e}") from e

    # Validate fft_mask
    fft_shape = a_fft.shape[-len(image_shape) :]
    if fft_mask is not None and fft_mask.shape != fft_shape:
        raise ValueError("fft_mask must have same shape as fft output.")

    # Compute frequency grid and prepare FFT data
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=False,
        norm=True,
        device=a_fft.device,
    )

    # Apply mask and flatten spatial dimensions
    a_fft_flat, b_fft_flat, frequencies = _prepare_fft_data(
        a_fft, b_fft, frequency_grid, fft_mask, len(image_shape)
    )

    # Compute frequency bins using weighted approach
    bin_centers = _compute_frequency_bins_weighted(image_shape, a_fft.device)

    # Compute broadcast batch dimensions for correlation computation
    broadcast_shape = torch.broadcast_shapes(a_fft.shape, b_fft.shape)
    batch_dims = broadcast_shape[: -len(image_shape)]

    # Compute correlations using weighted interpolation
    correlations = _compute_shell_correlations_weighted(
        a_fft_flat,
        b_fft_flat,
        frequencies,
        bin_centers,
        batch_dims,
    )

    return torch.real(correlations)



def fsc_auc(fsc, freqs):
    return torch.trapz(fsc, freqs).item()

def fsc_thresh(fsc, freqs):
    return (1/freqs[fsc>=0.5].max()).item(), (1/freqs[fsc>=0.143].max()).item()


def fourier_mask(D, cutoff=0.45, smooth_width=0.05, device="cpu"):
    """
    Generate a boolean spherical frequency mask matching rfftn output shape.

    Parameters
    ----------
    D : int
        Volume dimension (e.g., 128).
    cutoff : float
        Normalized frequency radius (0–0.5) beyond which mask = False.
    smooth_width : float
        If > 0, defines a soft transition band where True/False are randomized.
        Default 0 for sharp cutoff.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    mask : torch.BoolTensor, shape (D, D, D//2 + 1)
    """
    nx, ny, nz = D, D, D // 2 + 1

    # Frequency grids (normalized 0–0.5)
    fx = torch.fft.fftfreq(nx, d=1.0, device=device)[:, None, None]
    fy = torch.fft.fftfreq(ny, d=1.0, device=device)[None, :, None]
    fz = torch.fft.rfftfreq(D, d=1.0, device=device)[None, None, :]

    # Radial frequency
    fr = torch.sqrt(fx**2 + fy**2 + fz**2)

    # Boolean mask
    mask = fr <= cutoff

    # Optional fuzzy band for smooth_width > 0 (optional)
    if smooth_width > 0:
        lower = cutoff - smooth_width / 2
        upper = cutoff + smooth_width / 2
        transition = (fr > lower) & (fr < upper)
        rand = torch.rand_like(fr)
        mask[transition] = rand[transition] < (
            (upper - fr[transition]) / smooth_width
        )

    return mask

def spherical_soft_mask(N: int, radius: float = 0.90, edge: float = 0.1, device=None):
    """
    Create a 3D smooth spherical mask of size (N, N, N).

    Args:
        N (int): Cube dimension.
        radius (float): Core radius as a fraction of half-box (0–1 range).
                        Example: 0.45 → 90% of box diameter.
        edge (float): Soft edge width as a fraction of half-box (0–1 range).
                      A typical choice: 0.05.
        device: torch device.

    Returns:
        mask (torch.Tensor): shape (N, N, N), float32, values in [0, 1].
    """
    if device is None:
        device = torch.device("cpu")

    # coordinate grid centered at 0
    lin = torch.linspace(-1, 1, N, device=device)
    z, y, x = torch.meshgrid(lin, lin, lin, indexing='ij')
    r = torch.sqrt(x**2 + y**2 + z**2)

    core = radius
    ramp_start = core
    ramp_end = core + edge

    mask = torch.zeros_like(r)
    inside = r <= ramp_start
    transition = (r > ramp_start) & (r < ramp_end)

    mask[inside] = 1.0
    # cosine falloff in edge region
    mask[transition] = 0.5 * (1 + torch.cos(
        torch.pi * (r[transition] - ramp_start) / edge
    ))
    # outside ramp_end stays 0
    return mask.float()
