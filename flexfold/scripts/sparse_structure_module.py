
from functools import reduce
import importlib
import math
import sys
from operator import mul

import torch
import torch.nn as nn
from typing import Optional, Tuple, Sequence, Union

from openfold.model.primitives import Linear, LayerNorm, ipa_point_weights_init_
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from openfold.utils.geometry.quat_rigid import QuatRigid
from openfold.utils.geometry.rigid_matrix_vector import Rigid3Array
from openfold.utils.geometry.vector import Vec3Array, square_euclidean_distance
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.precision_utils import is_fp16_enabled
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)
from openfold.model.structure_module import AngleResnetBlock, AngleResnet, PointProjection, BackboneUpdate,StructureModuleTransitionLayer, StructureModuleTransition, StructureModule

attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")
import matplotlib.pyplot as plt
import math

def sparse_softmax(scores, denom_bias, dim):
    # scores: (N, M,)

    # Compute max per row for stability
    max_per_row = torch.max(scores, dim=dim, keepdim=True).values
    max_per_row =0.0
    scores_exp = torch.exp(scores - max_per_row)

    # Sum exp over valid connections
    denom = torch.sum(scores_exp, dim=dim, keepdim=True) +denom_bias

    return scores_exp / denom, 1/denom


class SparseIPA(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        is_multimer: bool = False,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(SparseIPA, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.is_multimer = is_multimer

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=(not is_multimer))

        self.linear_q_points = PointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
            self.is_multimer
        )

        if(is_multimer):
            self.linear_k = Linear(self.c_s, hc, bias=False)
            self.linear_v = Linear(self.c_s, hc, bias=False)
            self.linear_k_points = PointProjection(
                self.c_s,
                self.no_qk_points,
                self.no_heads,
                self.is_multimer
            )

            self.linear_v_points = PointProjection(
                self.c_s,
                self.no_v_points,
                self.no_heads,
                self.is_multimer
            )
        else:
            self.linear_kv = Linear(self.c_s, 2 * hc)
            self.linear_kv_points = PointProjection(
                self.c_s,
                self.no_qk_points + self.no_v_points,
                self.no_heads,
                self.is_multimer
            )

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Union[Rigid, Rigid3Array],
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
        it=None,
        amask=None
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if (_offload_inference and inplace_safe):
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))


        # [*, N_res, H, P_qk]
        q_pts = self.linear_q_points(s, r)

        # The following two blocks are equivalent
        # They're separated only to preserve compatibility with old AF weights
        if(self.is_multimer):
            # [*, N_res, H * C_hidden]
            k = self.linear_k(s)
            v = self.linear_v(s)

            # [*, N_res, H, C_hidden]
            k = k.view(k.shape[:-1] + (self.no_heads, -1))
            v = v.view(v.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, P_qk, 3]
            k_pts = self.linear_k_points(s, r)

            # [*, N_res, H, P_v, 3]
            v_pts = self.linear_v_points(s, r)
        else:
            # [*, N_res, H * 2 * C_hidden]
            kv = self.linear_kv(s)

            # [*, N_res, H, 2 * C_hidden]
            kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, C_hidden]
            k, v = torch.split(kv, self.c_hidden, dim=-1)

            kv_pts = self.linear_kv_points(s, r)

            # [*, N_res, H, P_q/P_v, 3]
            k_pts, v_pts = torch.split(
                kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
            )

        # [*, N_res, M, H, C_hidden]
        q_masked = permute_final_dims(q[..., amask[...,0], amask[...,2], :], (1,2,0,3))
        k_masked = permute_final_dims(k[..., amask[...,1], amask[...,2], :], (1,2,0,3))
        v_masked = permute_final_dims(v[..., amask[...,1], amask[...,2], :], (1,2,0,3))
        # [*, N_res, M, H, P_q/P_v, 3]
        q_pts_masked = permute_final_dims(q_pts[..., amask[...,0], amask[...,2], :, :], (1,2,0,3,4))
        k_pts_masked = permute_final_dims(k_pts[..., amask[...,1], amask[...,2], :, :], (1,2,0,3,4))
        v_pts_masked = permute_final_dims(v_pts[..., amask[...,1], amask[...,2], :, :], (1,2,0,3,4))
        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b( z[0])
        b_mask=permute_final_dims(b[..., amask[...,0], amask[...,1], amask[...,2]], (1,2,0))

        if (_offload_inference):
            assert (sys.getrefcount(z[0]) == 2)
            z[0] = z[0].cpu()

        a = torch.sum(k_masked*q_masked, dim=-1)
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * b_mask)

        a = permute_final_dims(a, (2,0,1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts_masked - k_pts_masked

        if (inplace_safe):
            pt_att *= pt_att
        else:
            pt_att = pt_att ** 2

        pt_att = sum(torch.unbind(pt_att, dim=-1))

        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )

        if (inplace_safe):
            pt_att *= head_weights
        else:
            pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = square_mask[..., amask[..., 0], amask[..., 1]]
        square_mask = self.inf * (square_mask - 1)


        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        if (inplace_safe):
            raise NotImplementedError()
            a += pt_att
            del pt_att
            a += square_mask.unsqueeze(-3)
            # in-place softmax
            attn_core_inplace_cuda.forward_(
                a,
                reduce(mul, a.shape[:-1]),
                a.shape[-1],
            )
        else:
            # a_tmp = a
            a = a + pt_att
            a = a + square_mask

            #[*, H,N,N]
            # M = amask.shape[-2]
            # N = amask.shape[-3]
            # bias = (N-M)*torch.exp(a.min(dim=-1, keepdim=True).values)
            # a, a_fill = sparse_softmax(a,bias, -1)
            a = self.softmax(a)

        # fig, ax = plt.subplots(4,12, figsize=(30,10))
        # for j, im in enumerate([a_tmp,torch.exp(pt_att),a]):
        #     tmp = torch.zeros(12, N,N, device=a.device)
        #     tmp[amask[..., 2], amask[..., 0], amask[..., 1]] = im
        #     for i in range(12):
        #         ax[j,i].imshow(tmp[i].detach().cpu().numpy(), cmap="jet")

        # fig.savefig("../cryofold/test_mask%i.png"%it, dpi=150)
        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.sum(permute_final_dims(a, (1,2,0))[..., None] * v_masked, dim=-3)

        # o = o + permute_final_dims(a_fill, (1,0,2)) * v.sum(dim=-3, keepdim=True)  
        # o = o - permute_final_dims(a_fill, (1,0,2)) * v_masked.sum(dim=-3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        if (inplace_safe):
            raise NotImplementedError()
            v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
            o_pt = [
                torch.matmul(a, v.to(a.dtype))
                for v in torch.unbind(v_pts, dim=-3)
            ]
            o_pt = torch.stack(o_pt, dim=-3)
        else:   
            o_pt = torch.sum(a[..., None, None] * permute_final_dims(v_pts_masked, (2,0,1,3,4)), dim=-3)

            # o_pt = o_pt + a_fill[..., None] * permute_final_dims(v_pts.sum(dim=-4, keepdim=True), (1,0,2,3))
            # o_pt = o_pt - a_fill[..., None] * permute_final_dims(v_pts_masked.sum(dim=-4), (1,0,2,3))
            o_pt = permute_final_dims(o_pt, (0,3,1,2))


        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))

        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        o_pt = torch.unbind(o_pt, dim=-1)

        if (_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z]
        z_mask = z[0][..., amask[..., 0], amask[..., 1], :]
        o_pair =  torch.sum(a[..., None] * z_mask.to(dtype=a.dtype), dim=-2)
        o_pair = permute_final_dims(o_pair, (1,0,2))
        # a_fill = permute_final_dims(a_fill, (1,0,2)) 
        # o_pair = o_pair + a_fill * z[0].sum(dim=-2, keepdim=True)  
        # o_pair = o_pair - a_fill * z_mask.sum(dim=-2, keepdim=True)

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                (o, *o_pt, o_pt_norm, o_pair), dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s


class SparseIPAMultimer(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        is_multimer: bool = True,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(SparseIPAMultimer, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)

        self.linear_q_points = PointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
            is_multimer=True
        )

        self.linear_k = Linear(self.c_s, hc, bias=False)
        self.linear_v = Linear(self.c_s, hc, bias=False)
        self.linear_k_points = PointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
            is_multimer=True
        )

        self.linear_v_points = PointProjection(
            self.c_s,
            self.no_v_points,
            self.no_heads,
            is_multimer=True
        )

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-2)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Union[Rigid, Rigid3Array],
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
        amask=None
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if(_offload_inference and inplace_safe):
            z = _z_reference_list
        else:
            z = [z]

        a = 0.

        point_variance = (max(self.no_qk_points, 1) * 9.0 / 2)
        point_weights = math.sqrt(1.0 / point_variance)

        softplus = lambda x: torch.logaddexp(x, torch.zeros_like(x))

        head_weights = softplus(self.head_weights)
        point_weights = point_weights * head_weights

        #######################################
        # Generate scalar and point activations
        #######################################

        # [*, N_res, H, P_qk]
        q_pts = Vec3Array.from_array(self.linear_q_points(s, r))

        # [*, N_res, H, P_qk, 3]
        k_pts = Vec3Array.from_array(self.linear_k_points(s, r))

        pt_att = square_euclidean_distance(q_pts.unsqueeze(-3), k_pts.unsqueeze(-4), epsilon=0.)
        pt_att = torch.sum(pt_att * point_weights[..., None], dim=-1) * (-0.5)
        pt_att = pt_att.to(dtype=s.dtype)
        a = a + pt_att

        scalar_variance = max(self.c_hidden, 1) * 1.
        scalar_weights = math.sqrt(1.0 / scalar_variance)

        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        k = self.linear_k(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))

        q = q * scalar_weights
        a = a + torch.einsum('...qhc,...khc->...qkh', q, k)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if (_offload_inference):
            assert (sys.getrefcount(z[0]) == 2)
            z[0] = z[0].cpu()

        a = a + b

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        a = a + square_mask.unsqueeze(-1)
        a = a * math.sqrt(1. / 3)  # Normalize by number of logit terms (3)
        a = self.softmax(a)

        # [*, N_res, H * C_hidden]
        v = self.linear_v(s)

        # [*, N_res, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        o = torch.einsum('...qkh, ...khc->...qhc', a, v)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_res, H, P_v, 3]
        v_pts = Vec3Array.from_array(self.linear_v_points(s, r))

        # [*, N_res, H, P_v]
        o_pt = v_pts[..., None, :, :, :] * a.unsqueeze(-1)
        o_pt = o_pt.sum(dim=-3)
        # o_pt = Vec3Array(
        #     torch.sum(a.unsqueeze(-1) * v_pts[..., None, :, :, :].x, dim=-3),
        #     torch.sum(a.unsqueeze(-1) * v_pts[..., None, :, :, :].y, dim=-3),
        #     torch.sum(a.unsqueeze(-1) * v_pts[..., None, :, :, :].z, dim=-3),
        # )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(o_pt.shape[:-2] + (-1,))

        # [*, N_res, H, P_v]
        o_pt = r[..., None].apply_inverse_to_point(o_pt)
        o_pt_flat = [o_pt.x, o_pt.y, o_pt.z]
        o_pt_flat = [x.to(dtype=a.dtype) for x in o_pt_flat]

        # [*, N_res, H * P_v]
        o_pt_norm = o_pt.norm(epsilon=1e-8)

        if (_offload_inference):
            z[0] = z[0].to(o_pt.x.device)

        o_pair = torch.einsum('...ijh, ...ijc->...ihc', a, z[0].to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                (o, *o_pt_flat, o_pt_norm, o_pair), dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s




class SparseStructureModule(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        is_multimer=False,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super(SparseStructureModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf
        self.is_multimer = is_multimer

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        ipa = SparseIPA #if not self.is_multimer else SparseIPAMultimer
        self.ipa = ipa(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
            is_multimer=self.is_multimer,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        if self.is_multimer:
            self.bb_update = QuatRigid(self.c_s, full_quat=False)
        else:
            self.bb_update = BackboneUpdate(self.c_s)

        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
        amask=None
    ):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        if(self.is_multimer):
            outputs = self._forward_multimer(evoformer_output_dict, aatype, mask, inplace_safe, _offload_inference,  amask)
        else:
            outputs = self._forward_monomer(evoformer_output_dict, aatype, mask, inplace_safe, _offload_inference, amask)

        return outputs
    

    def _forward_multimer(
            self,
            evoformer_output_dict,
            aatype,
            mask=None,
            inplace_safe=False,
            _offload_inference=False,
            amask=None
    ):
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if (_offload_inference):
            assert (sys.getrefcount(evoformer_output_dict["pair"]) == 2)
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid3Array.identity(
            s.shape[:-1], 
            s.device, 
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s,
                z,
                rigids,
                mask,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
                _z_reference_list=z_reference_list,
                amask=amask,
                it=i
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids @ self.bb_update(s)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                rigids.scale_translation(self.trans_scale_factor),
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )
            
            preds = {
                "frames": rigids.scale_translation(self.trans_scale_factor).to_tensor(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
            }

            preds = {k: v.to(dtype=s.dtype) for k, v in preds.items()}

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        if (_offload_inference):
            evoformer_output_dict["pair"] = (
                evoformer_output_dict["pair"].to(s.device)
            )

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def _forward_monomer(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
        amask=None
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if (_offload_inference):
            assert (sys.getrefcount(evoformer_output_dict["pair"]) == 2)
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(
            s.shape[:-1], 
            s.dtype, 
            s.device, 
            self.training,
            fmt="quat",
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s, 
                z, 
                rigids, 
                mask, 
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference, 
                _z_reference_list=z_reference_list,
                it=i,
                amask=amask
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)
           
            # [*, N]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(
                    rot_mats=rigids.get_rots().get_rot_mats(), 
                    quats=None
                ),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(
                self.trans_scale_factor
            )

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)
            
            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        if (_offload_inference):
            evoformer_output_dict["pair"] = (
                evoformer_output_dict["pair"].to(s.device)
            )

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs


    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.dtype, r.device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.structure_module =SparseStructureModule(
        # self.structure_module =StructureModule(
                c_s=384,
                c_z=128,
                c_ipa=16,
                c_resnet=128,
                no_heads_ipa=12,
                no_qk_points=4,
                no_v_points=8,
                dropout_rate=0.1,
                no_blocks=8,
                no_transition_layers=1,
                no_resnet_blocks=2,
                no_angles=7,
                trans_scale_factor=20,
                epsilon=1e-8,
                inf=1e5,
                is_multimer=True,
            )
    def forward(self, *args, **kwargs):
        return self.structure_module(*args, **kwargs)



def mask_from_ca(crd, nmask= 100):
    N = crd.shape[-2]
    dist = torch.sqrt(((crd[..., None,:, :] - crd[..., :,None, :])**2).sum(dim=-1))
    _, cols = torch.topk(dist, k=nmask, dim=-1, largest=False)
    rows = torch.arange(N, device=dist.device).unsqueeze(1).expand_as(cols)
    return torch.stack((rows, cols), dim=-1) , dist # shape (N, M, 2)


def get_per_head_mask(mask, M):
    # full_mask=[
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, M ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10).argsort(dim=1)[:, :(M-10)]]), dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10).argsort(dim=1)[:, :(M-10)]]), dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, M ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10).argsort(dim=1)[:, :(M-10)]]), dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N//2).argsort(dim=1)[:, :(M-10)]]), dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N//2).argsort(dim=1)[:, :(M-10)]]), dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N//2).argsort(dim=1)[:, :(M-10)]]), dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N//2).argsort(dim=1)[:, :(M-10)]]), dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N//2).argsort(dim=1)[:, :(M-10)]]), dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N//2).argsort(dim=1)[:, :(M-10)]]), dim=1),
    #     torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N//2).argsort(dim=1)[:, :(M-10)]]), dim=1),
    # ]

    full_mask = [
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),
        torch.cat((mask[:,:10], mask[:,10:][torch.arange(N).unsqueeze(1), torch.rand(N, N-10 ).argsort(dim=1)[:, :(M-10)]]),  dim=1),

    ]
    full_mask = torch.cat(full_mask).reshape(12, N, M, 2)
    heads = torch.arange(12, device=mask.device).expand( N,M, 12).permute(2, 0, 1)
    full_mask = torch.cat((full_mask, heads[..., None]), dim=-1)
    return full_mask


from openfold.utils.import_weights import convert_deprecated_v1_keys
from flexfold.core import output_single_pdb
from openfold.utils.feats import (
    atom14_to_atom37,
)
from openfold.utils.loss import fape_loss, compute_renamed_ground_truth
import time
from flexfold.models import import_weights,embeddings_keys
from openfold.config import model_config
from openfold.data import data_transforms

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
device="cuda"

###############################################################
#MODEL INIT
###############################################################
model = Model().to(device)
# model = import_weights(model, "../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt")
model = import_weights(model, "../openfold/openfold/resources/params/params_model_1_multimer_v3.npz")

config = model_config(
    "model_1_multimer_v3", 
    train=True, 
    low_prec=False,
) 

###############################################################
#DATA INIT
###############################################################
# Embeddings
# embeddings = torch.load("data/cryofold/spike-md/embeddings_crop_mask.pt", map_location="cpu")
embeddings = torch.load("data/cryofold/jillsData/pred2/embeddings_fixed.pt", map_location="cpu")
embeddings = {k: torch.tensor(v) for k, v in embeddings.items() if k in embeddings_keys.keys()}
embeddings = {k:v.to(device) for k,v in embeddings.items()}
embeddings["pair"].requires_grad=True
embeddings["single"].requires_grad=True
N = embeddings["pair"].shape[-2]

mask, dist = mask_from_ca(embeddings["final_atom_positions"][..., 1, :], nmask=N)
# #### Plots mask
# maskval = torch.zeros((12, N,N), device=device)
# maskval[full_mask[..., 2], full_mask[..., 0], full_mask[..., 1]] =1
# fig, ax = plt.subplots(1,12, figsize=(30,5))
# for i in range(12):
#     ax[i].imshow(maskval[i].cpu().numpy())
#     # ax[1].imshow(dist.cpu().numpy())
# fig.savefig("../cryofold/test_mask.png", dpi=150)
# #### 

def make_gt(feats):
    feats["all_atom_positions"] = feats["final_atom_positions"] 
    feats["all_atom_mask"] = feats["final_atom_mask"] 
    return feats
fc = [
    # data_transforms.make_fixed_size(embeddings_keys,0,0,500,0),
    make_gt,
    data_transforms.make_atom14_positions,
    data_transforms.atom37_to_frames,
    data_transforms.atom37_to_torsion_angles(""),
    data_transforms.make_pseudo_beta(""),
    data_transforms.get_backbone_frames,
    data_transforms.get_chi_angles,
]
for f in fc:
    embeddings = f(embeddings)
###############################################################
#OPtimizer loop
###############################################################
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs=1000
M = N -11
M_limit = 50
for i in range(n_epochs):
    optimizer.zero_grad()         

    outputs = {}
    outputs["sm"]  = model(
            {
                "pair": embeddings["pair"],
                "single":embeddings["single"]
            },
            embeddings["aatype"],
            mask=embeddings["seq_mask"],
            inplace_safe=False,
            _offload_inference=False,
            amask = get_per_head_mask(mask, M)

        )
    outputs["final_atom_positions"] = atom14_to_atom37(
        outputs["sm"]["positions"][-1], embeddings
    )
    outputs["final_atom_mask"] = embeddings["atom37_atom_exists"]
    outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

    embeddings.update(
        compute_renamed_ground_truth(
            embeddings,
            outputs["sm"]["positions"][-1],
        )
    )

    loss = fape_loss(
        out =outputs,
        batch = embeddings,
        config = config.loss.fape)
    loss.backward()
    optimizer.step()

    print("Iter=%i; M=%i; Loss=%.2f"%(i,M, loss.item()))

    if M>= M_limit:
        M-=1


    if i%50 == 0:
        output_single_pdb(all_atom_positions=outputs["final_atom_positions"].detach().cpu().numpy(),
                        all_atom_mask=outputs["final_atom_mask"].detach().cpu().numpy(),
                        aatype=embeddings["aatype"].detach().cpu().numpy(),
                        file="../cryofold/SparseIPA/%s.pdb"%str(i+1).zfill(5),
                        chain_index=embeddings["asym_id"].detach().cpu().numpy(),
                        residue_index=embeddings["residue_index"].detach().cpu().numpy())

# dist = torch.sqrt(((final_atom_positions[None,:,1] - final_atom_positions[:,None,1])**2).sum(dim=-1))
