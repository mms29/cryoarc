# Copyright 2025 Anonymous
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# remi : beartype>=0.20.2

import torch
import torch.nn as nn
from typing import Optional, List, Sequence
import math
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
from dataclasses import dataclass


import torch
import torch.nn as nn
from typing import Optional, Callable
import math
import numpy as np
from scipy.stats import truncnorm




from openfold.model.structure_module import  StructureModuleTransition, BackboneUpdate, AngleResnet

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
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)

attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")



# class PointProjection(nn.Module):
#     def __init__(self,
#         c_hidden: int,
#         num_points: int,
#         no_heads: int,
#         is_multimer: bool,
#         return_local_points: bool = False,
#     ):
#         super().__init__()
#         self.return_local_points = return_local_points
#         self.no_heads = no_heads
#         self.num_points = num_points
#         self.is_multimer = is_multimer

#         # Multimer requires this to be run with fp32 precision during training
#         precision = torch.float32 if self.is_multimer else None
#         self.linear = Linear(c_hidden, no_heads * 3 * num_points, precision=precision)

#     def forward(self, 
#         activations: torch.Tensor, 
#         rigids: Union[Rigid, Rigid3Array],
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         # TODO: Needs to run in high precision during training
#         points_local = self.linear(activations)
#         out_shape = points_local.shape[:-1] + (self.no_heads, self.num_points, 3)

#         if self.is_multimer:
#             points_local = points_local.view(
#                 points_local.shape[:-1] + (self.no_heads, -1)
#             )

#         points_local = torch.split(
#             points_local, points_local.shape[-1] // 3, dim=-1
#         )

#         points_local = torch.stack(points_local, dim=-1).view(out_shape)

#         points_global = rigids[..., None, None].apply(points_local)

#         if(self.return_local_points):
#             return points_global, points_local

#         return points_global


# class InvariantPointAttention(nn.Module):
#     """
#     Implements Algorithm 22.
#     """
#     def __init__(
#         self,
#         c_s: int,
#         c_z: int,
#         c_hidden: int,
#         no_heads: int,
#         no_qk_points: int,
#         no_v_points: int,
#         inf: float = 1e5,
#         eps: float = 1e-8,
#         is_multimer: bool = False,
#     ):
#         """
#         Args:
#             c_s:
#                 Single representation channel dimension
#             c_z:
#                 Pair representation channel dimension
#             c_hidden:
#                 Hidden channel dimension
#             no_heads:
#                 Number of attention heads
#             no_qk_points:
#                 Number of query/key points to generate
#             no_v_points:
#                 Number of value points to generate
#         """
#         super(InvariantPointAttention, self).__init__()

#         self.c_s = c_s
#         self.c_z = c_z
#         self.c_hidden = c_hidden
#         self.no_heads = no_heads
#         self.no_qk_points = no_qk_points
#         self.no_v_points = no_v_points
#         self.inf = inf
#         self.eps = eps
#         self.is_multimer = is_multimer

#         # These linear layers differ from their specifications in the
#         # supplement. There, they lack bias and use Glorot initialization.
#         # Here as in the official source, they have bias and use the default
#         # Lecun initialization.
#         hc = self.c_hidden * self.no_heads
#         self.linear_q = Linear(self.c_s, hc, bias=(not is_multimer))

#         self.linear_q_points = PointProjection(
#             self.c_s,
#             self.no_qk_points,
#             self.no_heads,
#             self.is_multimer
#         )

#         if(is_multimer):
#             self.linear_k = Linear(self.c_s, hc, bias=False)
#             self.linear_v = Linear(self.c_s, hc, bias=False)
#             self.linear_k_points = PointProjection(
#                 self.c_s,
#                 self.no_qk_points,
#                 self.no_heads,
#                 self.is_multimer
#             )

#             self.linear_v_points = PointProjection(
#                 self.c_s,
#                 self.no_v_points,
#                 self.no_heads,
#                 self.is_multimer
#             )
#         else:
#             self.linear_kv = Linear(self.c_s, 2 * hc)
#             self.linear_kv_points = PointProjection(
#                 self.c_s,
#                 self.no_qk_points + self.no_v_points,
#                 self.no_heads,
#                 self.is_multimer
#             )

#         self.linear_b = Linear(self.c_z, self.no_heads)

#         self.head_weights = nn.Parameter(torch.zeros((no_heads)))
#         ipa_point_weights_init_(self.head_weights)

#         concat_out_dim = self.no_heads * (
#             self.c_z + self.c_hidden + self.no_v_points * 4
#         )
#         self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

#         self.softmax = nn.Softmax(dim=-1)
#         self.softplus = nn.Softplus()

#     def forward(
#         self,
#         s: torch.Tensor,
#         z: torch.Tensor,
#         r: Union[Rigid, Rigid3Array],
#         mask: torch.Tensor,
#         inplace_safe: bool = False,
#         _offload_inference: bool = False,
#         _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         """
#         Args:
#             s:
#                 [*, N_res, C_s] single representation
#             z:
#                 [*, N_res, N_res, C_z] pair representation
#             r:
#                 [*, N_res] transformation object
#             mask:
#                 [*, N_res] mask
#         Returns:
#             [*, N_res, C_s] single representation update
#         """
#         if (_offload_inference and inplace_safe):
#             z = _z_reference_list
#         else:
#             z = [z]

#         #######################################
#         # Generate scalar and point activations
#         #######################################
#         # [*, N_res, H * C_hidden]
#         q = self.linear_q(s)

#         # [*, N_res, H, C_hidden]
#         q = q.view(q.shape[:-1] + (self.no_heads, -1))

#         # [*, N_res, H, P_qk]
#         q_pts = self.linear_q_points(s, r)

#         # The following two blocks are equivalent
#         # They're separated only to preserve compatibility with old AF weights
#         if(self.is_multimer):
#             # [*, N_res, H * C_hidden]
#             k = self.linear_k(s)
#             v = self.linear_v(s)

#             # [*, N_res, H, C_hidden]
#             k = k.view(k.shape[:-1] + (self.no_heads, -1))
#             v = v.view(v.shape[:-1] + (self.no_heads, -1))

#             # [*, N_res, H, P_qk, 3]
#             k_pts = self.linear_k_points(s, r)

#             # [*, N_res, H, P_v, 3]
#             v_pts = self.linear_v_points(s, r)
#         else:
#             # [*, N_res, H * 2 * C_hidden]
#             kv = self.linear_kv(s)

#             # [*, N_res, H, 2 * C_hidden]
#             kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

#             # [*, N_res, H, C_hidden]
#             k, v = torch.split(kv, self.c_hidden, dim=-1)

#             kv_pts = self.linear_kv_points(s, r)

#             # [*, N_res, H, P_q/P_v, 3]
#             k_pts, v_pts = torch.split(
#                 kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
#             )

#         ##########################
#         # Compute attention scores
#         ##########################
#         # [*, N_res, N_res, H]
#         b = self.linear_b(z[0])

#         if (_offload_inference):
#             assert (sys.getrefcount(z[0]) == 2)
#             z[0] = z[0].cpu()

#         # [*, H, N_res, N_res]
#         if (is_fp16_enabled()):
#             with torch.cuda.amp.autocast(enabled=False):
#                 a = torch.matmul(
#                     permute_final_dims(q.float(), (1, 0, 2)),  # [*, H, N_res, C_hidden]
#                     permute_final_dims(k.float(), (1, 2, 0)),  # [*, H, C_hidden, N_res]
#                 )
#         else:
#             a = torch.matmul(
#                 permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
#                 permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
#             )

#         a *= math.sqrt(1.0 / (3 * self.c_hidden))
#         a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

#         # [*, N_res, N_res, H, P_q, 3]
#         pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)

#         if (inplace_safe):
#             pt_att *= pt_att
#         else:
#             pt_att = pt_att ** 2

#         pt_att = sum(torch.unbind(pt_att, dim=-1))

#         head_weights = self.softplus(self.head_weights).view(
#             *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
#         )
#         head_weights = head_weights * math.sqrt(
#             1.0 / (3 * (self.no_qk_points * 9.0 / 2))
#         )

#         if (inplace_safe):
#             pt_att *= head_weights
#         else:
#             pt_att = pt_att * head_weights

#         # [*, N_res, N_res, H]
#         pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

#         # [*, N_res, N_res]
#         square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
#         square_mask = self.inf * (square_mask - 1)

#         # [*, H, N_res, N_res]
#         pt_att = permute_final_dims(pt_att, (2, 0, 1))

#         if (inplace_safe):
#             a += pt_att
#             del pt_att
#             a += square_mask.unsqueeze(-3)
#             # in-place softmax
#             attn_core_inplace_cuda.forward_(
#                 a,
#                 reduce(mul, a.shape[:-1]),
#                 a.shape[-1],
#             )
#         else:
#             a = a + pt_att
#             a = a + square_mask.unsqueeze(-3)
#             a = self.softmax(a)

#         ################
#         # Compute output
#         ################
#         # [*, N_res, H, C_hidden]
#         o = torch.matmul(
#             a, v.transpose(-2, -3).to(dtype=a.dtype)
#         ).transpose(-2, -3)

#         # [*, N_res, H * C_hidden]
#         o = flatten_final_dims(o, 2)

#         # [*, H, 3, N_res, P_v]
#         if (inplace_safe):
#             v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
#             o_pt = [
#                 torch.matmul(a, v.to(a.dtype))
#                 for v in torch.unbind(v_pts, dim=-3)
#             ]
#             o_pt = torch.stack(o_pt, dim=-3)
#         else:
#             o_pt = torch.sum(
#                 (
#                         a[..., None, :, :, None]
#                         * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
#                 ),
#                 dim=-2,
#             )

#         # [*, N_res, H, P_v, 3]
#         o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
#         o_pt = r[..., None, None].invert_apply(o_pt)

#         # [*, N_res, H * P_v]
#         o_pt_norm = flatten_final_dims(
#             torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
#         )

#         # [*, N_res, H * P_v, 3]
#         o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
#         o_pt = torch.unbind(o_pt, dim=-1)

#         if (_offload_inference):
#             z[0] = z[0].to(o_pt.device)

#         # [*, N_res, H, C_z]
#         o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

#         # [*, N_res, H * C_z]
#         o_pair = flatten_final_dims(o_pair, 2)

#         # [*, N_res, C_s]
#         s = self.linear_out(
#             torch.cat(
#                 (o, *o_pt, o_pt_norm, o_pair), dim=-1
#             ).to(dtype=z[0].dtype)
#         )

#         return s


class StructureModule(nn.Module):
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
        super(StructureModule, self).__init__()

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

        ipa = InvariantPointAttention 

        ipa_conf = IPAConfig(
            use_flash_attn=True,
            attn_dtype="bf16",
            c_s=self.c_s,
            c_z=self.c_z,
            c_hidden=self.c_ipa,
            no_heads=self.no_heads_ipa,
            z_factor_rank=4,  # Rank of the factorization of the edge embedding
            no_qk_points=self.no_qk_points,
            no_v_points=self.no_v_points,
        )

        self.ipa = ipa(ipa_conf)

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

    def _forward_monomer(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
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

        print(z.shape)
        z_factor_1, z_factor_2 = factorize(z, R = self.ipa_conf.z_factor_rank)
        z_factor_1 = z_factor_1.contiguous()
        z_factor_2 = z_factor_2.contiguous()

        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s=s, 
                z=z, 
                z_factor_1=z_factor_1,
                z_factor_2=z_factor_2,
                r=rigids, 
                mask=mask, 
                _offload_inference=_offload_inference, 
                _z_reference_list=z_reference_list
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

    def _forward_multimer(
            self,
            evoformer_output_dict,
            aatype,
            mask=None,
            inplace_safe=False,
            _offload_inference=False,
    ):
        raise
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
                _z_reference_list=z_reference_list
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

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
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
            outputs = self._forward_multimer(evoformer_output_dict, aatype, mask, inplace_safe, _offload_inference)
        else:
            outputs = self._forward_monomer(evoformer_output_dict, aatype, mask, inplace_safe, _offload_inference)

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




# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from se3_diffusion (https://github.com/jasonkyuyim/se3_diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------
"""Versions of OpenFold's vector update functions patched to support masking."""

import numpy as np
import torch
from beartype.typing import Any, Callable, List, Optional, Tuple, Union
from jaxtyping import Float

NODE_MASK_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes"]
UPDATE_NODE_MASK_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 1"]
QUATERNION_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 4"]
ROTATION_TENSOR_TYPE = Float[torch.Tensor, "... 3 3"]
COORDINATES_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 3"]


def rot_matmul(a: ROTATION_TENSOR_TYPE, b: ROTATION_TENSOR_TYPE) -> ROTATION_TENSOR_TYPE:
    """Performs matrix multiplication of two rotation matrix tensors. Written out by hand to avoid
    AMP downcasting.

    Args:
        a: [*, 3, 3] left multiplicand
        b: [*, 3, 3] right multiplicand
    Returns:
        The product ab
    """
    row_1 = torch.stack(
        [
            a[..., 0, 0] * b[..., 0, 0] + a[..., 0, 1] * b[..., 1, 0] + a[..., 0, 2] * b[..., 2, 0],
            a[..., 0, 0] * b[..., 0, 1] + a[..., 0, 1] * b[..., 1, 1] + a[..., 0, 2] * b[..., 2, 1],
            a[..., 0, 0] * b[..., 0, 2] + a[..., 0, 1] * b[..., 1, 2] + a[..., 0, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_2 = torch.stack(
        [
            a[..., 1, 0] * b[..., 0, 0] + a[..., 1, 1] * b[..., 1, 0] + a[..., 1, 2] * b[..., 2, 0],
            a[..., 1, 0] * b[..., 0, 1] + a[..., 1, 1] * b[..., 1, 1] + a[..., 1, 2] * b[..., 2, 1],
            a[..., 1, 0] * b[..., 0, 2] + a[..., 1, 1] * b[..., 1, 2] + a[..., 1, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_3 = torch.stack(
        [
            a[..., 2, 0] * b[..., 0, 0] + a[..., 2, 1] * b[..., 1, 0] + a[..., 2, 2] * b[..., 2, 0],
            a[..., 2, 0] * b[..., 0, 1] + a[..., 2, 1] * b[..., 1, 1] + a[..., 2, 2] * b[..., 2, 1],
            a[..., 2, 0] * b[..., 0, 2] + a[..., 2, 1] * b[..., 1, 2] + a[..., 2, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )

    return torch.stack([row_1, row_2, row_3], dim=-2)


def rot_vec_mul(r: ROTATION_TENSOR_TYPE, t: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
    """Applies a rotation to a vector. Written out by hand to avoid transfer to avoid AMP
    downcasting.

    Args:
        r: [*, 3, 3] rotation matrices
        t: [*, 3] coordinate tensors
    Returns:
        [*, 3] rotated coordinates
    """
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


def identity_rot_mats(
    batch_dims: Union[Union[Tuple[int], Tuple[np.int64]], torch.Size],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> ROTATION_TENSOR_TYPE:
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)

    return rots


def identity_trans(
    batch_dims: Union[Union[Tuple[int], Tuple[np.int64]], torch.Size],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> COORDINATES_TENSOR_TYPE:
    trans = torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return trans


def identity_quats(
    batch_dims: Union[Union[Tuple[int], Tuple[np.int64]], torch.Size],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> QUATERNION_TENSOR_TYPE:
    quat = torch.zeros((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)

    with torch.no_grad():
        quat[..., 0] = 1

    return quat


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs: List[Tuple[str, int]]) -> np.ndarray:
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: QUATERNION_TENSOR_TYPE) -> ROTATION_TENSOR_TYPE:
    """Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = quat.new_tensor(_QTR_MAT, requires_grad=False)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(rot: ROTATION_TENSOR_TYPE) -> QUATERNION_TENSOR_TYPE:
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [
        [
            xx + yy + zz,
            zy - yz,
            xz - zx,
            yx - xy,
        ],
        [
            zy - yz,
            xx - yy - zz,
            xy + yx,
            xz + zx,
        ],
        [
            xz - zx,
            xy + yx,
            yy - xx - zz,
            yz + zy,
        ],
        [
            yx - xy,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ],
    ]

    k = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]


def quat_multiply(quat1: QUATERNION_TENSOR_TYPE, quat2: QUATERNION_TENSOR_TYPE) -> QUATERNION_TENSOR_TYPE:
    """Multiply a quaternion by another quaternion."""
    mat = quat1.new_tensor(_QUAT_MULTIPLY)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2))


def quat_multiply_by_vec(quat: QUATERNION_TENSOR_TYPE, vec: COORDINATES_TENSOR_TYPE) -> QUATERNION_TENSOR_TYPE:
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = quat.new_tensor(_QUAT_MULTIPLY_BY_VEC)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None], dim=(-3, -2))


def invert_rot_mat(rot_mat: ROTATION_TENSOR_TYPE) -> ROTATION_TENSOR_TYPE:
    return rot_mat.transpose(-1, -2)


def invert_quat(quat: QUATERNION_TENSOR_TYPE, mask: Optional[NODE_MASK_TENSOR_TYPE] = None) -> QUATERNION_TENSOR_TYPE:
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    if mask is not None:
        # avoid creating NaNs with masked nodes' "missing" values via division by zero
        inv, quat_mask = quat_prime, mask.bool()
        inv[quat_mask] = inv[quat_mask] / torch.sum(quat[quat_mask] ** 2, dim=-1, keepdim=True)
    else:
        inv = quat_prime / torch.sum(quat**2, dim=-1, keepdim=True)
    return inv


class Rotation:
    """A 3D rotation.

    Depending on how the object is initialized, the rotation is represented by either a rotation
    matrix or a quaternion, though both formats are made available by helper functions. To simplify
    gradient computation, the underlying format of the rotation cannot be changed in-place. Like
    Rigid, the class is designed to mimic the behavior of a torch Tensor, almost as if each
    Rotation object were a tensor of rotations, in one format or another.
    """

    def __init__(
        self,
        rot_mats: Optional[ROTATION_TENSOR_TYPE] = None,
        quats: Optional[QUATERNION_TENSOR_TYPE] = None,
        quats_mask: Optional[NODE_MASK_TENSOR_TYPE] = None,
        normalize_quats: bool = True,
    ):
        """
        Args:
            rot_mats:
                A [*, 3, 3] rotation matrix tensor. Mutually exclusive with
                quats
            quats:
                A [*, 4] quaternion. Mutually exclusive with rot_mats. If
                normalize_quats is not True, must be a unit quaternion
            quats_mask:
                A [*] quaternion mask. If quats is specified and normalize_quats
                is True, this will be used to subset the elements of quats
                being normalized.
            normalize_quats:
                If quats is specified, whether to normalize quats
        """
        if (rot_mats is None and quats is None) or (rot_mats is not None and quats is not None):
            raise ValueError("Exactly one input argument must be specified")

        if (rot_mats is not None and rot_mats.shape[-2:] != (3, 3)) or (quats is not None and quats.shape[-1] != 4):
            raise ValueError("Incorrectly shaped rotation matrix or quaternion")

        # Force full-precision
        if quats is not None:
            quats = quats.type(torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.type(torch.float32)

        # Parse mask
        if quats is not None and quats_mask is not None:
            quats_mask = quats_mask.type(torch.bool)

        if quats is not None and normalize_quats:
            if quats_mask is not None:
                quats[quats_mask] = quats[quats_mask] / torch.linalg.norm(quats[quats_mask], dim=-1, keepdim=True)
            else:
                quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        self._rot_mats = rot_mats
        self._quats = quats

    @staticmethod
    def identity(
        shape: Tuple[Union[int, np.int64]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ):
        """Returns an identity Rotation.

        Args:
            shape:
                The "shape" of the resulting Rotation object. See documentation
                for the shape property
            dtype:
                The torch dtype for the rotation
            device:
                The torch device for the new rotation
            requires_grad:
                Whether the underlying tensors in the new rotation object
                should require gradient computation
            fmt:
                One of "quat" or "rot_mat". Determines the underlying format
                of the new object's rotation
        Returns:
            A new identity rotation
        """
        if fmt == "rot_mat":
            rot_mats = identity_rot_mats(
                shape,
                dtype,
                device,
                requires_grad,
            )
            return Rotation(rot_mats=rot_mats, quats=None)
        elif fmt == "quat":
            quats = identity_quats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")

    # Magic methods

    def __getitem__(self, index: Any):
        """Allows torch-style indexing over the virtual shape of the rotation object. See
        documentation for the shape property.

        Args:
            index:
                A torch index. E.g. (1, 3, 2), or (slice(None,))
        Returns:
            The indexed rotation
        """
        if type(index) != tuple:
            index = (index,)

        if self._rot_mats is not None:
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            return Rotation(rot_mats=rot_mats)
        elif self._quats is not None:
            quats = self._quats[index + (slice(None),)]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __mul__(self, right: torch.Tensor) -> "Rotation":
        """Pointwise left multiplication of the rotation with a tensor. Can be used to e.g., mask
        the Rotation.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        if self._rot_mats is not None:
            rot_mats = self._rot_mats * right[..., None, None]
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats * right[..., None]
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __rmul__(self, left: torch.Tensor) -> "Rotation":
        """Reverse pointwise multiplication of the rotation with a tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        return self.__mul__(left)

    # Properties

    @property
    def shape(self) -> torch.Size:
        """Returns the virtual shape of the rotation object. This shape is defined as the batch
        dimensions of the underlying rotation matrix or quaternion. If the Rotation was initialized
        with a [10, 3, 3] rotation matrix tensor, for example, the resulting shape would be [10].

        Returns:
            The virtual shape of the rotation object
        """
        s = None
        if self._quats is not None:
            s = self._quats.shape[:-1]
        else:
            s = self._rot_mats.shape[:-2]

        return s

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the underlying rotation.

        Returns:
            The dtype of the underlying rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats.dtype
        elif self._quats is not None:
            return self._quats.dtype
        else:
            raise ValueError("Both rotations are None")

    @property
    def device(self) -> torch.device:
        """The device of the underlying rotation.

        Returns:
            The device of the underlying rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats.device
        elif self._quats is not None:
            return self._quats.device
        else:
            raise ValueError("Both rotations are None")

    @property
    def requires_grad(self) -> bool:
        """Returns the requires_grad property of the underlying rotation.

        Returns:
            The requires_grad property of the underlying tensor
        """
        if self._rot_mats is not None:
            return self._rot_mats.requires_grad
        elif self._quats is not None:
            return self._quats.requires_grad
        else:
            raise ValueError("Both rotations are None")

    def reshape(
        self,
        new_rots_shape: Optional[torch.Size] = None,
    ) -> "Rotation":
        """Returns the corresponding reshaped rotation.

        Returns:
            The reshaped rotation
        """
        if self._quats is not None:
            new_rots = self._quats.reshape(new_rots_shape) if new_rots_shape else self._quats
            new_rot = Rotation(quats=new_rots, normalize_quats=False)
        else:
            new_rots = self._rot_mats.reshape(new_rots_shape) if new_rots_shape else self._rot_mats
            new_rot = Rotation(rot_mats=new_rots, normalize_quats=False)

        return new_rot

    def get_rot_mats(self) -> ROTATION_TENSOR_TYPE:
        """Returns the underlying rotation as a rotation matrix tensor.

        Returns:
            The rotation as a rotation matrix tensor
        """
        rot_mats = self._rot_mats
        if rot_mats is None:
            if self._quats is None:
                raise ValueError("Both rotations are None")
            else:
                rot_mats = quat_to_rot(self._quats)

        return rot_mats

    def get_quats(self) -> QUATERNION_TENSOR_TYPE:
        """Returns the underlying rotation as a quaternion tensor.

        Depending on whether the Rotation was initialized with a
        quaternion, this function may call torch.linalg.eigh.

        Returns:
            The rotation as a quaternion tensor.
        """
        quats = self._quats
        if quats is None:
            if self._rot_mats is None:
                raise ValueError("Both rotations are None")
            else:
                quats = rot_to_quat(self._rot_mats)

        return quats

    def get_cur_rot(self) -> Union[QUATERNION_TENSOR_TYPE, ROTATION_TENSOR_TYPE]:
        """Return the underlying rotation in its current form.

        Returns:
            The stored rotation
        """
        if self._rot_mats is not None:
            return self._rot_mats
        elif self._quats is not None:
            return self._quats
        else:
            raise ValueError("Both rotations are None")

    def get_rotvec(self, eps: float = 1e-6) -> torch.Tensor:
        """Return the underlying axis-angle rotation vector.

        Follow's scipy's implementation:
        https://github.com/scipy/scipy/blob/HEAD/scipy/spatial/transform/_rotation.pyx#L1385-L1402

        Returns:
            The stored rotation as a axis-angle vector.
        """
        quat = self.get_quats()
        # w > 0 to ensure 0 <= angle <= pi
        flip = (quat[..., :1] < 0).float()
        quat = (-1 * quat) * flip + (1 - flip) * quat

        angle = 2 * torch.atan2(torch.linalg.norm(quat[..., 1:], dim=-1), quat[..., 0])

        angle2 = angle * angle
        small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
        large_angle_scales = angle / torch.sin(angle / 2 + eps)

        small_angles = (angle <= 1e-3).float()
        rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
        rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
        return rot_vec

    # Rotation functions

    def compose_q_update_vec(
        self,
        q_update_vec: torch.Tensor,
        normalize_quats: bool = True,
        update_mask: Optional[UPDATE_NODE_MASK_TENSOR_TYPE] = None,
    ) -> "Rotation":
        """Returns a new quaternion Rotation after updating the current object's underlying
        rotation with a quaternion update, formatted as a [*, 3] tensor whose final three columns
        represent x, y, z such that (1, x, y, z) is the desired (not necessarily unit) quaternion
        update.

        Args:
            q_update_vec:
                A [*, 3] quaternion update tensor
            normalize_quats:
                Whether to normalize the output quaternion
            update_mask:
                An optional [*, 1] node mask indicating whether to update a node's geometry.
        Returns:
            An updated Rotation
        """
        quats = self.get_quats()
        quat_update = quat_multiply_by_vec(quats, q_update_vec)
        if update_mask is not None:
            quat_update = quat_update * update_mask
        new_quats = quats + quat_update
        return Rotation(
            rot_mats=None,
            quats=new_quats,
            quats_mask=update_mask.squeeze(-1),
            normalize_quats=normalize_quats,
        )

    def compose_r(self, r: "Rotation") -> "Rotation":
        """Compose the rotation matrices of the current Rotation object with those of another.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        r1 = self.get_rot_mats()
        r2 = r.get_rot_mats()
        new_rot_mats = rot_matmul(r1, r2)
        return Rotation(rot_mats=new_rot_mats, quats=None)

    def compose_q(self, r: "Rotation", normalize_quats: bool = True) -> "Rotation":
        """Compose the quaternions of the current Rotation object with those of another.

        Depending on whether either Rotation was initialized with
        quaternions, this function may call torch.linalg.eigh.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        q1 = self.get_quats()
        q2 = r.get_quats()
        new_quats = quat_multiply(q1, q2)
        return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)

    def apply(self, pts: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        """Apply the current Rotation as a rotation matrix to a set of 3D coordinates.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] rotated points
        """
        rot_mats = self.get_rot_mats()
        return rot_vec_mul(rot_mats, pts)

    def invert_apply(self, pts: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        """The inverse of the apply() method.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] inverse-rotated points
        """
        rot_mats = self.get_rot_mats()
        inv_rot_mats = invert_rot_mat(rot_mats)
        return rot_vec_mul(inv_rot_mats, pts)

    def invert(self, mask: Optional[NODE_MASK_TENSOR_TYPE] = None) -> "Rotation":
        """Returns the inverse of the current Rotation.

        Args:
            mask:
                An optional node mask indicating whether to invert a node's geometry.
        Returns:
            The inverse of the current Rotation
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=invert_rot_mat(self._rot_mats), quats=None)
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=invert_quat(self._quats, mask=mask),
                normalize_quats=False,
                quats_mask=mask,
            )
        else:
            raise ValueError("Both rotations are None")

    # "Tensor" stuff

    def unsqueeze(self, dim: int) -> "Rotation":
        """Analogous to torch.unsqueeze. The dimension is relative to the shape of the Rotation
        object.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed Rotation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")

        if self._rot_mats is not None:
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    @staticmethod
    def cat(rs, dim: int) -> "Rotation":
        """Concatenates rotations along one of the batch dimensions. Analogous to torch.cat().

        Note that the output of this operation is always a rotation matrix,
        regardless of the format of input rotations.

        Args:
            rs:
                A list of rotation objects
            dim:
                The dimension along which the rotations should be
                concatenated
        Returns:
            A concatenated Rotation object in rotation matrix format
        """
        rot_mats = [r.get_rot_mats() for r in rs]
        rot_mats = torch.cat(rot_mats, dim=dim if dim >= 0 else dim - 2)

        return Rotation(rot_mats=rot_mats, quats=None)

    def map_tensor_fn(self, fn: Callable) -> "Rotation":
        """Apply a Tensor -> Tensor function to underlying rotation tensors, mapping over the
        rotation dimension(s). Can be used e.g. to sum out a one-hot batch dimension.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rotation
        Returns:
            The transformed Rotation object
        """
        if self._rot_mats is not None:
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            rot_mats = torch.stack(list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1)
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = torch.stack(list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def cuda(self) -> "Rotation":
        """Analogous to the cuda() method of torch Tensors.

        Returns:
            A copy of the Rotation in CUDA memory
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
        elif self._quats is not None:
            return Rotation(rot_mats=None, quats=self._quats.cuda(), normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> "Rotation":
        """Analogous to the to() method of torch Tensors.

        Args:
            device:
                A torch device
            dtype:
                A torch dtype
        Returns:
            A copy of the Rotation using the new device and dtype
        """
        if self._rot_mats is not None:
            return Rotation(
                rot_mats=self._rot_mats.to(device=device, dtype=dtype),
                quats=None,
            )
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.to(device=device, dtype=dtype),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    def detach(self) -> "Rotation":
        """Returns a copy of the Rotation whose underlying Tensor has been detached from its torch
        graph.

        Returns:
            A copy of the Rotation whose underlying Tensor has been detached
            from its torch graph
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.detach(), quats=None)
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.detach(),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")


class Rigid:
    """A class representing a rigid transformation.

    Little more than a wrapper around two objects: a Rotation object and a [*, 3] translation
    Designed to behave approximately like a single torch tensor with the shape of the shared batch
    dimensions of its component parts.
    """

    def __init__(
        self,
        rots: Optional[Rotation],
        trans: Optional[COORDINATES_TENSOR_TYPE],
    ):
        """
        Args:
            rots: A [*, 3, 3] rotation tensor
            trans: A corresponding [*, 3] translation tensor
        """
        # (we need device, dtype, etc. from at least one input)

        batch_dims, dtype, device, requires_grad = None, None, None, None
        if trans is not None:
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif rots is not None:
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
            requires_grad = rots.requires_grad
        else:
            raise ValueError("At least one input argument must be specified")

        if rots is None:
            rots = Rotation.identity(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )
        elif trans is None:
            trans = identity_trans(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )

        if (rots.shape != trans.shape[:-1]) or (rots.device != trans.device):
            raise ValueError("Rots and trans incompatible")

        # Force full precision. Happens to the rotations automatically.
        trans = trans.type(torch.float32)

        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[Union[int, np.int64]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> "Rigid":
        """Constructs an identity transformation.

        Args:
            shape:
                The desired shape
            dtype:
                The dtype of both internal tensors
            device:
                The device of both internal tensors
            requires_grad:
                Whether grad should be enabled for the internal tensors
        Returns:
            The identity transformation
        """
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )

    def __getitem__(self, index: Any) -> "Rigid":
        """Indexes the affine transformation with PyTorch-style indices. The index is applied to
        the shared dimensions of both the rotation and the translation.

        E.g.::

            r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None)
            t = Rigid(r, torch.rand(10, 10, 3))
            indexed = t[3, 4:6]
            assert(indexed.shape == (2,))
            assert(indexed.get_rots().shape == (2,))
            assert(indexed.get_trans().shape == (2, 3))

        Args:
            index: A standard torch tensor index. E.g. 8, (10, None, 3),
            or (3, slice(0, 1, None))
        Returns:
            The indexed tensor
        """
        if type(index) != tuple:
            index = (index,)

        return Rigid(
            self._rots[index],
            self._trans[index + (slice(None),)],
        )

    def __mul__(self, right: torch.Tensor) -> "Rigid":
        """Pointwise left multiplication of the transformation with a tensor. Can be used to e.g.
        mask the Rigid.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def __rmul__(self, left: torch.Tensor) -> "Rigid":
        """Reverse pointwise multiplication of the transformation with a tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of the shared dimensions of the rotation and the translation.

        Returns:
            The shape of the transformation
        """
        s = self._trans.shape[:-1]
        return s

    @property
    def device(self) -> torch.device:
        """Returns the device on which the Rigid's tensors are located.

        Returns:
            The device on which the Rigid's tensors are located
        """
        return self._trans.device

    def reshape(
        self,
        new_rots_shape: Optional[torch.Size] = None,
        new_trans_shape: Optional[torch.Size] = None,
    ) -> "Rigid":
        """Returns the corresponding reshaped rotation and reshaped translation.

        Returns:
            The reshaped transformation
        """
        new_rots = self._rots.reshape(new_rots_shape=new_rots_shape) if new_rots_shape else self._rots
        new_trans = self._trans.reshape(new_trans_shape) if new_trans_shape else self._trans

        return Rigid(new_rots, new_trans)

    def get_rots(self) -> Rotation:
        """Getter for the rotation.

        Returns:
            The rotation object
        """
        return self._rots

    def get_trans(self) -> COORDINATES_TENSOR_TYPE:
        """Getter for the translation.

        Returns:
            The stored translation
        """
        return self._trans

    def compose_q_update_vec(
        self,
        q_update_vec: Float[torch.Tensor, "... num_nodes 6"],  # noqa: F722
        update_mask: Optional[UPDATE_NODE_MASK_TENSOR_TYPE] = None,
    ) -> "Rigid":
        """Composes the transformation with a quaternion update vector of shape [*, 6], where the
        final 6 columns represent the x, y, and z values of a quaternion of form (1, x, y, z)
        followed by a 3D translation.

        Args:
            q_update_vec:
                The quaternion update vector.
            update_mask:
                An optional [*, 1] node mask indicating whether to update a node's geometry.
        Returns:
            The composed transformation.
        """
        q_vec, t_vec = q_update_vec[..., :3], q_update_vec[..., 3:]
        new_rots = self._rots.compose_q_update_vec(q_vec, update_mask=update_mask)

        trans_update = self._rots.apply(t_vec)
        if update_mask is not None:
            trans_update = trans_update * update_mask
        new_translation = self._trans + trans_update

        return Rigid(new_rots, new_translation)

    def compose(self, r: "Rigid") -> "Rigid":
        """Composes the current rigid object with another.

        Args:
            r:
                Another Rigid object
        Returns:
            The composition of the two transformations
        """
        new_rot = self._rots.compose_r(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def compose_r(self, rot: "Rigid", order: str = "right") -> "Rigid":
        """Composes the current rigid object with another.

        Args:
            r:
                Another Rigid object
            order:
                Order in which to perform rotation multiplication.
        Returns:
            The composition of the two transformations
        """
        if order == "right":
            new_rot = self._rots.compose_r(rot)
        elif order == "left":
            new_rot = rot.compose_r(self._rots)
        else:
            raise ValueError(f"Unrecognized multiplication order: {order}")
        return Rigid(new_rot, self._trans)

    def apply(self, pts: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        """Applies the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor.
        Returns:
            The transformed points.
        """
        rotated = self._rots.apply(pts)
        return rotated + self._trans

    def invert_apply(self, pts: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        """Applies the inverse of the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor
        Returns:
            The transformed points.
        """
        pts = pts - self._trans
        return self._rots.invert_apply(pts)

    def invert(self) -> "Rigid":
        """Inverts the transformation.

        Returns:
            The inverse transformation.
        """
        rot_inv = self._rots.invert()
        trn_inv = rot_inv.apply(self._trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn: Callable) -> "Rigid":
        """Apply a Tensor -> Tensor function to underlying translation and rotation tensors,
        mapping over the translation/rotation dimensions respectively.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rigid
        Returns:
            The transformed Rigid object
        """
        new_rots = self._rots.map_tensor_fn(fn)
        new_trans = torch.stack(list(map(fn, torch.unbind(self._trans, dim=-1))), dim=-1)

        return Rigid(new_rots, new_trans)

    def to_tensor_4x4(self) -> Float[torch.Tensor, "... num_nodes 4 4"]:  # noqa: F722
        """Converts a transformation to a homogeneous transformation tensor.

        Returns:
            A [*, 4, 4] homogeneous transformation tensor
        """
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        tensor[..., :3, 3] = self._trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(t: Float[torch.Tensor, "... num_nodes 4 4"]) -> "Rigid":  # noqa: F722
        """Constructs a transformation from a homogeneous transformation tensor.

        Args:
            t: [*, 4, 4] homogeneous transformation tensor
        Returns:
            T object with shape [*]
        """
        if t.shape[-2:] != (4, 4):
            raise ValueError("Incorrectly shaped input tensor")

        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        trans = t[..., :3, 3]

        return Rigid(rots, trans)

    def to_tensor_7(self) -> Float[torch.Tensor, "... num_nodes 7"]:  # noqa: F722
        """Converts a transformation to a tensor with 7 final columns, four for the quaternion
        followed by three for the translation.

        Returns:
            A [*, 7] tensor representation of the transformation
        """
        tensor = self._trans.new_zeros((*self.shape, 7))
        tensor[..., :4] = self._rots.get_quats()
        tensor[..., 4:] = self._trans

        return tensor

    @staticmethod
    def from_tensor_7(t: Float[torch.Tensor, "... num_nodes 7"], normalize_quats: bool = False) -> "Rigid":  # noqa: F722
        if t.shape[-1] != 7:
            raise ValueError("Incorrectly shaped input tensor")

        quats, trans = t[..., :4], t[..., 4:]

        rots = Rotation(rot_mats=None, quats=quats, normalize_quats=normalize_quats)

        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: COORDINATES_TENSOR_TYPE,
        origin: COORDINATES_TENSOR_TYPE,
        p_xy_plane: COORDINATES_TENSOR_TYPE,
        eps: float = 1e-8,
    ) -> "Rigid":
        """Implements algorithm 21. Constructs transformations from sets of 3 points using the
        Gram-Schmidt algorithm.

        Args:
            p_neg_x_axis: [*, 3] coordinates
            origin: [*, 3] coordinates used as frame origins
            p_xy_plane: [*, 3] coordinates
            eps: Small epsilon value
        Returns:
            A transformation object of shape [*]
        """
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum(c * c for c in e0) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum(c * c for c in e1) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, torch.stack(origin, dim=-1))

    def unsqueeze(self, dim: int) -> "Rigid":
        """Analogous to torch.unsqueeze. The dimension is relative to the shared dimensions of the
        rotation/translation.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed transformation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    @staticmethod
    def cat(ts: List["Rigid"], dim: int) -> "Rigid":
        """Concatenates transformations along a new dimension.

        Args:
            ts:
                A list of T objects
            dim:
                The dimension along which the transformations should be
                concatenated
        Returns:
            A concatenated transformation object
        """
        rots = Rotation.cat([t._rots for t in ts], dim)
        trans = torch.cat([t._trans for t in ts], dim=dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    def apply_rot_fn(self, fn: Callable) -> "Rigid":
        """Applies a Rotation -> Rotation function to the stored rotation object.

        Args:
            fn: A function of type Rotation -> Rotation
        Returns:
            A transformation object with a transformed rotation.
        """
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn: Callable) -> "Rigid":
        """Applies a Tensor -> Tensor function to the stored translation.

        Args:
            fn:
                A function of type Tensor -> Tensor to be applied to the
                translation
        Returns:
            A transformation object with a transformed translation.
        """
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor: float) -> "Rigid":
        """Scales the translation by a constant factor.

        Args:
            trans_scale_factor:
                The constant factor
        Returns:
            A transformation object with a scaled translation.
        """
        return self.apply_trans_fn(lambda t: t * trans_scale_factor)

    def stop_rot_gradient(self) -> "Rigid":
        """Detaches the underlying rotation object.

        Returns:
            A transformation object with detached rotations
        """
        return self.apply_rot_fn(lambda r: r.detach())

    @staticmethod
    def make_transform_from_reference(
        n_xyz: COORDINATES_TENSOR_TYPE,
        ca_xyz: COORDINATES_TENSOR_TYPE,
        c_xyz: COORDINATES_TENSOR_TYPE,
        eps: float = 1e-20,
    ) -> "Rigid":
        """Returns a transformation object from reference coordinates.

        Note that this method does not take care of symmetries. If you
        provide the atom positions in the non-standard way, the N atom will
        end up not at [-0.527250, 1.359329, 0.0] but instead at
        [-0.527250, -1.359329, 0.0]. You need to take care of such cases in
        your code.

        Args:
            n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
            ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
            c_xyz: A [*, 3] tensor of carbon xyz coordinates.
        Returns:
            A transformation object. After applying the translation and
            rotation to the reference backbone, the coordinates will
            approximately equal to the input coordinates.
        """
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = (c_xyz[..., i] for i in range(3))
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm
        zeros = sin_c1.new_zeros(sin_c1.shape)
        ones = sin_c1.new_ones(sin_c1.shape)

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x**2 + c_y**2 + c_z**2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x**2 + c_y**2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c1_rots[..., 2, 0] = -1 * sin_c2
        c1_rots[..., 2, 2] = cos_c2

        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = (n_xyz[..., i] for i in range(3))
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = rot_matmul(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, translation)

    def cuda(self) -> "Rigid":
        """Moves the transformation object to GPU memory.

        Returns:
            A version of the transformation on GPU
        """
        return Rigid(self._rots.cuda(), self._trans.cuda())


def create_rigid(rots, trans):
    rots = Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)




attn_dtype_dict = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class IPAConfig:
    use_flash_attn: bool = True
    attn_dtype: str = "bf16"  # "fp16", "bf16", "fp32". For flash ipa, bf16 or fp16. For original, fp32.
    use_packed: bool = True
    c_s: int = 256
    c_z: int = 128
    c_hidden: int = 128
    no_heads: int = 8
    z_factor_rank: int = 2  # 0 for no factorization
    no_qk_points: int = 8
    no_v_points: int = 12
    seq_tfmr_num_heads: int = 4
    seq_tfmr_num_layers: int = 2
    num_blocks: int = 6


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22, with flash IPA.
    """

    def __init__(
        self,
        ipa_conf: IPAConfig,
        inf: float = 1e5,
        eps: float = 1e-8,
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
        super(InvariantPointAttention, self).__init__()
        self._ipa_conf = ipa_conf

        self.use_flash_attn = ipa_conf.use_flash_attn
        self.attn_dtype = attn_dtype_dict[ipa_conf.attn_dtype]
        self.use_packed = ipa_conf.use_packed

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)
        if self.c_z > 0:
            self.linear_b = Linear(self.c_z, self.no_heads)
            self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.c_hidden + self.no_v_points * 4 + self.c_z #// 4
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        self.headdim_eff = max(
            self._ipa_conf.c_hidden + 5 * self.no_qk_points + (self._ipa_conf.z_factor_rank * self.no_heads),
            self._ipa_conf.c_hidden + 3 * self.no_v_points + (self._ipa_conf.z_factor_rank * self.c_z ),
        )

        if self.headdim_eff > 256:
            print(self.headdim_eff)
            assert (
                self.use_flash_attn is False or self.attn_dtype == torch.float16
            ), "For headdim_eff > 256, you must use either naive attention or FFPA, which requires fp16 dtype."

    def flash_ipa_fwd(self, q, k, v, q_pts, k_pts, v_pts, z_factor_1, z_factor_2, r, mask, z_full=None):
        """
        Compute squared norm components (used for SE(3) invariance part)
        """
        q_pts_norm_sq = torch.norm(q_pts, dim=-1) ** 2
        k_pts_norm_sq = torch.norm(k_pts, dim=-1) ** 2

        """
        Compute non-zero padding (used for SE(3) invariance part)
        """
        head_weights = self.softplus(self.head_weights)
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        q_pad = torch.ones_like(q_pts_norm_sq)
        k_pad = torch.ones_like(k_pts_norm_sq) * (-0.5) * head_weights.view(1, 1, -1, 1)

        """
        Compute pair bias factors
        """
        if z_full is not None:
            print(z_full[0].shape)

            b = self.linear_b(z_full[0])
            print(b.shape)
            b1, b2 = factorize(b)
            b1 = permute_final_dims(b1 , (0,2,1))
            b2 = permute_final_dims(b2 , (0,2,1))

            print(b1.shape)
            print(b2.shape)

            print(z_factor_1.shape)
            print(z_factor_2.shape)
        elif z_factor_1 is not None and z_factor_2 is not None:
            # z_factor_1 has shape [B, N_res, rank, C_z]
            z_comb = torch.cat([z_factor_1.unsqueeze(1), z_factor_2.unsqueeze(1)], dim=1)
            b = self.linear_b(z_comb)
            b1 = b[:, 0, :, :, :].permute(0, 1, 3, 2)  # B, N_res, H, rank
            b2 = b[:, 1, :, :, :].permute(0, 1, 3, 2)  # B, N_res, H, rank

            z_comb_down = self.down_z(z_comb)
            z_factor_1 = z_comb_down[:, 0, :, :, :]  # B, N_res, rank, C_z//4
            z_factor_2 = z_comb_down[:, 1, :, :, :]  # B, N_res, rank, C_z//4


        """
        Compute q_aggregated
        """
        if z_factor_1 is not None:
            q_aggregated = torch.cat(
                [q, q_pts.view(q_pts.shape[0], q_pts.shape[1], q_pts.shape[2], -1), q_pts_norm_sq, q_pad, b1], dim=-1
            )
        else:
            q_aggregated = torch.cat(
                [q, q_pts.view(q_pts.shape[0], q_pts.shape[1], q_pts.shape[2], -1), q_pts_norm_sq, q_pad], dim=-1
            )
        """
        Compute k_aggregated
        """
        k_scaled = k * math.sqrt(1.0 / (3 * self.c_hidden))
        k_pts_scaled = k_pts.view(k_pts.shape[0], k_pts.shape[1], k_pts.shape[2], -1) * head_weights.view(1, 1, -1, 1)
        k_pts_norm_sq_scaled = k_pts_norm_sq * (-0.5) * head_weights.view(1, 1, -1, 1)
        if z_factor_2 is not None:
            k_aggregated = torch.cat([k_scaled, k_pts_scaled, k_pad, k_pts_norm_sq_scaled, b2], dim=-1)
        else:
            k_aggregated = torch.cat([k_scaled, k_pts_scaled, k_pad, k_pts_norm_sq_scaled], dim=-1)

        """
        Compute v_aggregated
        """
        if z_factor_2 is not None:
            v_aggregated = torch.cat(
                [
                    v,
                    v_pts.view(*v_pts.shape[:3], -1),
                    z_factor_2.view(*z_factor_2.shape[:2], 1, -1).expand(-1, -1, self.no_heads, -1),
                ],
                dim=-1,
            )
        else:
            v_aggregated = torch.cat([v, v_pts.view(v_pts.shape[0], v_pts.shape[1], v_pts.shape[2], -1)], dim=-1)

        if mask is None:
            mask = torch.ones((q.shape[0], q.shape[1]), device=q.device, dtype=torch.bool)

        """
        Pass through FA2 or FFPA depending on headdim size
        """

        if self.headdim_eff <= 256:  # use FA2
            # FA2 requires that QKV have same size for last dimension. So just choose the smallest possible size.
            max_dim_sz = max(q_aggregated.shape[-1], k_aggregated.shape[-1], v_aggregated.shape[-1])
            q_aggregated = F.pad(q_aggregated, (0, max_dim_sz - q_aggregated.shape[-1]), value=0.0)
            k_aggregated = F.pad(k_aggregated, (0, max_dim_sz - k_aggregated.shape[-1]), value=0.0)
            v_aggregated = F.pad(v_aggregated, (0, max_dim_sz - v_aggregated.shape[-1]), value=0.0)
            if self.use_packed:
                qkv = torch.cat([q_aggregated.unsqueeze(2), k_aggregated.unsqueeze(2), v_aggregated.unsqueeze(2)], dim=2)
                (
                    qkv,
                    indices,
                    cu_seqlens,
                    max_seqlen,
                    _,
                ) = unpad_input(qkv, mask)

                if qkv.dtype != self.attn_dtype:
                    qkv = qkv.to(self.attn_dtype)

                attn_res = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, softmax_scale=1)

            else:
                q_aggregated, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q_aggregated, mask)
                k_aggregated, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k_aggregated, mask)
                v_aggregated, _, cu_seqlens_v, max_seqlen_v, _ = unpad_input(v_aggregated, mask)

                if (
                    q_aggregated.dtype != self.attn_dtype
                    or k_aggregated.dtype != self.attn_dtype
                    or v_aggregated.dtype != self.attn_dtype
                ):
                    q_aggregated = q_aggregated.to(self.attn_dtype)
                    k_aggregated = k_aggregated.to(self.attn_dtype)
                    v_aggregated = v_aggregated.to(self.attn_dtype)

                attn_res = flash_attn_varlen_func(
                    q_aggregated,
                    k_aggregated,
                    v_aggregated,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=1,
                )
            attn_res = pad_input(
                attn_res,
                indices=indices,
                batch=q.shape[0],
                seqlen=q.shape[1],
            )
        else:
            raise ValueError(f"self.headdim_eff has to be <= 256 for FA2 to work: {self.headdim_eff}")

        if attn_res.dtype != torch.float32:
            attn_res = attn_res.float()

        if z_factor_2 is not None:
            attn_res = attn_res[
                :, :, :, : self.c_hidden + 3 * self.no_v_points + self._ipa_conf.z_factor_rank * z_factor_2.shape[-1]
            ]
        else:
            attn_res = attn_res[:, :, :, : self.c_hidden + 3 * self.no_v_points]

        o = attn_res[:, :, :, : self.c_hidden]
        o = flatten_final_dims(o, 2)

        # B,L,H,D
        o_pt = attn_res[:, :, :, self.c_hidden : self.c_hidden + 3 * self.no_v_points]
        # [*, H, 3, N_res, P_v]
        o_pt = rearrange(o_pt, "B L H (P_v r) -> B H r L P_v", P_v=self.no_v_points)

        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # calculate o_pair
        if z_factor_1 is not None and z_factor_2 is not None:
            o_pair = attn_res[:, :, :, self.c_hidden + 3 * self.no_v_points :].view(
                *attn_res.shape[:3], self._ipa_conf.z_factor_rank, -1
            )  # B, L, H, rank, C_z//4
            o_pair = torch.einsum("b n r d, b n h r d -> b n h d", z_factor_1, o_pair)
            o_pair = flatten_final_dims(o_pair, 2)
            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        else:
            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats]
        print(o.shape)
        print(torch.unbind(o_pt, dim=-1).shape)
        print(o_pt_norm_feats.shape)
        print(torch.cat(o_feats, dim=-1).shape)
        s = self.linear_out(torch.cat(o_feats, dim=-1))

        return s

    def slow_ipa_fwd(self, q, k, v, q_pts, k_pts, v_pts, z, r, mask, _offload_inference=False):
        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]

        if not z is None:
            b = self.linear_b(z[0])

            if _offload_inference:
                z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))

        if not z is None:
            a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if not z is None:
            if _offload_inference:
                z[0] = z[0].to(o_pt.device)

            # [*, N_res, H, C_z // 4]
            pair_z = self.down_z(z[0])
            o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

            # [*, N_res, H * C_z // 4]
            o_pair = flatten_final_dims(o_pair, 2)

            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        else:
            o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats]

        # [*, N_res, C_s]
        s = self.linear_out(torch.cat(o_feats, dim=-1))
        return s

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        z_factor_1: Optional[torch.Tensor],
        z_factor_2: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
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
        if not z is None:
            if _offload_inference:
                z = _z_reference_list
            else:
                z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        if self.use_flash_attn:
            s = self.flash_ipa_fwd(
                q,
                k,
                v,
                q_pts,
                k_pts,
                v_pts,
                z_factor_1,
                z_factor_2,
                r,
                mask=mask,
                z_full=z,
            )

        else:
            s = self.slow_ipa_fwd(
                q,
                k,
                v,
                q_pts,
                k_pts,
                v_pts,
                z,
                r,
                mask=mask,
                _offload_inference=_offload_inference,
            )
        return s


"""
Unpadding and padding operations for FlashAttention
"""


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        rearrange(hidden_states, "b s ... -> (b s) ...")[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.structure_module =StructureModule(
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
                is_multimer=False,
            )
    def forward(self, *args, **kwargs):
        return self.structure_module(*args, **kwargs)


###############################################################
#MODEL INIT
###############################################################
model = Model().to(device)
model = import_weights(model, "../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt")
# model = import_weights(model, "../openfold/openfold/resources/params/params_model_1_multimer_v3.npz")

d = torch.load( "../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt")

for k,v in d.items():
    if "structure_module.ipa" in k:
        print(k, v.shape)


config = model_config(
    "finetuning_no_templ", 
    train=True, 
    low_prec=False,
) 

###############################################################
#DATA INIT
###############################################################
# Embeddings
embeddings = torch.load("data/cryofold/embeddings/4ake_A_embeddings.pt", map_location="cpu")
# embeddings = torch.load("data/cryofold/spike-md/embeddings_crop_mask.pt", map_location="cpu")
# embeddings = torch.load("data/cryofold/jillsData/pred2/embeddings_fixed.pt", map_location="cpu")
embeddings = {k: torch.tensor(v) for k, v in embeddings.items() if k in embeddings_keys.keys()}
embeddings = {k:v.to(device) for k,v in embeddings.items()}
embeddings["pair"].requires_grad=True
embeddings["single"].requires_grad=True
N = embeddings["pair"].shape[-2]


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



def factorize(x, R = 4):
    # [..., L, L, D]

    x = permute_final_dims(x, (2,0,1))

    # batch SVD: returns U [D,L,L], S [D,L], Vh [D,L,L]
    # If L is big and D large, consider torch.linalg.svd_lowrank or randomized route
    U, S, Vh = torch.linalg.svd(x)  # uses batched SVD if X is batched

    # take top-R components
    U_r = U[...,  :, :R]              # [D, L, R]
    V_r = permute_final_dims(Vh[..., :R, :],(0, 2, 1))  # V = Vh.T -> [D, L, R]

    # scale by sqrt of singular values: shape [D, R]
    S_r = S[..., :R]                 # [D, R]
    sqrtS = torch.sqrt(S_r)       # [D, R]

    # incorporate sqrtS into U_r and V_r (broadcast)
    A = U_r * sqrtS.unsqueeze(-2)   # [D, L, R]
    B = V_r * sqrtS.unsqueeze(-2)   # [D, L, R]

    # return in requested shape (L, R, D)
    x1 = permute_final_dims(A,(1, 2, 0))   # [L, R, D]
    x2 = permute_final_dims(B,(1, 2, 0))   # [L, R, D]

    return x1, x2

z = embeddings["pair"]
b = model.structure_module.ipa.linear_b(z)
b1, b2 = factorize(b)

model(
            {
                "pair": embeddings["pair"][None],
                "single":embeddings["single"][None]
            },
            embeddings["aatype"][None],
            mask=embeddings["seq_mask"][None],
            inplace_safe=False,
            _offload_inference=False,

        )

###############################################################
#OPtimizer loop
###############################################################
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# n_epochs=1000
# M = N -11
# M_limit = 50
# for i in range(n_epochs):
#     optimizer.zero_grad()         

#     outputs = {}
#     outputs["sm"]  = model(
#             {
#                 "pair": embeddings["pair"][None],
#                 "single":embeddings["single"][None]
#             },
#             embeddings["aatype"][None],
#             mask=embeddings["seq_mask"][None],
#             inplace_safe=False,
#             _offload_inference=False,

#         )
#     outputs["final_atom_positions"] = atom14_to_atom37(
#         outputs["sm"]["positions"][-1], embeddings
#     )
#     outputs["final_atom_mask"] = embeddings["atom37_atom_exists"]
#     outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

#     embeddings.update(
#         compute_renamed_ground_truth(
#             embeddings,
#             outputs["sm"]["positions"][-1],
#         )
#     )

#     loss = fape_loss(
#         out =outputs,
#         batch = embeddings,
#         config = config.loss.fape)
#     loss.backward()
#     optimizer.step()

#     print("Iter=%i; M=%i; Loss=%.2f"%(i,M, loss.item()))

#     if M>= M_limit:
#         M-=1


#     if i%50 == 0:
#         output_single_pdb(all_atom_positions=outputs["final_atom_positions"].detach().cpu().numpy(),
#                         all_atom_mask=outputs["final_atom_mask"].detach().cpu().numpy(),
#                         aatype=embeddings["aatype"].detach().cpu().numpy(),
#                         file="../cryofold/SparseIPA/%s.pdb"%str(i+1).zfill(5),
#                         chain_index=embeddings["asym_id"].detach().cpu().numpy(),
#                         residue_index=embeddings["residue_index"].detach().cpu().numpy())

# dist = torch.sqrt(((final_atom_positions[None,:,1] - final_atom_positions[:,None,1])**2).sum(dim=-1))
