
import sys

import torch
import torch.nn as nn

from openfold.utils.geometry.rigid_matrix_vector import Rigid3Array

from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)
from openfold.model.structure_module import StructureModule
from functools import  partial
from typing import Any, Tuple, List, Callable, Optional
import torch
import torch.utils.checkpoint


BLOCK_ARG = Any
BLOCK_ARGS = List[BLOCK_ARG]


def get_checkpoint_fn(use_reentrant=False):
    checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=use_reentrant)
    return checkpoint


@torch.jit.ignore
def checkpoint_blocks(
    blocks: List[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: Optional[int],
    use_reentrant=False
) -> BLOCK_ARGS:

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)

        return exec_sliced

    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)

    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    checkpoint = get_checkpoint_fn(use_reentrant) 

    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        args = checkpoint(chunker(s, e), *args)
        args = wrap(args)

    return args




class StructureModuleCheckpoint(StructureModule):
    # Openfold's structure module with IPA checkpointing

    def __init__(self,**kwargs):
        super(StructureModuleCheckpoint, self).__init__(**kwargs)

    def ipa_block(self,
            s,
            z,
            rigids,
            mask,
            inplace_safe,
            _offload_inference,
            z_reference_list,
            ):
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
        
        return s

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
        
        ipa_func = partial(self.ipa_block, mask=mask, inplace_safe=inplace_safe, _offload_inference=_offload_inference, z_reference_list=z_reference_list)
        for i in range(self.no_blocks):
            # [*, N, C_s]
            if torch.is_grad_enabled():
                checkpoint =get_checkpoint_fn()
                s = checkpoint(ipa_func, *(s, z, rigids))
            else:
                s = ipa_func(s, z, rigids)
            
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
        ipa_func = partial(self.ipa_block, mask=mask, inplace_safe=inplace_safe, _offload_inference=_offload_inference, z_reference_list=z_reference_list)
        for i in range(self.no_blocks):
            # [*, N, C_s]
            if torch.is_grad_enabled():
                checkpoint =get_checkpoint_fn()
                s = checkpoint(ipa_func, *(s, z, rigids))
            else:
                s = ipa_func(s, z, rigids)
            

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
