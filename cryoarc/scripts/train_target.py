
import argparse
import os
import logging
from datetime import datetime as dt

import torch


from openfold.utils.tensor_utils import tensor_tree_map

from openfold.utils.loss import fape_loss,  supervised_chi_loss, find_structural_violations, violation_loss
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from cryoarc.core import  struct_to_pdb
from pytorch_lightning.strategies import DDPStrategy

# from openfold.utils.loss import fape_loss, compute_renamed_ground_truth
from torch.utils.data import Dataset, DataLoader

from cryoarc.scripts.train import LitDataModule,LitHetOnlyVAE, save_checkpoint,  add_args
from pytorch_lightning.plugins.environments import MPIEnvironment

logger = logging.getLogger(__name__)

from openfold.utils.loss import backbone_loss

def fape_loss(
    out,
    batch,
    config,
) -> torch.Tensor:
    traj = out["sm"]["frames"]
    asym_id = batch.get("asym_id")
    if asym_id is not None:
        intra_chain_mask = (asym_id[..., None] == asym_id[..., None, :]).to(dtype=traj.dtype)
        intra_chain_bb_loss = backbone_loss(
            traj=traj,
            pair_mask=intra_chain_mask,
            **{**batch, **config.intra_chain_backbone},
        )
        interface_bb_loss = backbone_loss(
            traj=traj,
            pair_mask=1. - intra_chain_mask,
            **{**batch, **config.interface_backbone},
        )
        weighted_bb_loss = (intra_chain_bb_loss * config.intra_chain_backbone.weight
                            + interface_bb_loss * config.interface_backbone.weight)
    else:
        bb_loss = backbone_loss(
            traj=traj,
            **{**batch, **config.backbone},
        )
        weighted_bb_loss = bb_loss * config.backbone.weight

    return torch.mean(weighted_bb_loss)


class LitTarget(LitHetOnlyVAE):
    
    def training_step(self, batch):

        # Get optimizers
        if args.do_pose_sgd:
            optimizer, _ = self.optimizers()
        else:
            optimizer = self.optimizers()

        # Set gradients to zero
        optimizer.zero_grad()

        # Get scheduler
        sch = self.lr_schedulers() 

        z_mu = torch.zeros(self.model.decoder.zdim, device=self.device)[None]
        z_logvar = torch.ones(self.model.decoder.zdim, device=self.device)[None]
        z = self.model.reparameterize(z_mu, z_logvar)

        struct = self.model.decoder.structure_decoder(z)



        # Struct violations loss
        struct_violations = find_structural_violations(
            struct,
            struct["sm"]["positions"][-1],
            **self.model.decoder.loss_config.violation,
        )
        viol_loss = violation_loss(
                    struct_violations,
                    **{**struct, **self.model.decoder.loss_config.violation},
                )

        # Torsion angle loss
        chi_loss = supervised_chi_loss(
                    struct["sm"]["angles"],
                    struct["sm"]["unnormalized_angles"],
                    **{**struct, **self.model.decoder.loss_config.supervised_chi},
                )

        loss = fape_loss(
            out =struct,
            batch = struct,
            config = self.model.decoder.loss_config.fape)

        loss += chi_loss *self.args.chi_loss_weight + viol_loss *self.args.viol_loss_weight
        
        # Backward pass
        self.manual_backward(loss)

        # Optimizer step
        optimizer.step()

        # Scheduler step
        sch.step()
        
        if self.global_step %100 == 0:
            if self.trainer.is_global_zero:
                logger.info("Writing checkpoint at step %s ..."%self.global_step)
                out_weights = "{}/weights.{}.pkl".format(self.args.outdir, self.global_step)
                out_z = "{}/z.{}.pkl".format(self.args.outdir, self.global_step)
                out_pdb = "{}/fit.{}.pdb".format(self.args.outdir, self.global_step)

                
                save_checkpoint(self.model, self.optimizers(), self.current_epoch, z_mu, z_logvar, out_weights, out_z)
                struct_to_pdb(tensor_tree_map(lambda x: (x.float() if x.dtype==torch.bfloat16 else x).detach().cpu().numpy()[-1], struct), out_pdb)

class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 1000  # Only one element total

    def __getitem__(self, idx):
        return torch.empty(0)


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        # optionally split data if needed; skip if you only have train
        self.train_data = DummyDataset()

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1)
    
def main(args: argparse.Namespace) -> None:


    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.overwrite:
        os.system("rm -rvf %s/*"%args.outdir )

    pl.seed_everything(args.seed)

    # load dataset ------------------------------------------------------------------------------------------------------------------------
    logger.info(f"Loading dataset")
    dummydatamodule = DummyDataModule(args)
    dummydatamodule.setup()
    datamodule = LitDataModule(args)
    datamodule.setup()
    # load model ----------------------------------------------------------------------
    # --------------------------------------------------
    model = LitTarget(args, datamodule.imageDataset.D, datamodule.imageDataset.N)
    if args.load:
        logger.info("Loading checkpoint from {}".format(args.load))
        checkpoint = torch.load(args.load)
        # filter unwanted keys
        exclude_prefixes = [
            "lattice",
            "decoder.embeddings",
        ]
        exclude_exact = {"decoder.rot_init", "decoder.trans_init"}

        filtered_state_dict = {
            k: v for k, v in checkpoint["model_state_dict"].items()
            if not any(k.startswith(p) for p in exclude_prefixes)
            and k not in exclude_exact
        }

        # now load
        missing, unexpected = model.model.load_state_dict(filtered_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        optim = model.configure_optimizers()
        optim = optim[0][0]# if isinstance(optim, (list, tuple)) else optim
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        optim.load_state_dict(optimizer_state_dict)
        logger.info("Successfully restored states from {}".format(args.load))


    cluster_environment = MPIEnvironment() if args.mpi_plugin else None

    n_devices = torch.cuda.device_count() if args.devices == "auto" else int(args.devices)

    if n_devices >1:
        strategy = DDPStrategy(find_unused_parameters=True,
                               static_graph=True,
                                cluster_environment=cluster_environment,
                                process_group_backend="nccl")
                                # process_group_backend="gloo")
    else:
        strategy="auto"


    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        strategy=strategy,
        # strategy=DDPStrategy(),
        devices=args.devices,                 # or >1 for multi-GPU
        precision=args.precision,  # AMP support
        log_every_n_steps=10,
        callbacks=None,           # optional callbacks like ModelCheckpoint
        logger=CSVLogger(args.outdir, name="", version="")
    )

    trainer.fit(model, datamodule=dummydatamodule)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)