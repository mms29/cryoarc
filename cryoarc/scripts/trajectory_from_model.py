
import torch 
from cryodrgn import config
from cryodrgn.utils import load_pkl
from cryoarc.core import dcd2numpyArr, numpyArr2dcd, struct_to_pdb
from cryoarc.models import HetOnlyVAE, struct_to_crd
import numpy as np
import tqdm
from openfold.utils.tensor_utils import tensor_tree_map
import argparse
import os
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.plugins.environments import MPIEnvironment

import torch.nn.functional as F

def pad_to_max(tensor, max_size):
    pad_size = max_size - tensor.size(0)
    if pad_size > 0:
        padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)  # pad last dim only
        tensor = F.pad(tensor, padding, value=-1)
    return tensor
class InferenceModule(pl.LightningModule):
    def __init__(self, model, N, output_prefix, z_mean, gather):
        super().__init__()
        self.model = model
        self.model.eval()

        self.N=N
        self.output_prefix = output_prefix

        self.trajectory = []
        self.val_z_idx = []

        self.crd0=None
        self.z_mean=z_mean

        self.gather = gather

    def on_predict_start(self):

        if self.trainer.is_global_zero:
            with torch.no_grad():

                struct0 = self.model.decoder.structure_decoder(self.z_mean)
                struct0["final_atom_positions"] = struct0["final_atom_positions"] @ self.model.decoder.rot_init+ self.model.decoder.trans_init[..., None, :]
                crd0,coefs = struct_to_crd(struct0, ca=False, coefs=self.model.decoder.atom_coefs)
                struct0 = tensor_tree_map(lambda x: x[-1].detach().cpu().numpy(), struct0)
                struct_to_pdb(struct0, self.output_prefix+"/reference.pdb")
                torch.save(coefs.cpu(), self.output_prefix+"/coefs.pt")

            self.crd0=crd0

    def predict_step(self, batch, batch_idx):
        z_val, idx = batch                         
        with torch.no_grad():
            struct = self.model.decoder.structure_decoder(z_val)
            struct["final_atom_positions"] = struct["final_atom_positions"] @ self.model.decoder.rot_init+ self.model.decoder.trans_init[..., None, :]
            crd,coefs = struct_to_crd(struct, ca=False, coefs=self.model.decoder.atom_coefs)

        self.trajectory.append(crd.to(torch.float16))
        self.val_z_idx.append(idx)

        return crd
    
    def on_predict_end(self):
        if self.gather:
            trajectory = self.gather_outptus()
            if self.trainer.is_global_zero: 
                self.output_coordinates(trajectory, self.output_prefix+"/coordinates.dcd")

        else:
            rank = self.global_rank
            trajectory = torch.cat(self.trajectory, dim=0)
            z_idx = torch.cat(self.val_z_idx, dim=0)
            self.trajectory.clear()
            self.val_z_idx.clear()

            prefix = self.output_prefix+"/chunk_%i_"%rank
            self.output_coordinates(trajectory, prefix+"coordinates.dcd")
            np.savetxt(prefix+"indices.txt", z_idx.cpu().numpy())


    def gather_outptus(self):
        print("Start all gather ....")
        # Stack tensors from local GPU
        print("----- CAT -----")
        trajectory = torch.cat(self.trajectory, dim=0)
        z_idx = torch.cat(self.val_z_idx, dim=0)

        print(trajectory.dtype)

        batch_size = torch.tensor(trajectory.size(0), device=trajectory.device)
        max_size = self.all_gather(batch_size).max()

        print("----- PAD-----")
        trajectory = pad_to_max(trajectory, max_size)
        z_idx = pad_to_max(z_idx, max_size)

        # Gather across all GPUs
        print("----- ALL GATHER -----")
        self.trajectory.clear()
        print(trajectory.shape)
        trajectory = self.all_gather(trajectory)
        print("----- ALL GATHER out -----")
        z_idx = self.all_gather(z_idx)

        # Remove padded entries (idx == -1)
        valid_mask = z_idx != -1
        trajectory = trajectory[valid_mask]
        z_idx = z_idx[valid_mask]

        # Reorder by idx
        def get_first_idx(A):
            unique, idx, counts = torch.unique(A, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0], device=A.device), cum_sum[:-1]))
            return ind_sorted[cum_sum]
        sorted_idx = get_first_idx(z_idx)
        trajectory = trajectory[sorted_idx]
        z_idx = z_idx[sorted_idx]

        assert  torch.all(z_idx[1:]>z_idx[:-1]), "not continuous!"

        self.trajectory.clear()
        self.val_z_idx.clear() 
        print("----- DONE -----")

        return trajectory
    
    def output_coordinates(self, trajectory, filename):
        n_crd = (trajectory.sum(dim=(-1,-2)) != 0.0 ).sum()
        print("Calculated coordinates : ", n_crd)
        print("Total coordinates : ", self.N)

        numpyArr2dcd(trajectory.detach().cpu().numpy(), filename)
        # numpyArr2dcd(np.concatenate((trajectory.detach().cpu().numpy(), self.crd0.detach().cpu().to(torch.float16).numpy()),  axis=0  ), filename)

    # No training, no validation, no optimizer
    def configure_optimizers(self):
        return None


class ZDataset(Dataset):
    def __init__(self,z):
        self.z = z

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return self.z[idx], idx


def main(args):
    epoch=args.epoch
    run_dir = args.run_dir
    output_prefix =args.output_prefix
    batch_size=args.batch_size

    config_file = "%s/config.yaml"%run_dir
    weight_file = "%s/weights.%i.pkl"%(run_dir, epoch)
    z_file =  "%s/z.%i.pkl"%(run_dir, epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = config.load(config_file)


    model, lattice = HetOnlyVAE.load(cfg,weight_file, device=device)
    model = model.to(device)
    model.eval()
    z = torch.tensor(load_pkl(z_file)).to(device)

    loader = DataLoader(ZDataset(z), batch_size=batch_size, num_workers=0, shuffle=False)
    N = z.shape[0]




    model = InferenceModule(model, N, output_prefix, z_mean = z.mean(dim=-2, keepdim=True), gather=args.gather)


    cluster_environment = MPIEnvironment() if args.mpi_plugin else None

    n_devices = torch.cuda.device_count() if args.devices == "auto" else int(args.devices)

    if n_devices >1:
        strategy = DDPStrategy(find_unused_parameters=False,
                                cluster_environment=cluster_environment,
                                process_group_backend="nccl")
                                # process_group_backend="gloo")
    else:
        strategy="auto"

    trainer = pl.Trainer(
        accelerator="auto",
        strategy=strategy,
        devices=n_devices, 
        num_nodes = args.num_nodes ,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.predict(model, dataloaders=loader)




def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "run_dir",
        type=os.path.abspath,
        help="TODO",
    )
    parser.add_argument(
        "output_prefix",
        type=os.path.abspath,
        help="TODO",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="TODO",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="TODO",
    )
    parser.add_argument(
        "--devices", type=str, default="auto", help="TODO"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="TODO"
    )        
    parser.add_argument(
        "--mpi_plugin", action="store_true", help="TODO"
    )       
    parser.add_argument(
        "--gather", action="store_true", help="TODO"
    )      
    return parser



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)


