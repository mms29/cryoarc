import torch 

from src.analyze import ModelAnalyzer, VolumeGenerator
from src.reconstruct import ModelTrainer
from src.configuration import TrainingConfigurations
from src import models
from src.lattice import Lattice
from src.utils import load_pkl

import numpy as np
if not hasattr(np, "product"):
    np.product = np.prod  

class DrgnaiWrapper:
    def __init__(self, config_file, ckpt_file, device="cpu"):
        train_config_vals = ModelTrainer.load_configs(config_file)
        out_cfgs = {k: v for k, v in train_config_vals.items() if k != 'training'}
        if 'data_norm_mean' not in out_cfgs:
            out_cfgs['data_norm_mean'] = 0.
        if 'data_norm_std' not in out_cfgs:
            out_cfgs['data_norm_std'] = 1.
        train_configs = TrainingConfigurations(**train_config_vals['training'])
        checkpoint = torch.load(ckpt_file, weights_only=False)
        hypervolume_params = checkpoint['hypervolume_params']
        self.hypervolume = models.HyperVolume(**hypervolume_params)
        self.hypervolume.load_state_dict(checkpoint['hypervolume_state_dict'])
        self.hypervolume.eval()
        self.hypervolume.to(device)
        self.lattice = Lattice(checkpoint['hypervolume_params']['resolution'],
                            extent=0.5, device=device)

        self.z_dim = checkpoint['hypervolume_params']['z_dim']
        self.radius_mask = (checkpoint['output_mask_radius']
                        if 'output_mask_radius' in checkpoint else None)

        self.data_norm=(out_cfgs['data_norm_mean'], out_cfgs['data_norm_std'])
    def eval_volume(self, z):
        return models.eval_volume_method(self.hypervolume, self.lattice,
                                self.z_dim, self.data_norm, zval=z,
                                radius=self.radius_mask)




import torch 
from flexfold.models import HetOnlyVAE as flexfoldVae
from cryodrgn.models import HetOnlyVAE as cryodrgnVae
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
import argparse
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import os
import glob
import time
from scipy.ndimage import shift

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--epoch",
        type=int,
        help="Epoch",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "-i",
        "--run_dir",
        type=os.path.abspath,
        required=True,
        help="Run directory",
    )
    parser.add_argument(
        "--gt_pdbs",
        type=str,
        help="TODO",
    )
    parser.add_argument(
        "--gt_vols",
        type=str,
        help="TODO",
    ) 
    parser.add_argument(
        "--skip",
        type=int,
        default=100,
        help="TODO",
    )
    parser.add_argument(
        "--debug", action="store_true", help="TODO"
    )   
    return parser

def main(args: argparse.Namespace) -> None:


    config_file = args.run_dir
    weight_file = "%s/weights.%i.pkl"%(args.run_dir, args.epoch)
    z_file =  "%s/conf.%i.pkl"%(args.run_dir, args.epoch)
    outdir  = args.outdir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    n_gt = 100
    n_expand = 100
    gt_vols = list(glob.glob(args.gt_vols.strip("'")))
    gt_vols.sort() 

    print("==== Arguments readout : ==== ")
    print("-> run_dir ", args.run_dir)
    print("-> outdir ", args.outdir)
    print("-> gt_vols : %s ..."%str(gt_vols[:2]))


    assert len(gt_vols)== n_gt

    outputs = {
        "fsc": [],
        "auc": [],
        "res_05": [],
        "res_0143": [],
    }
    timings = {
        "model":0.0,
        "fsc":0.0,
        "rmsd":0.0,
        "parsing":0.0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DrgnaiWrapper(config_file, weight_file, device)
    z = torch.tensor(load_pkl(z_file)).to(device)

    with torch.no_grad():

        assert z.shape[0] == n_gt*n_expand

        for ind in range(n_gt):

            dt = time.time()
            # GT VOLUME
            vol,header =parse_mrc(gt_vols[ind])
            vol =  torch.tensor(vol).to(device)
            apix = header.apix

            #GT PDB
            timings ["parsing"] += time.time()-dt

            for j in np.arange(n_expand)[::args.skip]:
                i = ind*n_expand +j 
                print("============================== ITER %i ======================================="%i)
                dt = time.time()
                v = model.eval_volume(z[i])
                v = torch.tensor(v, device=device)

                timings ["model"] += time.time()-dt

                # Volume FSC
                dt = time.time()
                fsc_curve,freqs = fourier_shell_correlation(v,vol,None, apix=apix)
                auc = fsc_auc(fsc_curve,freqs )
                res_05, res_0143 = fsc_thresh(fsc_curve,freqs )
                # create_fsc_plot(fsc, "../cryofold/AKMD/fsc.png", apix)
                if args.debug:
                    write_mrc(outdir + "/debug_%s_pred.mrc"%str(i+1).zfill(6),v.cpu() )
                    write_mrc(outdir + "/debug_%s_gt.mrc"%str(i+1).zfill(6),vol.cpu() )
                timings ["fsc"] += time.time()-dt

                dt = time.time()
                timings ["rmsd"] += time.time()-dt

                if not "pixres" in outputs : 
                    outputs["pixres"] = freqs.cpu().numpy()
                outputs["fsc"].append(fsc_curve.cpu().numpy())
                outputs["auc"].append(auc)
                print("FSCAUC = %s"%str(auc))
                outputs["res_05"].append(res_05)
                outputs["res_0143"].append(res_0143)
                print("Timings :")
                for k,v in timings.items():
                    print("-> %s : %.2f s"%(k,v))
            with open(outdir+"/assert.pkl", "wb") as f:
                pickle.dump(outputs, f)

        # plot
        res1 = np.array(outputs["res_0143"]).mean()/apix
        res5 = np.array(outputs["res_05"]).mean()/apix
        fig, ax = plt.subplots(1,1)
        ax.errorbar(outputs["pixres"], y=np.mean(outputs["fsc"],axis=0), yerr=np.std(outputs["fsc"],axis=0))
        def fraction_formatter(x, pos):
            if x == 0:
                return "0"
            return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places
        ax.xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
        ax.axhline(0.143, c="red")
        ax.axhline(0.5, c="green")
        ax.axvline(1/res1, c="red")
        ax.axvline(1/res5, c="green")
        ax.set_xlabel("Resolution ($1/\AA$)")
        ax.set_ylabel("Fourier Shell Correlation")
        ax.set_title("AVG FSC Resolution %.2f $\AA$ (%.2f AUC)"%(res1, np.mean(outputs["auc"])))
        fig.savefig(outdir+"/assert_fsc.png")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)