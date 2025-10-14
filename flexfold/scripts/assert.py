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
        "--drgn", action="store_true", help="TODO"
    )   
    parser.add_argument(
        "--debug", action="store_true", help="TODO"
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
    return parser

def aligned_rmsd(structure1, structure2):
    def get_ca_atoms(structure):
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=False) and "CA" in residue:
                        atoms.append(residue["CA"])
        return atoms

    atoms1 = get_ca_atoms(structure1)
    atoms2 = get_ca_atoms(structure2)

    # Superimpose and calculate RMSD
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)  # This aligns atoms2 onto atoms1
    rmsd = sup.rms
    return rmsd

def main(args: argparse.Namespace) -> None:


    config_file = "%s/config.yaml"%args.run_dir
    weight_file = "%s/weights.%i.pkl"%(args.run_dir, args.epoch)
    z_file =  "%s/z.%i.pkl"%(args.run_dir, args.epoch)
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
    print("-> drgn ", args.drgn)
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
    cfg = config.load(config_file)
    D = cfg["lattice_args"]["D"]  # image size + 1
    zdim = cfg["model_args"]["zdim"]
    norm = [float(x) for x in cfg["dataset_args"]["norm"]]

    # mask_real = spherical_soft_mask(D-1,radius=0.45, edge=0.05, device=device)
    # mask_ft = fourier_mask(D-1, device=device)
    mask_real = None
    mask_ft = None

    if args.drgn : 
        model, lattice = cryodrgnVae.load(cfg,weight_file, device=device)
    else:
        model, lattice = flexfoldVae.load(cfg,weight_file, device=device)
        parser = PDBParser(QUIET=True)
        outputs["rmsd"] = []
        gt_pdbs = list(glob.glob(args.gt_pdbs.strip("'") ))
        gt_pdbs.sort() 
        print(gt_pdbs)
        assert len(gt_pdbs) == len(gt_vols)

    model = model.to(device)
    model.eval()
    z = torch.tensor(load_pkl(z_file)).to(device)

    with torch.no_grad():

        assert z.shape[0] == n_gt*n_expand

        for ind in range(n_gt):

            dt = time.time()
            # GT VOLUME
            vol,header =parse_mrc(gt_vols[ind])
            if not args.drgn : 
                vol= shift(vol, -np.ones(3)*0.5)
            vol =  torch.tensor(vol).to(device)
            if mask_real is not None:
                vol*= mask_real
            apix = header.apix

            #GT PDB
            if not args.drgn : 
                gt_structure = parser.get_structure("ref", gt_pdbs[ind])
            timings ["parsing"] += time.time()-dt

            for j in np.arange(n_expand)[::args.skip]:
                i = ind*n_expand +j 
                print("============================== ITER %i ======================================="%i)
                dt = time.time()
                if args.drgn : 
                    v = model.decoder.eval_volume(lattice.coords, lattice.D, lattice.extent, norm,z[i])
                    v = v.to(device)
                    # v[v<0.0] =0.0 #FIXME
                    v += v.min()#FIXME

                else:
                    v,s = model.decoder.eval_volume(lattice.coords, D-1, None, None,z[i])
                timings ["model"] += time.time()-dt

                # Volume FSC
                dt = time.time()
                if mask_real is not None:
                    v*= mask_real
                fsc_curve,freqs = fourier_shell_correlation(v,vol,mask_ft, apix=apix)
                auc = fsc_auc(fsc_curve,freqs )
                res_05, res_0143 = fsc_thresh(fsc_curve,freqs )
                # create_fsc_plot(fsc, "../cryofold/AKMD/fsc.png", apix)
                if args.debug:
                    write_mrc(outdir + "/debug_%s_pred.mrc"%str(i+1).zfill(6),v.cpu() )
                    write_mrc(outdir + "/debug_%s_gt.mrc"%str(i+1).zfill(6),vol.cpu() )
                timings ["fsc"] += time.time()-dt

                dt = time.time()
                if not args.drgn : 
                    s["final_atom_positions"] = s["final_atom_positions"] @ model.decoder.rot_init + model.decoder.trans_init
                    pdb_string = struct_to_pdb(tensor_tree_map(lambda x: x.cpu().numpy()[-1], s),"", return_string=True)
                    pred_structure = parser.get_structure("mobile",StringIO(pdb_string))
                    rmsd = aligned_rmsd(gt_structure, pred_structure)
                    print("RMSD = %.2f Ang"%rmsd)
                    if args.debug:
                        with open(outdir + "/debug_%s_pred.pdb"%str(i+1).zfill(6), "w") as f:
                            f.write(pdb_string)
                        os.system("cp %s %s"%(gt_pdbs[ind], outdir + "/debug_%s_gt.pdb"%str(i+1).zfill(6)))
                timings ["rmsd"] += time.time()-dt

                if not "pixres" in outputs : 
                    outputs["pixres"] = freqs.cpu().numpy()
                outputs["fsc"].append(fsc_curve.cpu().numpy())
                outputs["auc"].append(auc)
                outputs["res_05"].append(res_05)
                outputs["res_0143"].append(res_0143)
                if not args.drgn : 
                    outputs["rmsd"].append(rmsd)

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

        if not args.drgn : 
            fig, ax = plt.subplots(1,1)
            sns.kdeplot(outputs["rmsd"], fill=True, alpha=.5)
            rmsdmean = np.mean(outputs["rmsd"])
            rmsdmedian = np.median(outputs["rmsd"])
            ax.axvline(rmsdmean, c="red")
            ax.axvline(rmsdmedian, c="green")
            ax.set_title("RMSD = %.2f $\AA$ (%.2f $\AA$)"%(rmsdmean, rmsdmedian))
            ax.set_xlabel("RMSD($\AA$) ")
            fig.savefig(outdir+"/assert_rmsd.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)