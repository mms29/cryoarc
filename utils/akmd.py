


import matplotlib.pyplot as plt
import numpy as np
import torch
from cryodrgn.utils import load_pkl
from umap import UMAP
from sklearn.decomposition import PCA
import os 

base_dirs=[
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1",
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01",
"/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"]

noise_levels = [0,-1,-2,-2.5,-3]



methods=[
"run_4ake_conv_mlp",
"run_4ake_conv_pair",
"run_2rh5_conv_mlp",
"run_2rh5_conv_pair",
"run_3cm0_conv_mlp",
"run_3cm0_conv_pair",
"run_1p3j_conv_mlp",
"run_1p3j_conv_pair",
"run_1p4s_conv_mlp",
"run_1p4s_conv_pair",
"run_4k46_conv_mlp",
"run_4k46_conv_pair",
"run_1ki9_conv_mlp",
"run_1ki9_conv_pair",
"run_1aky_conv_mlp",
"run_1aky_conv_pair",
"run_5x6k_conv_mlp",
"run_5x6k_conv_pair"
         ]

# 4AKE	Escherichia coli	Bacteria (Proteobacteria)
# 2RH5	Aquifex aeolicus	Bacteria (Aquificae)
# 3CM0	Thermus thermophilus	Bacteria (Deinococcus–Thermus)
# 1P3J	Bacillus subtilis	Bacteria (Firmicutes)
# 1P4S	Mycobacterium tuberculosis	Bacteria (Actinobacteria)
# 4K46	Photobacterium profundum	Bacteria (Proteobacteria)
# 1KI9	Methanococcus thermolithotrophicus	Archaea (Euryarchaeota)
# 1AKY	Saccharomyces cerevisiae	Eukaryota (Fungi)
# 5X6K	Notothenia coriiceps	Eukaryota (Vertebrate)

names=[
"Escherichia coli Bacteria (model 1)",
"Escherichia coli Bacteria (model 2)",
"Aquifex aeolicus Bacteria (model 1)",
"Aquifex aeolicus Bacteria (model 2)",
"Thermus thermophilus Bacteria (model 1)",
"Thermus thermophilus Bacteria (model 2)",
"Bacillus subtilis Bacteria (model 1)",
"Bacillus subtilis Bacteria (model 2)",
"Mycobacterium tuberculosis Bacteria (model 1)",
"Mycobacterium tuberculosis Bacteria (model 2)",
"Photobatcterium profundum Bacteria (model 1)",
"Photobatcterium profundum Bacteria (model 2)",
"Methanococcus thermolithotrophicus	Archaea (model 1)",
"Methanococcus thermolithotrophicus	Archaea (model 2)",
"Saccharomyces cerevisiae Eukaryota (model 1)",
"Saccharomyces cerevisiae Eukaryota (model 2)",
"Notothenia coriiceps (model 1)",
"Notothenia coriiceps (model 2)",
       ]
prefix = "/home/vuillemr/flexfold/data/cryofold/AKMD/species"
methods_ff =[ m for m in names if not ("DRGN" in m or "Homogeneous" in m) ]

colors =   [plt.cm.Greens(v) for v in np.linspace(0.7, 0.8, 1) ] + [plt.cm.Reds(v) for v in np.linspace(0.6, 0.8,3) ] + [plt.cm.Blues(v) for v in np.linspace(0.5, 0.8, len(methods_ff)) ]



# base_dirs=[
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"]

# noise_levels = [0,-1,-2,-2.5,-3]

# methods=["avg",
#          "run_cryodrgn", "run_cryodrgn_sgd", 
#          "run_drgnai",
#          "run","run_sgd", "run_conv_sgd", "run_fourier","run_fourier_sgd","run_conv_fourier_sgd", "run_conv_pair_fourier", "run_conv_pair_fourier_sgd",
#          "run_conv_sgd_beta", "run_conv_sgd_highlr", "run_conv_sgd_highwd", "run_conv_sgd_low_struct_loss", "run_conv_sgd_table", "run_conv_pair_sgd_4blocks_lowlr"]
# names=["Homogeneous",
#        "CryoDRGN", "CryoDRGN-Pose", 
#        "DRGN-AI",
#        "MLP-MLP","MLP-MLP-Pose", "CNN-MLP-Pose", "MLP-MLP-FT","MLP-MLP-FT-Pose","CNN-MLP-FT-Pose","CNN-Pair-FT","CNN-Pair-FT-Pose",
#        "Beta", "HighLR", "HighWD",  "LowStructLoss", "Table", "PairLowLR4Blocks"
#        ]
# prefix = "/home/vuillemr/flexfold/data/cryofold/AKMD/"
# methods_ff =[ m for m in names if not ("DRGN" in m or "Homogeneous" in m) ]

# colors =   [plt.cm.Greens(v) for v in np.linspace(0.7, 0.8, 1) ] + [plt.cm.Reds(v) for v in np.linspace(0.6, 0.8,3) ] + [plt.cm.Blues(v) for v in np.linspace(0.5, 0.8, len(methods_ff)) ]

# base_dirs=[
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"]

# noise_levels = [0,-1,-2,-2.5,-3]

# methods=["avg",
#          "run_cryodrgn", "run_cryodrgn_sgd", 
#          "run_drgnai",
#          "run_table_pair","run_table_pair_sgd", "run_conv_sgd", "run_conv_pair_new_sgd", "run_conv_mlp_new_sgd"]
# names=["Homogeneous",
#        "CryoDRGN", "CryoDRGN-Pose", 
#        "DRGN-AI",
#        "run_table_pair", "run_table_pair_sgd", "run_conv_sgd", "run_conv_pair_new_sgd", "run_conv_mlp_new_sgd"
#        ]
# prefix = "/home/vuillemr/flexfold/data/cryofold/AKMD/table"
# methods_ff =[ m for m in names if not ("DRGN" in m or "Homogeneous" in m) ]

# colors =   [plt.cm.Greens(v) for v in np.linspace(0.7, 0.8, 1) ] + [plt.cm.Reds(v) for v in np.linspace(0.6, 0.8,3) ] + [plt.cm.Blues(v) for v in np.linspace(0.5, 0.8, len(methods_ff)) ]



# base_dirs=[
# # "/home/vuillemr/flexfold/data/cryofold/AKMD/snr1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005",
# "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"]

# noise_levels = [-1, -2,-2.5,-3]

# methods=["avg","run_cryodrgn_sgd","run_drgnai","run_conv_sgd"]
# names=["Homogeneous","CryoDRGN","DRGN-AI","Ours"]
# prefix = "/home/vuillemr/flexfold/data/cryofold/AKMD/essential_"


# colors = ["tab:green", "tab:orange", "tab:red", "tab:blue"]


##################### 
# COnformatinoal space
#####################

data_pca = []
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, m in enumerate(methods):
        if "drgnai" in m:
            filename = basedir + "/" + m + "/out/conf.99.pkl"
        else:
            filename = basedir + "/" + m + "/z.99.pkl"
        if not os.path.isfile(filename):
            data_pca.append(None)
            print("missing %s "%filename)
        else:
            z = load_pkl(filename)
            print(i)
            # dimred = UMAP(n_components=2, n_neighbors=50, min_dist=0.1)
            # data_umap.append(dimred.fit_transform(z)[:,:2])
            dimred = PCA(n_components=2)
            data_pca.append(dimred.fit_transform(z)[:,:2])

data=data_pca
nrows = len(base_dirs)
ncols = len(methods)
cmap = "plasma"
c = np.repeat(np.arange(100), 100)
fig, ax = plt.subplots(nrows, (ncols), figsize=((ncols)*4, nrows*3), layout="constrained")  
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, (m,n) in enumerate(zip(methods, names)):
        ind = ncols*i + j
        if data[ind] is not None:
            ax[i,j].scatter(data[ind][:,0],data[ind][:,1], c=c, cmap=cmap, s=2.0, alpha=0.5)
            ax[i,j].set_title(f"{n} - $SNR=10^{{{noise:g}}}$")
        else:
            print("missing %s "%n)

fig.savefig(prefix+"summary_pca.png", dpi=150)


# data_umap = []
# for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
#     for j, m in enumerate(methods):
#         filename = basedir + "/" + m + "/z.99.pkl"
#         if not os.path.isfile(filename):
#             data_umap.append(None)
#         else:
#             z = load_pkl(filename)
#             print(i)
#             dimred = UMAP(n_components=2, )#n_neighbors=50, min_dist=0.1)
#             data_umap.append(dimred.fit_transform(z)[:,:2])
# data=data_umap
# nrows = len(base_dirs)
# ncols = len(methods)
# cmap = "plasma"
# c = np.repeat(np.arange(100), 100)
# fig, ax = plt.subplots(nrows, (ncols-1), figsize=((ncols-1)*4, nrows*3), layout="constrained")  
# for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
#     for j, (m,n) in enumerate(zip(methods[1:], names[1:])):
#         ind = ncols*i + j +1
#         if data[ind] is not None:
#             ax[i,j].scatter(data[ind][:,0],data[ind][:,1], c=c, cmap=cmap, s=2.0, alpha=0.5)
#             ax[i,j].set_title(f"{n} - $SNR=10^{{{noise:g}}}$")
# fig.savefig(prefix+"summary_umap.png", dpi=300)

##################### 
# FSC
#####################
from matplotlib.ticker import FuncFormatter
def fraction_formatter(x, pos):
    if x == 0:
        return "0"
    return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places
# tab_colors = list(plt.cm.tab10.colors)

methods_ff =[ m for m in names if not ("CryoDRGN" in m or "Homogeneous" in m) ]

fig, ax = plt.subplots(1, len(noise_levels), figsize=(len(noise_levels)*4,5), layout="constrained")
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, (m,n) in enumerate(zip(methods, names)):
        if "cryodrgn" in m :
            loc = "analyze.100"
        elif "drgnai" in m:
            loc = 'out/analysis_99'
        else:
            loc = "analysis"
        filename = basedir + "/" + m + "/" +loc  + "/assert.pkl"
        if os.path.isfile(filename):
            assert_dict = load_pkl(filename)
            # ax[i].errorbar(x=assert_dict["pixres"], y=np.mean(assert_dict["fsc"],axis=0), yerr=np.std(assert_dict["fsc"],axis=0),
            #                label=m)
            ax[i].plot(assert_dict["pixres"], np.mean(assert_dict["fsc"],axis=0), label=n, color=colors[j])
    ax[i].set_title(f"$SNR=10^{{{noise:g}}}$")
    # ax[i].set_xscale("log")
    ax[i].xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
    ax[i].set_xlabel("Resolution ($1/\AA$)")
    ax[i].set_ylabel("Fourier Shell Correlation")
    ax[i].set_ylim(0,1.05)
    ax[i].axhline(0.143, ls="--", color="grey", alpha=0.5)
    ax[i].legend(loc="upper right")
fig.savefig(prefix+"summary_fsc.png", dpi=300)


fscauc = [[] for i in noise_levels]
fig, ax = plt.subplots(1, len(noise_levels), figsize=(int(len(methods)/4 * len(noise_levels))+8,5), layout="constrained")
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    for j, (m,n) in enumerate(zip(methods, names)):
        if "cryodrgn" in m :
            loc = "analyze.100"
        elif "drgnai" in m:
            loc = 'out/analysis_99'
        else:
            loc = "analysis"
        filename = basedir + "/" + m + "/" +loc  + "/assert.pkl"
        if os.path.isfile(filename):
            assert_dict = load_pkl(filename)
            fscauc[i].append([ np.mean(assert_dict["auc"]), np.std(assert_dict["auc"])])
            ax[i].bar(j,fscauc[i][-1][0],yerr=fscauc[i][-1][1], label=n, error_kw={"capsize":6}, color=colors[j])
    ax[i].set_title(f"$SNR=10^{{{noise:g}}}$")
    ax[i].set_xticks(np.arange(len(methods)))
    ax[i].set_xticklabels(names, rotation=-45,ha='left', rotation_mode='anchor')
    ax[i].set_ylabel("FSC AUC")
    ax[i].set_ylim(0.05,0.35)
fig.savefig(prefix+"summary_fscauc.png", dpi=300)

fscauc =np.array(fscauc)
fig, ax = plt.subplots(1,1, figsize=(10,5), layout="constrained")
for j in range(fscauc.shape[1]):
    ax.errorbar(x=noise_levels, y=fscauc[:, j, 0], yerr=fscauc[:, j, 1],capsize=6, color=colors[j], label=names[j])
ax.legend()
fig.savefig(prefix+"summary_fscauc_noise.png", dpi=300)



fig, ax = plt.subplots(1, len(noise_levels), figsize=(int((len(methods_ff)+1)/4 * len(noise_levels))+10,5), layout="constrained")
for i, (basedir, noise) in enumerate(zip(base_dirs, noise_levels)):
    skip=0
    for j, (m,n) in enumerate(zip(methods, names)):
        if "cryodrgn" in m :
            loc = "analyze.100"
        elif "drgnai" in m:
            loc = 'out/analysis_99'
        else:
            loc = "analysis"
        filename = basedir + "/" + m + "/" +loc  + "/assert.pkl"
        if os.path.isfile(filename):
            assert_dict = load_pkl(filename)
            if "rmsd" in assert_dict:
                ax[i].bar(j-skip, np.mean(assert_dict["rmsd"]),yerr=np.std(assert_dict["rmsd"]), label=n, error_kw={"capsize":6}, color=colors[j])
            else:
                skip+=1
    ax[i].set_title(f"$SNR=10^{{{noise:g}}}$")
    ax[i].set_xticks(np.arange(len(methods_ff) + 1 ))
    ax[i].set_xticklabels([names[0]] + methods_ff, rotation=-45,ha='left', rotation_mode='anchor')
    ax[i].set_ylabel("RMSD ($\AA$)")
    ax[i].set_ylim(0.0,5)
fig.savefig(prefix+"summary_rmsd.png", dpi=300)


##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
##########################################################################################################################################################""
from flexfold.models import map_sequences
from Bio.PDB import PPBuilder, PDBParser, PDBIO
from Bio import pairwise2
from Bio.PDB import PDBParser, Superimposer, is_aa
from Bio.SVDSuperimposer import SVDSuperimposer
import copy
import glob
from flexfold.core import dcd2numpyArr
import numpy as np

from Bio.PDB import Structure, Model, Chain, Residue, Atom
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import pickle
warnings.simplefilter("ignore", PDBConstructionWarning)

def read_traj(files):
    file_list = glob.glob(files)
    file_list.sort()
    indices_files = [f[:-15]+"indices.txt" for f in file_list]
    arr = [dcd2numpyArr(f) for f in file_list]
    ind = [np.loadtxt(f).astype(int) for f in indices_files]

    outarr = np.zeros_like(arr).reshape(-1, arr[0].shape[-2], 3)
    for a, i in zip(arr, ind):
        outarr[i] = a

    return outarr


def save_PDB(strcutre, filename):
    io = PDBIO()
    io.set_structure(strcutre)
    io.save(filename)


def read_fasta_sequences(filename):
    sequences = []
    seq = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append("".join(seq))
                    seq = []
            else:
                seq.append(line)

        if seq:
            sequences.append("".join(seq))

    return sequences


def get_ca_atoms(structure):
    atoms = {}
    chainres = {}
    for model in structure:
        for chain in model:
            atoms[chain.id] = []
            chainres[chain.id] = ""

            for residue in chain:
                if is_aa(residue, standard=False) and "CA" in residue:
                    atoms[chain.id].append(residue["CA"].get_coord())
    return atoms

def ca_atom_indices(structure):
    """
    atoms: list of Atom objects in PDB order
    Returns indices of CA atoms
    """
    atoms = []
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    atoms.append(atom)
    return [
        i for i, atom in enumerate(atoms)
        if atom.get_name() == "CA"
        and is_aa(atom.get_parent(), standard=False)
    ]

def pdb_chain_sequences(structure):
    ppb = PPBuilder()
    chain_seqs = {}
    for model in structure:
        for chain in model:
            seq = ""
            for pp in ppb.build_peptides(chain):
                seq += str(pp.get_sequence())  # concatenate fragments
            if len(seq) >= 1:
                chain_seqs[chain.id] = seq

    return chain_seqs


def map_msa_to_ca(msa_seq, ca_atoms):
    """
    Returns a list of length len(msa_seq):
    - CA atom if present
    - None if gap
    """
    mapping = []
    res_idx = 0

    for aa in msa_seq:
        if aa == "-":
            mapping.append(None)
        else:
            mapping.append(ca_atoms[res_idx])
            res_idx += 1

    if res_idx != len(ca_atoms):
        raise ValueError("MSA and CA atom count do not match")

    return mapping

from Bio.SeqUtils import seq3

def one_to_three(aa):
    """
    Convert 1-letter amino acid code to 3-letter code.
    Returns 'UNK' for unknowns.
    """
    if aa == "-" or aa is None:
        return None
    try:
        return seq3(aa).upper()
    except Exception:
        return "UNK"

def build_ca_structure(coords, seq, structure_id="model", chain_id="A"):
    """
    coords : list of length M
             each element is either None (gap) or array-like of shape (3,)
    seq    : gapped sequence string of length M

    Builds a CA-only Biopython Structure.
    """

    if len(coords) != len(seq):
        raise ValueError("coords and seq must have the same length")

    structure = Structure.Structure(structure_id)
    model = Model.Model(0)
    chain = Chain.Chain(chain_id)

    structure.add(model)
    model.add(chain)

    resseq = 1
    atom_serial = 1

    for aa, coord in zip(seq, coords):

        # Gap → no residue
        if aa == "-" or coord is None:
            resseq += 1
            continue

        residue_id = (" ", resseq, " ")
        residue = Residue.Residue(residue_id, one_to_three(aa), " ")

        ca_atom = Atom.Atom(
            name="CA",
            coord=coord,
            bfactor=0.0,
            occupancy=1.0,
            altloc=" ",
            fullname=" CA ",
            serial_number=atom_serial,
            element="C",
        )

        residue.add(ca_atom)
        chain.add(residue)

        atom_serial += 1
        resseq += 1

    return structure

def renumber_structure_msa(structure, msa_seq, chain_id=None, start=1):
    """
    Modify residue numbers in place according to the MSA sequence.

    structure: Biopython Structure object
    msa_seq: aligned sequence string with gaps ('-')
    chain_id: optional, only renumber this chain; None = all chains
    start: starting residue number
    """
    structure = copy.deepcopy(structure)
    for model in structure:
        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            res_idx = start
            # Iterate over residues in chain
            for res in chain.get_residues():
                # Skip heteroatoms / water if needed
                if res.id[0] != " ":
                    continue
                # Skip gaps in MSA
                while res_idx - start < len(msa_seq) and msa_seq[res_idx - start] == "-":
                    res_idx += 1
                # Stop if MSA shorter than chain
                if res_idx - start >= len(msa_seq):
                    break
                # Update residue number
                res.id = (' ', res_idx, ' ')
                res_idx += 1
    return structure


def read_coords(pdb_file):
    """
    Read coords in PDB file
    :param pdb_file: PDB file
    :return: array of n_atoms*3
    """
    coords = []
    with open(pdb_file, "r") as f:
        for line in f:
            if 'ATOM' in line or 'HETATM' in line:
                coords.append([
                    line[30:38], line[38:46], line[46:54]
                ])
    return np.array(coords).astype(float)



def structure_atoms(structure):
    """
    Returns a flat list of all atoms in the structure, in PDB file order.
    """
    atoms = []
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    atoms.append(atom.get_coord())
    return np.array(atoms)


def aligned_rmsd(coords_ref, coords_mob):
    """
    coords_ref : (N, 3) numpy array (reference)
    coords_mob : (N, 3) numpy array (mobile)

    Returns: RMSD after optimal superposition
    """

    assert coords_ref.shape == coords_mob.shape
    assert coords_ref.shape[1] == 3

    sup = SVDSuperimposer()
    sup.set(coords_ref, coords_mob)
    sup.run()
    return sup.get_rms()

def unaligned_rmsd(coord1, coord2):
    return np.sqrt(np.mean(np.square(np.linalg.norm(coord1 - coord2, axis=1))))

def rmsd_if_exists(crd1, crd2):
    assert len(crd1) == len(crd2)

    aligned_crd1 = []
    aligned_crd2 = []
    for c1, c2 in zip(crd1, crd2):
        if c1 is not None and c2 is not None :
            aligned_crd1.append(c1)
            aligned_crd2.append(c2)

    
    return unaligned_rmsd(np.array(aligned_crd1), np.array(aligned_crd2))

import re

def parse_tmscore_output(filepath):
    """
    Parse TM-score output file and extract RMSD, TM-score, GDT-TS, GDT-HA.

    Returns a dictionary with floats.
    """

    with open(filepath, "r") as f:
        text = f.read()

    results = {}

    # RMSD
    m = re.search(r"RMSD of\s+the common residues=\s*([0-9.]+)", text)
    if m:
        results["RMSD"] = float(m.group(1))

    # TM-score
    m = re.search(r"TM-score\s*=\s*([0-9.]+)", text)
    if m:
        results["TM-score"] = float(m.group(1))

    # GDT-TS
    m = re.search(r"GDT-TS-score\s*=\s*([0-9.]+)", text)
    if m:
        results["GDT-TS"] = float(m.group(1))

    # GDT-HA
    m = re.search(r"GDT-HA-score\s*=\s*([0-9.]+)", text)
    if m:
        results["GDT-HA"] = float(m.group(1))

    return results

import os
def calculate_tm_score(s1, s2):
    f1 =prefix+"/tmp1.pdb"
    f2 =prefix+"/tmp2.pdb"
    out = prefix+"/tmp.txt"
    save_PDB(s1, f1)
    save_PDB(s2, f2)
    os.system("data/cryofold/AKMD/TMscore_cpp %s %s > %s "%(f1, f2, out))

    return parse_tmscore_output(out)

files = [
    "initial_pose_4ake.pdb",
    "initial_pose_2rh5.pdb",
    "initial_pose_3cm0.pdb",
    "initial_pose_1p3j.pdb",
    "initial_pose_1p4s.pdb",
    "initial_pose_4k46.pdb",
    "initial_pose_1ki9.pdb",
    "initial_pose_1aky.pdb",
    "initial_pose_5x6k.pdb",
]
traj_files = [
    "snr0.1/run_4ake_conv_mlp/chunk_*_coordinates.dcd",
    "snr0.1/run_2rh5_conv_mlp/chunk_*_coordinates.dcd",
    "snr0.1/run_3cm0_conv_mlp/chunk_*_coordinates.dcd",
    "snr0.1/run_1p3j_conv_mlp/chunk_*_coordinates.dcd",
    "snr0.1/run_1p4s_conv_mlp/chunk_*_coordinates.dcd",
    "snr0.1/run_4k46_conv_mlp/chunk_*_coordinates.dcd",
    "snr0.1/run_1ki9_conv_mlp/chunk_*_coordinates.dcd",
    "snr0.1/run_1aky_conv_mlp/chunk_*_coordinates.dcd",
    "snr0.1/run_5x6k_conv_mlp/chunk_*_coordinates.dcd",
]
prefix = "data/cryofold/AKMD/"

gt_files = prefix + "pdbs/*pdb"
gt_files = glob.glob(gt_files)
gt_files.sort()

parser = PDBParser(QUIET=True)

gt_ca = np.array([np.array(get_ca_atoms(parser.get_structure("", f))["A"]) for f in gt_files])
gt_atoms = np.array([read_coords(f) for f in gt_files])
gt_atoms_test = np.array([np.array(structure_atoms(parser.get_structure("", f))) for f in gt_files])



# assert all(np.linalg.norm(gt_atoms - gt_atoms_test, axis=-1).mean(axis=-1) < 1e-5)

# structures = [parser.get_structure("", prefix + f) for f in files]

# seqs = [pdb_chain_sequences(s)["A"] for s in structures]
# msa = read_fasta_sequences(prefix + "align_sequences.fasta")
# ca = [np.array(get_ca_atoms(s)["A"]) for s in structures]
# ca_indices = [np.array(ca_atom_indices(s)) for s in structures]
# trajs = [read_traj(prefix  + tf) for tf in traj_files]


# atoms = [structure_atoms(s) for s in structures]
# atoms_test = [read_coords(prefix + f) for f in files]
# for a1, a2, ca_, id_ in zip(atoms, atoms_test, ca, ca_indices):
#     assert all(np.linalg.norm(a1 - a2, axis=-1) < 1e-5)
#     assert all(np.linalg.norm(a1[id_] - ca_, axis=-1) < 1e-5)

# ca_mapped = [map_msa_to_ca(m, c) for m, c in zip(msa, ca)]
# gt_ca_mapped = [map_msa_to_ca(msa[0], c) for c in gt_ca]

# structure_msa = [renumber_structure_msa(s, m) for s, m in zip(structures, msa)]

# for s, f in zip(structure_msa, files):
#     save_PDB(s, prefix + "msa_" + f)




n_seqs = 9
n_gt = 100
n_sampl = 4
n_per_gt = 100

names = ["4ake", "2rh5", "3cm0", "1p3j", "1p4s", "4k46", "1ki9", "1aky", "5x6k"]
scores = ['RMSD', 'RMSD2', 'TM-score', 'GDT-TS', 'GDT-HA']
# all_scores={k:{n:[] for n in names} for k in scores}
# all_scores_ini={k:{n:[] for n in names} for k in scores}
# for ss in range(n_seqs):
#     print(names[ss])

#     for ii in range(n_gt):
#         print(ii)
#         crd_gt = gt_ca_mapped[ii]
#         crd_ini = map_msa_to_ca(msa[ss], ca[ss])

#         structure_gt = build_ca_structure(crd_gt, msa[0])
#         structure_ini = build_ca_structure(crd_ini, msa[ss])
        
#         tm = calculate_tm_score(structure_gt, structure_ini)
#         tm["RMSD2"] = rmsd_if_exists(crd_gt, crd_ini)

#         for k,v in tm.items():
#             all_scores_ini[k][names[ss]].append(v)

#         for jj in range( n_sampl):
#             indice =ii*n_per_gt + jj

#             atom_pred = trajs[ss][ indice]
#             ca_pred = atom_pred [ca_indices[ss]]
#             crd_pred =  map_msa_to_ca(msa[ss], ca_pred)
#             structure_pred = build_ca_structure(crd_pred, msa[ss])

#             tm = calculate_tm_score(structure_gt, structure_pred)
#             tm["RMSD2"] = rmsd_if_exists(crd_gt, crd_pred)
#             for k,v in tm.items():
#                 all_scores[k][names[ss]].append(v)

#     print("INIT RMSD = ", np.mean(all_scores["RMSD"][names[ss]]))
#     print("MEAN RMSD = ", np.mean(all_scores_ini["RMSD"][names[ss]]))



# with open(prefix+"all_scores.pkl", "wb") as f:
#     pickle.dump({"all_scores":all_scores, "all_scores_ini":all_scores_ini}, f)

with open(prefix+"all_scores.pkl", "rb") as f:
    data = pickle.load(f)
    all_scores = data["all_scores"]
    all_scores_ini = data["all_scores_ini"]


names = ["4ake", "4k46", "1p3j", "1aky", "2rh5", "1p4s", "3cm0", "5x6k", "1ki9"]
seqid = [100, 72.90,47.66,46.26,45.63,44.20,42.74,35.38,18.23]
fsc = [2.68,2.84,2.87,3.16, 2.98, 2.91, 3.26, 3.17, 3.41]
scores = ['RMSD', 'TM-score']

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, len(scores), layout="constrained", figsize=(10,4))
# x = np.arange(n_seqs)
# for i, s in enumerate(scores):   
#     scores_pred = np.mean([np.array(all_scores[s][n]) for n in names ], axis=1)
#     scores_ini = np.mean([np.array(all_scores_ini[s][n]) for n in names ], axis=1)
#     scores_pred_std = np.std([np.array(all_scores[s][n]) for n in names ], axis=1)
#     scores_ini_std = np.std([np.array(all_scores_ini[s][n]) for n in names ], axis=1)
#     ax[i].errorbar(x-0.15, scores_pred, yerr=scores_pred_std, fmt='.', color='Black', elinewidth=2,capthick=2.0, ms=1, capsize = 4)
#     ax[i].bar(x-0.15, scores_pred,width=0.3, edgecolor='black', label="CryoARC", zorder=3, hatch="")
#     ax[i].errorbar(x+0.15, scores_ini, yerr=scores_ini_std, fmt='.', color='Black', elinewidth=2,capthick=2.0, ms=1, capsize = 4)
#     ax[i].bar(x+0.15, scores_ini,width=0.3, edgecolor='black', label="Baseline", zorder=3, hatch="///")
#     ax[i].legend()
#     ax[i].set_ylabel(s)
#     ax[i].set_xticks(x)
#     ax[i].set_xticklabels(names)

# fig.savefig(prefix+"/test.svg")



import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# fig, ax = plt.subplots(1, len(scores), layout="constrained", figsize=(10,4))
# x = np.arange(n_seqs)
# for i, s in enumerate(scores):   
#     scores_pred = np.mean([np.array(all_scores[s][n]) for n in names ], axis=1)
#     scores_ini = np.mean([np.array(all_scores_ini[s][n]) for n in names ], axis=1)
#     scores_pred_std = np.std([np.array(all_scores[s][n]) for n in names ], axis=1)
#     scores_ini_std = np.std([np.array(all_scores_ini[s][n]) for n in names ], axis=1)
#     ax[i].plot(seqid, scores_pred, ".", label="cryoARC")
#     ax[i].plot(seqid, scores_ini, ".", label="Baseline")
#     ax[i].plot(seqid, scores_ini, ".", label="Baseline")
#     ax[i].legend()
#     ax[i].set_ylabel(s)

# fig.savefig(prefix+"/test_seqid.svg")

fig, ax = plt.subplots(1, 3, layout="constrained", figsize=(12,3.5))

base=3.37
ax[0].scatter(seqid, fsc, label="cryoARC", s=100, edgecolor="black", c = np.arange(9), cmap="viridis")
ax[0].annotate("Baseline", (40, base) ,fontsize=14, ha='center', color="grey", xytext=(85,base+0.013))
ax[0].axhline(base,ls="--", color="grey")
ax[0].set_xlabel("Sequence Identity (%)")
ax[0].set_ylabel("FSC Resolution ($\AA$)")
ax[0].set_title("FSC Resolution")

scores_pred = np.mean([np.array(all_scores["TM-score"][n]) for n in names ], axis=1)
scores_pred_ini = np.mean([np.array(all_scores_ini["TM-score"][n]) for n in names ], axis=1)
base=scores_pred_ini[0]
ax[1].scatter(seqid, scores_pred, label="cryoARC", s=100, edgecolor="black", c = np.arange(9), cmap="viridis")
ax[1].annotate("Baseline", (40, base) ,fontsize=14, ha='center', color="grey", xytext=(85,base+0.01))
ax[1].axhline(base,ls="--", color="grey")
ax[1].set_xlabel("Sequence Identity (%)")
ax[1].set_ylabel("TM-score")
ax[1].set_title("Averaged TM-score")

scores_pred = np.mean([np.array(all_scores["RMSD"][n]) for n in names], axis=1)
scores_pred_ini = np.mean([np.array(all_scores_ini["RMSD"][n]) for n in names], axis=1)
base = scores_pred_ini[0]
ax[2].scatter(seqid, scores_pred, label="cryoARC", s=100, edgecolor="black", c = np.arange(9), cmap="viridis")
ax[2].annotate("Baseline", (40, base),fontsize=14, ha="center",color="grey", xytext=(35,base+0.18))
ax[2].axhline(base, ls="--", color="grey")
ax[2].set_xlabel("Sequence Identity (%)")
ax[2].set_ylabel("RMSD ($\AA$)")
ax[2].set_title("Averaged RMSD")

values = np.linspace(0, 1, 9)
import matplotlib.cm as cm
cmap = cm.get_cmap("viridis")
colors = cmap(values)
handles = [
    plt.Line2D(
        [0], [0],
        marker="s",
        linestyle="",
        markerfacecolor=colors[i],
        markeredgecolor="black",
        markersize=10,
        label=names[i]
    )
    for i in range(len(names))
]
ax[2].legend(
    handles=handles,
    # title="Structures",
    loc="best",
    fontsize=12,
    frameon=True
)

fig.savefig(prefix+"/all_species_stats.svg")

