# CryoARC - Cryo-EM Atomic-Resolution Conformations

# Installation

```
mamba env create -f cryoarc/environment.yml
mamba activate cryoarc
```

```
./install_openfold_dependencies.sh
```


# Usage

## Create sequence embeddings

Start by placing your amino acid sequences in a single file using FASTA format in a fresh directory `your/sequence/dir`. Select the model weight you want to use. For multimer, we recommend to use `model_1_multimer_v3`. You can refer to the weights description in [Openfold](https://openfold.readthedocs.io/en/latest/Inference.html) documentation. Next use the following script to generate embeddings : 

```
python scripts/run_pretrained_openfold.py  your/sequence/dir     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path $CONDA_PREFIX/bin/jackhmmer   \
                    --hhblits_binary_path $CONDA_PREFIX/bin/hhblits          --hmmsearch_binary_path $CONDA_PREFIX/bin/hmmsearch    \
                          --hmmbuild_binary_path $CONDA_PREFIX/bin/hmmbuild          --kalign_binary_path $CONDA_PREFIX/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:0"  \
                                  --jax_param_path resources/params/params_model_1_multimer_v3.npz \
                                          --output_dir your/results/dir          --embeddings_output_path your/results/dir/embeddings.pt
```

This will produce both a PDB/MMCIF file and a `embeddings.pt` file in the `your/results/dir` directory.

## Parsing particle images and metadata

You must provide the particle image, particle alignment and CTF parameters using Relion STAR format.

We use CryoDRGN subroutines for parsing STAR file.
```
cryodrgn parse_ctf_star particles.star -o ctf.pkl
cryodrgn parse_pose_star particles.star -o particles.pkl
```

To make sure the particles are well imported, we recommend to do a rigid backprojection using the following command. 
```
cryodrgn backproject_voxel particles.star --poses particles.pkl --ctf ctf.pkl -o backproject
```

