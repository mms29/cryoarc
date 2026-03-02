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

Start by placing your amino acid sequences in a single file using FASTA format in a fresh directory `your/sequence/dir`. Next use the following script to generate embeddings : 

```
python scripts/run_pretrained_openfold.py  your/sequence/dir     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path $CONDA_PREFIX/bin/jackhmmer   \
                    --hhblits_binary_path $CONDA_PREFIX/bin/hhblits          --hmmsearch_binary_path $CONDA_PREFIX/bin/hmmsearch    \
                          --hmmbuild_binary_path $CONDA_PREFIX/bin/hmmbuild          --kalign_binary_path $CONDA_PREFIX/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:0"  \
                                  --jax_param_path /home/vuillemr/openfold/openfold/resources/params/params_model_1_multimer_v3.npz\
                                          --output_dir your/results/dir          --embeddings_output_path your/results/dir/embeddings.pt
```

