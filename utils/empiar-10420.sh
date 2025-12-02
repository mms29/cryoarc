#Embeddings
python ~/flexfold/flexfold/scripts/run_pretrained_openfold.py  data/cryofold/EMPIAR-10420/     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path /home/vuillemr/.conda/envs/flexfold/bin/jackhmmer   \
                    --hhblits_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hhblits          --hmmsearch_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmsearch    \
                          --hmmbuild_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmbuild          --kalign_binary_path /home/vuillemr/.conda/envs/flexfold/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:0"  \
                                  --jax_param_path /home/vuillemr/openfold/openfold/resources/params/params_model_1.npz\
                                          --output_dir data/cryofold/EMPIAR-10420/          --embeddings_output_path data/cryofold/EMPIAR-10420/embeddings.pt  \
                                            #   --use_precomputed_alignments data/cryofold/jillsData/pred2/alignments       --data_random_seed 43 


BASE_DIR="data/cryofold/EMPIAR-10420/"

cryodrgn parse_ctf_star $BASE_DIR/EmbB-Combined18nov21d-19may24c.star -o $BASE_DIR/ctf.pkl --Apix 1.0 -D 256
cryodrgn parse_pose_star $BASE_DIR/EmbB-Combined18nov21d-19may24c.star -o $BASE_DIR/particles.pkl --Apix 1.0 -D 256
cryodrgn backproject_voxel $BASE_DIR/EmbB-Combined18nov21d-19may24c.star --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject --lazy
