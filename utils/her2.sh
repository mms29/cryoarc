python flexfold/scripts/run_pretrained_openfold.py  ../cryofold/HER2/sequence/     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path /home/vuillemr/.conda/envs/flexfold/bin/jackhmmer   \
                    --hhblits_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hhblits          --hmmsearch_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmsearch    \
                          --hmmbuild_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmbuild          --kalign_binary_path /home/vuillemr/.conda/envs/flexfold/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:0"   \
                                  --jax_param_path /home/vuillemr/openfold/openfold/resources/params/params_model_1_multimer_v3.npz\
                                          --output_dir data/cryofold/HER2/pred          --embeddings_output_path data/cryofold/cryobench_IgD/pred/embeddings.pt  \
                                              --use_precomputed_alignments data/cryofold/HER2/pred/alignments/      --data_random_seed 42

# Metadata handling
BASE_DIR="/home/vuillemr/cryofold/HER2/data/"
python ./flexfold/scripts/rln2xmp.py $BASE_DIR/*.star $BASE_DIR/particles.xmd
cd $BASE_DIR
xmipp_image_convert -i particles.xmd -o particles.mrcs --save_metadata_stack particles_restack.xmd --keep_input_columns 
python ./flexfold/scripts/rln2xmp.py  $BASE_DIR/particles_restack.xmd $BASE_DIR/particles.star --inverse --optics_group_from $BASE_DIR/J503_csparc2star-particles.star \
 --add_missing_cols_from $BASE_DIR/J503_csparc2star-particles.star --pixel_size 1.16 --dimension 280


# Convert metadata
cryodrgn parse_ctf_star $BASE_DIR/particles.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles.star -o $BASE_DIR/particles.pkl
# Convert metadata
cryodrgn parse_ctf_star $BASE_DIR/particles_100K.star -o $BASE_DIR/ctf_100K.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_100K.star -o $BASE_DIR/particles_100K.pkl

# Backproject for verification
cryodrgn backproject_voxel $BASE_DIR/particles.mrcs --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject --lazy 


# Cryodrgn
RUN_DIR=$BASE_DIR/run_cryodrgn_sgd_100K
cryodrgn train_vae $BASE_DIR/particles_100K.star  \
    --poses $BASE_DIR/particles_100K.pkl \
    --ctf $BASE_DIR/ctf_100K.pkl \
    --lazy\
    -n 4 \
    -o $RUN_DIR \
    --batch-size 32  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \
    --domain hartley \
    --do-pose-sgd \
    --pretrain 2

RUN_DIR=$BASE_DIR/run
python ./flexfold/scripts/analyze.py  -o $RUN_DIR/analysis $RUN_DIR 1 --pc 2 







BASE_DIR="../cryofold/HER2/data"
RUN_DIR=$BASE_DIR/run

python ./flexfold/scripts/flexible_backprojection.py  \
    --reference $RUN_DIR/reference.pdb\
     -o $RUN_DIR/backproject_flex/   \
    --coordinates "$RUN_DIR/chunk_*_coordinates.dcd" --indices "$RUN_DIR/chunk_*_indices.txt" \
    --chunk \
    --coefs $RUN_DIR/coefs.pt    \
    --batch_size 4 --particles $BASE_DIR/particles_100K.star --lazy   \
    --poses  $BASE_DIR/particles_100K.pkl --ctf $BASE_DIR/ctf_100K.pkl  \
    --wiener_constant 10.0 --sigma 1.0 --gaussian_threshold 0.9 --pixel_size 1.16

python ./flexfold/scripts/flexible_backprojection.py  -o $RUN_DIR/backproject_flex/ --wiener_constant 1.0 --pixel_size 1.16 --volumes $RUN_DIR/backproject_flex


BASE_DIR="../cryofold/HER2/data"
RUN_DIR=$BASE_DIR/run2

python ./flexfold/scripts/flexible_backprojection.py  \
    --reference $RUN_DIR/reference.pdb\
     -o $RUN_DIR/backproject_flex/   \
    --coordinates "$RUN_DIR/chunk_*_coordinates.dcd" --indices "$RUN_DIR/chunk_*_indices.txt" \
    --chunk \
    --coefs $RUN_DIR/coefs.pt    \
    --batch_size 4 --particles $BASE_DIR/particles.mrcs --lazy   \
    --poses  $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl  \
    --wiener_constant 10.0 --sigma 1.0 --gaussian_threshold 0.9 --pixel_size 1.16


python ./flexfold/scripts/flexible_backprojection.py  -o $RUN_DIR/backproject_flex/ --wiener_constant 0.1 --pixel_size 1.16 --volumes $RUN_DIR/backproject_flex
