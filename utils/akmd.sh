BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/test"

BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr1"
BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1"
BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01"
BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005"
BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.001"

RUN_DIR=$BASE_DIR/run


# Convert metadata
cryodrgn parse_ctf_star $BASE_DIR/particles_*.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_*.star -o $BASE_DIR/particles.pkl

# Backproject for verification
cryodrgn backproject_voxel $BASE_DIR/Particles/particles.mrcs --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject

# Cryodrgn
RUN_DIR=$BASE_DIR/run_cryodrgn
cryodrgn train_vae $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --uninvert-data \
    -n 100 \
    -o $RUN_DIR \
    --batch-size 512  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \
    --domain hartley
python ./scripts/assert.py -i $RUN_DIR -o $RUN_DIR --gt_pdbs "$BASE_DIR/../pdbs/*pdb" --gt_vols "$BASE_DIR/../vols128/*mrc" --epoch 99 --drgn --skip 50


#*DIRECT REAL *****************************************************
# python ./scripts/compute_initial_pose.py $BASE_DIR/../initial_pose \
#  --from_aligned_pdb $BASE_DIR/../aligned_embedding.pdb \
#  --alignment_reference ../cryofold/embeddings/4ake_A_embeddings.pdb  --overwrite


RUN_DIR=$BASE_DIR/run_fourier
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --uninvert-data \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.0 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/../initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \
    --domain fourier \
    --overwrite\
    --all_atom \
    --wd 1e-5\
    --lr 1e-4\
    --warmup 100\
    --train_val_ratio 0.995\
    --domain_loss fourier


python ./scripts/analyze.py  -o $RUN_DIR/analysis $RUN_DIR 99 --pc 2 
python ./scripts/assert.py -i $RUN_DIR -o $RUN_DIR --gt_pdbs "$BASE_DIR/../pdbs/*pdb" --gt_vols "$BASE_DIR/../vols128/*mrc" --epoch 99

python ./scripts/assert.py -i $RUN_DIR -o $RUN_DIR/debug --gt_pdbs "$BASE_DIR/../pdbs/*pdb" --gt_vols "$BASE_DIR/../vols/*mrc" --epoch 99 --debug

RUN_DIR=$BASE_DIR/run_conv
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --uninvert-data \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.0 \
    --sigma 1.0 \
    --quality_ratio 5.0 \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/../initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 32 \
    --enc-layers 5 \
    --dec-dim 256 \
    --dec-layers 3 \
    --domain real \
    --encode-mode conv \
    --overwrite\
    --all_atom \
    --wd 1e-5\
    --lr 1e-4\
    --warmup 100\
    --train_val_ratio 0.995



RUN_DIR=$BASE_DIR/run_conv_pair
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.0 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/../initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 32 \
    --enc-layers 5 \
    --dec-dim 32 \
    --dec-layers 4 \
    --domain real \
    --encode-mode conv \
    --overwrite\
    --all_atom \
    --pair_stack\
    --wd 1e-4\
    --lr 5e-4\
    --warmup 100\
    --train_val_ratio 0.99
    
    --lora_structure_module\



RUN_DIR=$BASE_DIR/run_test

python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs \
 --poses $BASE_DIR/particles.pkl  \
 --uninvert-data  \
 -n 100 \
 -o $RUN_DIR  \
 --pixel_size 1.0 \
 --sigma 1.0 \
 --quality_ratio 5.0 \
 --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
 --initial_pose_path $BASE_DIR/../initial_pose.pt \
  --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt\
  --batch-size 64\
  --num-workers 0 \
  --zdim 4\
  --enc-dim 32\
  --enc-layers 5\
  --dec-dim 256\
  --dec-layers 3\
  --domain real\
  --encode-mode conv\
  --overwrite\
  --all_atom\
  --wd 1e-5\
  --lr 1e-4\
  --warmup 100\
  --train_val_ratio 0.995\
  --domain_loss fourier\
  --do-pose-sgd\
  --pretrain 1\
  --pose-lr 5e-4\
  --pose-wd 1e-3\
  --frozen_angle\
  --viol_loss_weight 0.1\
  --chi_loss_weight 0.1\


BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr1"


conda activate drgnai
RUN_DIR=$BASE_DIR/run_drgnai
drgnai setup $RUN_DIR --particles $BASE_DIR/Particles/particles.mrcs  \
    --pose $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    --capture-setup spa\
    --reconstruction-type het \
    --pose-estimation refine \
    --conf-estimation autodecoder
drgnai train $RUN_DIR

conda activate flexfold
python ./flexfold/scripts/drgnai_assert.py -i $RUN_DIR/out -o $RUN_DIR --gt_pdbs "$BASE_DIR/../pdbs/*pdb" --gt_vols "$BASE_DIR/../vols128/*mrc" --epoch 99



##############

BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.005"
RUN_DIR=$BASE_DIR/run_conv_pair_sgd_4blocks_lowlr

python ./flexfold/scripts/trajectory_from_model.py $RUN_DIR $RUN_DIR --epoch 99  --batch_size 16 --num_nodes 1 --devices 1


python ./flexfold/scripts/flexible_backprojection.py  \
    --reference $RUN_DIR/reference.pdb\
     -o $RUN_DIR/backproject_flex   \
    --coordinates "$RUN_DIR/chunk_*_coordinates.dcd" --indices "$RUN_DIR/chunk_*_indices.txt" \
    --chunk \
     --coefs $RUN_DIR/coefs.pt    \
    --batch_size 4 --particles $BASE_DIR/Particles/particles.mrcs --lazy   \
    --ctf $BASE_DIR/ctf.pkl  \
    --wiener_constant 5 --sigma 1.0 --gaussian_threshold 0.99 \
    --poses $RUN_DIR/checkpoints/epoch=99-step=*.ckpt 
    #--poses $BASE_DIR/particles.pkl \
    # --poses $RUN_DIR/particles.pkl



BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.1"
RUN_DIR=$BASE_DIR/run_4ake_conv_mlp
python ./flexfold/scripts/flexible_backprojection.py  --reference $RUN_DIR/reference.pdb -o $RUN_DIR/backproject_test/ --chunk   \
            --coordinates "$RUN_DIR/chunk_*_coordinates.dcd" --indices "$RUN_DIR/chunk_*_indices.txt" --coefs $RUN_DIR/coefs.pt    \
            --batch_size 32 --particles $BASE_DIR/Particles/particles.mrcs --poses  $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl  \
            --wiener_constant 1.0 --sigma 1.0 --gaussian_threshold 0.9 --pixel_size 1.0

RUN_DIR=$BASE_DIR/run_dynamight

conda activate dynamight

dynamight optimize-deformations  --refinement-star-file $BASE_DIR/particles_00*.star  --output-directory $RUN_DIR --initial-model $BASE_DIR/backproject/backproject.mrc
dynamight optimize-inverse-deformations $RUN_DIR --checkpoint-file $RUN_DIR/forward_deformations/checkpoints/075.pth
dynamight deformable-backprojection $RUN_DIR  --vae-directory  $RUN_DIR/forward_deformations/checkpoints/075.pth

















########


4AKE	Escherichia coli	Bacteria (Proteobacteria)
2RH5	Aquifex aeolicus	Bacteria (Aquificae)
3CM0	Thermus thermophilus	Bacteria (Deinococcus–Thermus)
1P3J	Bacillus subtilis	Bacteria (Firmicutes)
1P4S	Mycobacterium tuberculosis	Bacteria (Actinobacteria)
4K46	Photobacterium profundum	Bacteria (Proteobacteria)
1KI9	Methanococcus thermolithotrophicus	Archaea (Euryarchaeota)
1AKY	Saccharomyces cerevisiae	Eukaryota (Fungi)
5X6K	Notothenia coriiceps	Eukaryota (Vertebrate)


python flexfold/scripts/create_embeddings.py 2rh5_A data/cryofold/embeddings/
python flexfold/scripts/create_embeddings.py 3cm0_A data/cryofold/embeddings/
python flexfold/scripts/create_embeddings.py 1p3j_A data/cryofold/embeddings/
python flexfold/scripts/create_embeddings.py 1p4s_A data/cryofold/embeddings/
python flexfold/scripts/create_embeddings.py 4k46_A data/cryofold/embeddings/
python flexfold/scripts/create_embeddings.py 1ki9_A data/cryofold/embeddings/
python flexfold/scripts/create_embeddings.py 1aky_A data/cryofold/embeddings/
python flexfold/scripts/create_embeddings.py 5x6k_A data/cryofold/embeddings/

chimerax 4ake_A_embeddings.pdb 2rh5_A_embeddings.pdb 3cm0_A_embeddings.pdb 1p3j_A_embeddings.pdb 1p4s_A_embeddings.pdb  4k46_A_embeddings.pdb 1ki9_A_embeddings.pdb 1aky_A_embeddings.pdb 5x6k_A_embeddings.pdb


BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/"
for pdbid in 4ake 2rh5 3cm0 1p3j 1p4s 4k46 1ki9 1aky 5x6k; do
    python ~/flexfold/flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose_$pdbid --backproject_path $BASE_DIR/snr1/backproject/backproject.mrc\
    --embedding_pdb_path $BASE_DIR/../embeddings/"$pdbid"_A_embeddings.pdb  --overwrite
done

BASE_DIR="/home/vuillemr/flexfold/data/cryofold/AKMD/"
RUN_DIR=$BASE_DIR/snr0.1/run_4ake_conv_mlp/
python flexfold/scripts/assert.py -i $RUN_DIR -o $RUN_DIR --gt_pdbs "$BASE_DIR/pdbs/*pdb" --gt_vols "$BASE_DIR/vols128/*mrc" --epoch 99 --skip 50
