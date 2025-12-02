ROOT_DIR=data/cryofold/newmd
mkdir $ROOT_DIR
mkdir $ROOT_DIR/snr0.1
mkdir $ROOT_DIR/snr0.01
mkdir $ROOT_DIR/snr0.001
mkdir $ROOT_DIR/pdbs
mkdir $ROOT_DIR/vols
scp -r /media/WD_Data/ScipionUserData/projects/synth_data/Runs/012875_ProtRelionExportParticles/Export/* bigfoot.ciment:~/flexfold/$ROOT_DIR/snr0.1/
scp -r /media/WD_Data/ScipionUserData/projects/synth_data/Runs/013010_ProtRelionExportParticles/Export/* bigfoot.ciment:~/flexfold/$ROOT_DIR/snr0.01/
scp -r /media/WD_Data/ScipionUserData/projects/synth_data/Runs/013166_ProtRelionExportParticles/Export/* bigfoot.ciment:~/flexfold/$ROOT_DIR/snr0.001/
scp /media/WD_Data/ScipionUserData/projects/synth_data/Runs/012513_FlexProtSynthesizeImages/extra/*pdb bigfoot.ciment:~/flexfold/$ROOT_DIR/pdbs/

cd /media/WD_Data/ScipionUserData/projects/synth_data/Runs/012513_FlexProtSynthesizeImages/extra/
xmipp_metadata_selfile_create -p "*vol" -o vols.xmd
xmipp_image_convert -i vols.xmd --oroot /home/vuillemr/remote_bettik/openfold_data/data/cryofold/newmd/vols/ --oext mrc


BASE_DIR=data/cryofold/newmd/snr0.1/
cryodrgn parse_ctf_star $BASE_DIR/particles*.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles*.star -o $BASE_DIR/particles.pkl
cryodrgn backproject_voxel $BASE_DIR/Particles/particles.mrcs --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject --lazy


BASE_DIR=data/cryofold/newmd/snr0.01/
cryodrgn parse_ctf_star $BASE_DIR/particles*.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles*.star -o $BASE_DIR/particles.pkl
cryodrgn backproject_voxel $BASE_DIR/Particles/particles.mrcs --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject --lazy


BASE_DIR=data/cryofold/newmd/snr0.001/
cryodrgn parse_ctf_star $BASE_DIR/particles*.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles*.star -o $BASE_DIR/particles.pkl
cryodrgn backproject_voxel $BASE_DIR/Particles/particles.mrcs --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject --lazy


python ~/flexfold/flexfold/scripts/compute_initial_pose.py $ROOT_DIR/initial_pose \
 --from_aligned_pdb  $ROOT_DIR/aligned.pdb \
 --alignment_reference  $ROOT_DIR/2pbi_A_embeddings.pdb --overwrite

