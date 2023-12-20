# organoids_analysis

- Video:
  - handpick - Select video: Camera is moving, camera is static
  - ffjpg: splitting video to images, take the first image
  - rcnn - segmentation | raft
  - core localization
  - analysis
  - prediction

```
cd includes
!git clone https://github.com/princeton-vl/RAFT.git
cd RAFT

# Download models using the provided script
./download_models.sh

# Change back to the previous directory
cd ..

```
