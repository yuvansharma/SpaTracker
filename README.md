# SpaTracker

This repository contains the implementation of [SpatialTracker](https://henry123-boy.github.io/SpaTracker/) for [ARM4R](https://arm4r.github.io).

---

## Setup

### 1. Create Conda Environment

```bash
conda create -n SpaTrack python==3.10 -y
conda activate SpaTrack
```

### 2. Install PyTorch

We recommend using CUDA 12.1 (later versions might also work, but have not been tested):

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Project Requirements

```bash
git clone https://github.com/yuvansharma/SpaTracker.git
cd SpaTracker
pip install -r requirements.txt
```

> **Note:** For debugging general versioning or installation issues, please refer to the original SpatialTracker repository: [https://github.com/henry123-boy/SpaTracker](https://github.com/henry123-boy/SpaTracker)

---

## Download Checkpoints

1. Download `dpt_beit_large_384.pt` from the [MiDaS releases page](https://github.com/isl-org/MiDaS/releases/) under MiDaS 3.1.
2. Download ZoeDepth models `ZoeD_M12_K.pt` and `ZoeD_M12_NK.pt` from the [ZoeDepth v1.0 release](https://github.com/isl-org/ZoeDepth/releases/tag/v1.0).

Place these files in:

```
SpaTracker/models/monoD/zoeDepth/ckpts
```

3. Download `spaT_final.pth` from [here](https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ) and place it in:

```
SpaTracker/checkpoints/
```

---

## Running SpatialTracker

### 1. Dataset Setup

To reproduce our 3D point annotations for the Epic-Kitchens dataset:

- Download the dataset from [https://epic-kitchens.github.io](https://epic-kitchens.github.io)
- Download `epic_clips.json` and `epic_tasks_final.zip` from [HuggingFace](https://huggingface.co/datasets/yuvansharma/arm4r-data)

You can download the HuggingFace data with the commands below:
```bash 
# install git-lfs
sudo apt install git-lfs
git lfs install
# clone the dataset
git clone https://huggingface.co/datasets/yuvansharma/arm4r-data
# or you can download the files manually from here: https://huggingface.co/datasets/yuvansharma/arm4r-data
```
The structure of the `epic_tasks_final` folder is shown below:

```
epic_tasks_final/common_task
│ 
└── 000000
│   ├── instruction.zarr
│   └── points.zarr
│   └── images.json
└── 000001
│   ├── instruction.zarr
│   └── points.zarr
│   └── images.json
│...
```

Note that the ```images.json``` files have image paths that include a placeholder. This placeholder must be replaced with the actual path to your downloaded Epic-Kitchens data. You can use the script below to do this. For the prefix argument, provide the path before (and not including) the "frames" folder of the downloaded Epic-Kitchens data. For instance, if you have the file structure below:
```
.
└── epic_data
│    ├── frames
│    │   ├── P01
│    │   │   └── rgb_frames
│... 
```
Then your prefix should be the absolute path to the ```epic_data``` folder.

```bash
python scripts/rename_epic_prefix.py --root_dir path_to/epic_tasks_final/common_task --prefix path_to_epic_kitchens_data
```

Note: if you are already using the [arm4r](https://github.com/Dantong88/arm4r) repository, this script is also available under the ```tools``` directory for convenience.

### 2. Extracting 3D Points (Optional)

To generate `points.zarr` files (note: this will overwrite existing ones in `epic_tasks_final.zip`):

1. Set the `CONDA_ENV` and `CONDA_PREFIX` on lines 11 and 12 in `run_spatracker_mp.py`
2. Run the script:

```bash
conda activate SpaTrack
cd SpaTracker/scripts
python run_spatracker_mp.py --epic_path path_to/epic_tasks_final/common_task --gpu_ids 0,1,2,3
```

### 3. Generating Instruction Zarr Files (Optional)

To extract `instruction.zarr` files (also included in the original data from Huggingface, will be overwritten):

```bash
python scripts/generate_instruction_zarr.py --eps_list YOUR_PREFIX/epic_clips.json --save_root YOUR_PREFIX/epic_tasks_final/common_task
```

---

## Visualization

We provide a Blender-based interface for 3D visualizations, similar to the original SpatialTracker.

### 1. Install Blender

We used **Blender 4.0.2**, but other versions may also work.

### 2. Install Blender Python Dependencies

```bash
/opt/blender/4.0/python/bin/python3.10 -m ensurepip
/opt/blender/4.0/python/bin/python3.10 -m pip install numpy matplotlib trimesh
```

### 3. Run Visualization Example

```bash
cd viz
/opt/blender/blender -P create_viz.py -- --input ./colored_npy_files/pick_cube_example.npy
```

### 4. Visualize Your Own Videos

To visualize your own videos, follow these steps:

1. Get the `.npy` file as described in the [ARM4R repository](https://arm4r.github.io).
2. Add color to the predicted 3D point tracks. An example is provided for convenience:

```bash
cd raw_pred_npy
python add_color_to_pred.py
```

3. Visualize the colored output:

```bash
/opt/blender/blender -P create_viz.py -- --input ./raw_pred_npy/colored_raw_points_example.npy --track_len 5
```

> **Note:** You may need to manually adjust the camera position, point size and track length in Blender for different videos and scenes.

---

## Citation 
Please give us a star 🌟 on Github to support us!

Please cite our work and the original SpatialTracker if you find our work inspiring or use our code in your work:
```
@article{niu2025pre,
            title={Pre-training auto-regressive robotic models with 4d representations},
            author={Niu, Dantong and Sharma, Yuvan and Xue, Haoru and Biamby, Giscard and Zhang, Junyi and Ji, Ziteng and Darrell, Trevor and Herzig, Roei},
            journal={arXiv preprint arXiv:2502.13142},
            year={2025}
      }
```
```
@inproceedings{SpatialTracker,
    title={SpatialTracker: Tracking Any 2D Pixels in 3D Space},
    author={Xiao, Yuxi and Wang, Qianqian and Zhang, Shangzhan and Xue, Nan and Peng, Sida and Shen, Yujun and Zhou, Xiaowei},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```

## License

Please refer to the original licenses of [SpaTracker](https://github.com/henry123-boy/SpaTracker) and [ARM4R](https://arm4r.github.io) for usage guidelines and conditions.