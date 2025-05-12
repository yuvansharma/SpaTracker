# %%

# -------- import the base packages -------------
import sys
import os
from easydict import EasyDict as edict
import zarr
import torch
import torch.nn.functional as F
from base64 import b64encode
import numpy as np
from PIL import Image
import cv2
import argparse
from moviepy.editor import ImageSequenceClip
import torchvision.transforms as transforms

# -------- import cotracker -------------
from models.cotracker.utils.visualizer import Visualizer, read_video_from_path
from models.cotracker.predictor import CoTrackerPredictor

# -------- import spatialtracker -------------
from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer, read_video_from_path, get_images

# -------- import Depth Estimator -------------
from mde import MonoDEst
# set the arguments
parser = argparse.ArgumentParser()
# add the video and segmentation
parser.add_argument('--json_name', type=str, help='path to the json with image paths')
parser.add_argument('--vid_name', type=str, default='breakdance', help='path to the video')

# set the gpu
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
# set the model
parser.add_argument('--model', type=str, default='spatracker', help='model name')

# set the downsample factor
parser.add_argument('--downsample', type=float, default=1.0, help='downsample factor')
parser.add_argument('--grid_size', type=int, default=20, help='grid size')
# set the results outdir
parser.add_argument('--outdir', type=str, help='output directory')
parser.add_argument('--outdir_viz', type=str, help='output directory')
# set the fps
# draw the track length
parser.add_argument('--len_track', type=int, default=10, help='len_track')
parser.add_argument('--fps_vis', type=int, default=30, help='len_track')
# crop the video
parser.add_argument('--crop', action='store_true', help='whether to crop the video')
parser.add_argument('--crop_factor', type=float, default=1, help='whether to crop the video')
# backward tracking
parser.add_argument('--backward', action='store_true', help='whether to backward the tracking')

# if visualize the support points
parser.add_argument('--vis_support', action='store_true', help='whether to visualize the support points')
parser.add_argument('--save_viz', help='whether to save the visualization video etc.')
# query frame
parser.add_argument('--query_frame', type=int, default=0, help='query frame')
# set the visualized point size
parser.add_argument('--point_size', type=int, default=3, help='point size')
# take the RGBD as input
parser.add_argument('--rgbd', action='store_true', help='whether to take the RGBD as input')
parser.add_argument('--process_single_frame', action='store_true', help='whether to take the RGBD as input')

args = parser.parse_args()


fps_vis = args.fps_vis


seg_dir = ""

outdir = args.outdir
os.path.exists(outdir) or os.makedirs(outdir)
outdir_viz = args.outdir_viz
os.path.exists(outdir_viz) or os.makedirs(outdir_viz, exist_ok=True)

# set the paras
grid_size = args.grid_size
model_type = args.model
downsample = args.downsample
# set the gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# read the video
video = get_images(args.json_name, args.process_single_frame)

video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
transform = transforms.Compose([
    transforms.CenterCrop((int(384 * args.crop_factor),
                           int(512 * args.crop_factor))),
])
_, T, _, H, W = video.shape
if os.path.exists(seg_dir):
    segm_mask = np.array(Image.open(seg_dir))
else:
    segm_mask = np.ones((H, W), dtype=np.uint8)

if len(segm_mask.shape) == 3:
    segm_mask = (segm_mask[..., :3].mean(axis=-1) > 0).astype(np.uint8)
segm_mask = cv2.resize(segm_mask, (W, H), interpolation=cv2.INTER_NEAREST)
if args.crop:
    video = transform(video)
    segm_mask = transform(torch.from_numpy(segm_mask[None, None]))[0, 0].numpy()
_, _, _, H, W = video.shape
# adjust the downsample factor
if H > W:
    downsample = max(downsample, 640 // H)
elif H < W:
    downsample = max(downsample, 960 // W)
else:
    downsample = max(downsample, 640 // H)

video = F.interpolate(video[0], scale_factor=downsample,
                      mode='bilinear', align_corners=True)[None]
vidLen = video.shape[1]
idx = torch.range(0, vidLen - 1).long()
video = video[:, idx]
# save the first image
img0 = video[0, 0].permute(1, 2, 0).detach().cpu().numpy()


if args.model == "cotracker":
    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            './checkpoints/cotracker_pretrain/cotracker_stride_4_wind_8.pth'
        )
    )
    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
    pred_tracks, pred_visibility = model(video,
                                         grid_size=grid_size,
                                         backward_tracking=False,
                                         segm_mask=torch.from_numpy(segm_mask)[None, None])

    vis = Visualizer(save_dir=outdir_viz, grayscale=True,
                     fps=fps_vis, pad_value=0, tracks_leave_trace=args.len_track)
    video_vis = vis.visualize(video=video, tracks=pred_tracks,
                              visibility=pred_visibility, filename=args.vid_name + "_cotracker")
elif args.model == "spatracker":
    S_length = 12  # [8, 12, 16] choose one you want
    model = SpaTrackerPredictor(
        checkpoint=os.path.join(
            './checkpoints/spaT_final.pth',
        ),
        interp_shape=(384, 512),
        seq_length=S_length
    )
    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()

    cfg = edict({
        "mde_name": "zoedepth_nk"
    })

    if args.rgbd:
        raise NotImplementedError
        MonoDEst_M = None

        depths = []
        for dir_i in sorted(os.listdir(DEPTH_DIR), key=lambda x: int(os.path.splitext(x)[0])):
            depth = np.load(os.path.join(DEPTH_DIR, dir_i))
            depths.append(depth)
        depths = np.stack(depths, axis=0)
        depths = torch.from_numpy(depths).float().cuda()[:, None]
    else:
        MonoDEst_O = MonoDEst(cfg)
        MonoDEst_M = MonoDEst_O.model
        MonoDEst_M.eval()
        depths = None

    pred_tracks, pred_visibility, T_Firsts = (
        model(video, video_depth=depths,
              grid_size=grid_size, backward_tracking=args.backward,
              depth_predictor=MonoDEst_M, grid_query_frame=args.query_frame,
              segm_mask=torch.from_numpy(segm_mask)[None, None], wind_length=S_length)
    )

    vis = Visualizer(save_dir=outdir_viz, grayscale=True,
                     fps=fps_vis, pad_value=0, linewidth=args.point_size,
                     tracks_leave_trace=args.len_track)
    msk_query = (T_Firsts == args.query_frame)
    # visualize the all points
    if args.vis_support:
        video_vis = vis.visualize(video=video, tracks=pred_tracks[..., :2],
                                  visibility=pred_visibility,
                                  filename=args.vid_name + "_spatracker")
    else:
        pred_tracks = pred_tracks[:, :, msk_query.squeeze()]

        tracks_vis = pred_tracks[0].detach().cpu().numpy()

        zarr.save(os.path.join(outdir, 'points.zarr'), tracks_vis)


if eval(args.save_viz):
    pred_visibility = pred_visibility[:, :, msk_query.squeeze()]
    video_vis = vis.visualize(video=video, tracks=pred_tracks[..., :2],
                                      visibility=pred_visibility,
                                      filename=args.vid_name + "_spatracker")
    wide_list = list(video.unbind(1))
    wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
    clip = ImageSequenceClip(wide_list, fps=fps_vis)
    save_path = os.path.join(outdir_viz, f'{args.vid_name}_vid.mp4')
    clip.write_videofile(save_path, codec="libx264", fps=fps_vis, logger=None)

    T = pred_tracks[0].shape[0]
    # save the 3d trajectories
    xyzt = pred_tracks[0].cpu().numpy()  # T x N x 3
    intr = np.array([[W, 0.0, W // 2],
                     [0.0, W, H // 2],
                     [0.0, 0.0, 1.0]])
    xyztVis = xyzt.copy()
    xyztVis[..., 2] = 1.0

    xyztVis = np.linalg.inv(intr[None, ...]) @ xyztVis.reshape(-1, 3, 1)  # (TN) 3 1
    xyztVis = xyztVis.reshape(T, -1, 3)  # T N 3
    xyztVis *= xyzt[..., [2]]

    pred_tracks2d = pred_tracks[0][:, :, :2]
    S1, N1, _ = pred_tracks2d.shape
    video2d = video[0]  # T C H W
    H1, W1 = video[0].shape[-2:]
    pred_tracks2dNm = pred_tracks2d.clone()
    pred_tracks2dNm[..., 0] = 2 * (pred_tracks2dNm[..., 0] / W1 - 0.5)
    pred_tracks2dNm[..., 1] = 2 * (pred_tracks2dNm[..., 1] / H1 - 0.5)
    color_interp = torch.nn.functional.grid_sample(video2d, pred_tracks2dNm[:, :, None, :],
                                                   align_corners=True)

    color_interp = color_interp[:, :, :, 0].permute(0, 2, 1).cpu().numpy().astype(np.uint8)
    colored_pts = np.concatenate([xyztVis, color_interp], axis=-1)
    np.save(f'{outdir_viz}/{args.vid_name}_3d.npy', colored_pts)
