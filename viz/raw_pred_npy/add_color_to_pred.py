import numpy as np
import torch
import json
import cv2
import torch.nn.functional as F


def get_images(path):
    frames = []
    with open(path,'r') as f:
        data = json.load(f)

    imgs_paths = data['observation/ego_image']

    for path in imgs_paths:
        frame = cv2.imread(path)
        if frame is not None:
            frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            print(f"Error reading image {path}")

    if len(frames) == 0:
        print("No images found in the sequence")
        return None
    return np.stack(frames)

def get_video(path):
    video = get_images(path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    _, _, _, H, W = video.shape
    # adjust the downsample factor
    downsample = 1
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

    if torch.cuda.is_available():
        video = video.cuda()
    return video

if __name__ == '__main__':

    np_file_name = "raw_points_example.npy"
    images_json_path = f'images.json'

    pred = np.load(np_file_name)
    video = get_video(images_json_path)

    H,W = 384, 512
    xyzt = pred
    T = xyzt.shape[0]
    intr = np.array([[W, 0.0, W // 2],
                     [0.0, W, H // 2],
                     [0.0, 0.0, 1.0]])
    xyztVis = xyzt.copy()
    xyztVis[..., 2] = 1.0

    xyztVis = np.linalg.inv(intr[None, ...]) @ xyztVis.reshape(-1, 3, 1)  # (TN) 3 1
    xyztVis = xyztVis.reshape(T, -1, 3)  # T N 3
    xyztVis *= xyzt[..., [2]]

    pred_tracks2d = torch.from_numpy(xyzt.copy()[:, :, :2].astype(dtype=np.float32))
    # S1, N1, _ = pred_tracks2d.shape
    video2d = video[0]  # T C H W
    H1, W1 = video[0].shape[-2:]
    pred_tracks2dNm = pred_tracks2d.clone().cuda()
    pred_tracks2dNm[..., 0] = 2 * (pred_tracks2dNm[..., 0] / W1 - 0.5)
    pred_tracks2dNm[..., 1] = 2 * (pred_tracks2dNm[..., 1] / H1 - 0.5)
    color_interp = torch.nn.functional.grid_sample(video2d, pred_tracks2dNm[:, :, None, :],
                                                    align_corners=True)

    color_interp = color_interp[:, :, :, 0].permute(0, 2, 1).cpu().numpy().astype(np.uint8)
    colored_pts = np.concatenate([xyztVis, color_interp], axis=-1)

    np.save(f'colored_{np_file_name}', colored_pts)

