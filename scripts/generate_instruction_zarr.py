import argparse
import os
from tqdm import tqdm
import zarr
import open_clip
import json
import torch

class Text_Tokenizer():
    def __init__(self):
        self.model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def tokenize(self, text):
        text = self.tokenizer([text])

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text)

        return text_features

def save_arr_dict(data, out_zarr_path: str):
    zarr.save(out_zarr_path, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process rlbench, the input should be python dict including episodes')
    parser.add_argument('--eps_list', type=str,
                        default='YOUR_PREFIX/epic_clips.json')
    parser.add_argument('--save_root', type=str,
                        default='YOUR_PREFIX/epic_tasks_final/common_task')

    args = parser.parse_args()
    text_tokenizer = Text_Tokenizer()

    # load the processing list
    eps_list = json.load(open(args.eps_list))


    for eps in tqdm(eps_list):
        task_dir = args.save_root

        instruction_fp = os.path.join(task_dir, eps['clip_id'], "instruction.zarr")

        instruction = text_tokenizer.tokenize(eps['instruction'])[0].numpy()
        save_arr_dict(instruction, instruction_fp)