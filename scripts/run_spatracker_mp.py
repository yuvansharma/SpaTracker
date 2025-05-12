import subprocess
import os
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager
import time
import sys
import traceback
from argparse import ArgumentParser

CONDA_ENV = "/home/USER/anaconda3/envs/SpaTrack"
CONDA_PREFIX = "/home/USER/anaconda3"

def run_spatracker_script(arguments):
    """
    Runs the SpaTracker script with the specified arguments in the given Conda environment.

    Args:
        arguments (dict): Contains the args list and GPU id.
    """
    args = arguments['args']
    gpu_id = arguments['gpu_id']

    conda_env = CONDA_ENV
    script_path = os.path.join(os.path.dirname(__file__), '..', 'arm4r_demo.py')
    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    args[args.index("--gpu") + 1] = str(gpu_id)
    conda_exec = os.path.join(CONDA_PREFIX, "etc/profile.d/conda.sh")
    cmd = (
            f"bash -c 'source {conda_exec} && "
            f"conda activate {conda_env} && python {script_path} " + " ".join(args) + "'"
    )
    with open(os.devnull, 'w') as devnull:
        try:
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                shell=True,
                check=True,
                stdout=devnull,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"GPU {gpu_id}: Error running script. See traceback below:")
            print("\n--- Traceback from subprocess ---")
            print(e.stderr)
            print("\n--- End of subprocess traceback ---")
            traceback.print_exc()


def worker(task_args, gpu_queue, progress_counter, counter_lock):
    """
    Worker function for multiprocessing.
    """
    gpu_id = gpu_queue.get()

    try:
        run_spatracker_script({'args': task_args, 'gpu_id': gpu_id})
    except Exception as e:
        print(e)
        raise Exception
    finally:
        gpu_queue.put(gpu_id)
        with counter_lock:
            progress_counter.value += 1


def main(args):
    viz_outdir = args.viz_outdir

    tasks = []

    START_INDEX = 0
    END_INDEX = 76014

    gpu_ids = args.gpu_ids.replace(" ", "").split(",")
    gpu_count = len(gpu_ids)

    for i in tqdm(range(START_INDEX, END_INDEX), desc="Creating Queue"):

        frames_json_path = os.path.join(args.epic_path, f'{i:06}/images.json')
        if not os.path.exists(frames_json_path):
            print(f'{frames_json_path} path doesnt exist')
            raise Exception

        tracks_outdir = os.path.dirname(frames_json_path)

        save_viz = "True" if (args.viz_freq > 0 and (i + START_INDEX) % args.viz_freq == 0) else "False"

        arguments = [
            "--model", "spatracker",
            "--downsample", "1",
            "--json_name", frames_json_path,
            "--vid_name", f"epic_kitchens_{i + START_INDEX}",
            "--outdir", tracks_outdir,
            "--outdir_viz", viz_outdir,
            "--len_track", "10",
            "--fps_vis", "15",
            "--save_viz", save_viz,
            "--grid_size", "36",
            "--query_frame", "0",
            "--gpu", "0", # placeholder, overwritten in run_spatracker_script function
        ]
        tasks.append(arguments)

    with Manager() as manager:
        gpu_queue = manager.Queue()
        for id in gpu_ids:
            gpu_queue.put(id)

        progress_counter = manager.Value('i', 0)
        counter_lock = manager.Lock()

        total_tasks = len(tasks)
        with tqdm(total=total_tasks, desc="Processing tasks") as pbar:
            with Pool(processes=gpu_count) as pool:

                results = [
                    pool.apply_async(worker, args=(task, gpu_queue, progress_counter, counter_lock))
                    for task in tasks
                ]

                while True:
                    with counter_lock:
                        pbar.n = progress_counter.value
                        pbar.last_print_n = progress_counter.value
                        pbar.refresh()
                    if progress_counter.value >= total_tasks:
                        break
                    time.sleep(0.1)

                pool.close()
                pool.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--epic_path',
        type=str,
        default="YOUR_PREFIX/epic_tasks_final/common_task",
        help='Path to the Epic Kitchen root folder with episode-wise images.json files',
        )

    parser.add_argument(
        '--viz_outdir',
        type=str,
        default="epic_tasks_final_viz",
        help='Directory to store video visualization',
        )

    parser.add_argument(
        '--viz_freq',
        type=int,
        default=1,
        help='Frequency for saving visualizations (set to -1 for no visualization)',
        )

    parser.add_argument(
        '--gpu_ids',
        type=str,
        default="0,1,2",
        help='list of gpu ids to use',
        )

    args = parser.parse_args()
    main(args)
