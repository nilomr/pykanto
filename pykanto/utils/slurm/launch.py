# ─── DESCRIPTION ──────────────────────────────────────────────────────────────

"""Submit a slurm job that employs distributed/parallel computation using
ray."""

# ─── DEPENDENCIES ─────────────────────────────────────────────────────────────

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ──── SETTINGS ────────────────────────────────────────────────────────────────

JOB_NAME = "{{JOB_NAME}}"
NUM_NODES = "{{NUM_NODES}}"
GPU = "{{GPU}}"
PARTITION_NAME = "{{PARTITION_NAME}}"
COMMAND_PLACEHOLDER = "{{COMMAND_PLACEHOLDER}}"
GIVEN_NODE = "{{GIVEN_NODE}}"
COMMAND_SUFFIX = "{{COMMAND_SUFFIX}}"
LOAD_ENV = "{{LOAD_ENV}}"
TIME = "{{TIME}}"
MEMORY = "{{MEMORY}}"
OUT_DIR = "{{OUT_DIR}}"

# ──── FUNCTIONS ────────────────────────────────────────────────────────────────


def submit_job():
    """
    Parses arguments and submits a ray job to slurm.
    Code from Peng Zhenghao; modifications (c) 2021 Nilo M. Recalde.

    See `source code by Peng Zhenghao
    <https://github.com/pengzhenghao/use-ray-with-slurm>`_. Also see `ray
    instructions <https://docs.ray.io/en/ray-1.1.0/cluster/slurm.html>`_.

    Run `pykanto-slaunch --help` for arguments.
    Output bash and log files are saved in a `/logs` directory within
    the directory from which you called the script. You can easily change
    this behaviour by editing the `out_dir` below

    Note:
        This works as of 2022 @ Oxford University ARC HPC. Chances are it will
        not work for you 'out of the box'; this submodule is intended more as a
        guide or reference than a foolproof way of submitting multi-node / GPU
        ray jobs.

    """
    # TODO: #16 @nilomr: migrate CLI to typer

    # Locate bash template file
    template_file = Path(__file__).parent / "sbatch_template.sh"

    # Path to output folder
    out_dir = Path(os.getcwd()) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        "-exp",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).",
    )
    parser.add_argument(
        "--num-nodes", "-n", type=int, default=1, help="Number of nodes to use."
    )
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        default="",
        help="The specified nodes to use. Same format as the return of 'sinfo'. Default: ''.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=0,
        help="Number and constraints on GPUs to use. (Default: 0)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="00:10:00",
        help="Time allocated to job. (Default: '00:10:00')",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=20000,
        help="Memory allocated to job. (Default: 20000)",
    )
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
        default="short",
    )
    parser.add_argument(
        "--load-env",
        "-env",
        type=str,
        default="",
        required=True,
        help=(
            "The name of your environment. Note: you have to provide "
            "the location of your envs in the `sbatch_template.sh` file"
        ),
    )
    parser.add_argument(
        "--command",
        "-c",
        type=str,
        required=True,
        help="The command you wish to execute. For example: --command 'python "
        "test.py' Note that the command must be a string.",
    )
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(
        args.exp_name, time.strftime("%m%d-%H%M", time.localtime())
    )

    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(GPU, str(args.gpu))
    text = text.replace(TIME, str(args.time))
    text = text.replace(PARTITION_NAME, str(args.partition))
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(MEMORY, str(args.memory))
    text = text.replace(COMMAND_SUFFIX, "")
    text = text.replace(OUT_DIR, str(out_dir))

    # ===== Save the script =====
    script_file = str(out_dir / f"{job_name}.sh")
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Submitting job.")
    subprocess.Popen(["sbatch", script_file])
    print(
        f"Job submitted! Script file is at: {script_file}. "
        f"Log file is at: {str(out_dir / job_name)}.log"
    )
    sys.exit(0)


if __name__ == "__main__":
    submit_job()
