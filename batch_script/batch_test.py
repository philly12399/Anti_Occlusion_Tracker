import os
import sys
import click
from xinshuo_io import mkdir_if_missing
import pdb

@click.command()
### Add your options here
@click.option(
    "--det_root",
    "-d",
    type=str,
    default="./data/KITTI/detection/",
    help="Path of detection root .",
)
@click.option(
    "--config",
    "-c",
    type=str,
    default="KITTI_det_seq21.yml",
    help="Name of config file .",
)
@click.option(
    "--eval_config",
    "-ec",
    type=str,
    default="KITTI_seq21.yml",
    help="Name of eval config file .",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default="./data/track_exp/KITTI/",
    help="Dataset Name",
)
@click.option(
    "--exp",
    "-e",
    type=str,
    default="exp1",
    help="Name of exp.",
)
def main(det_root, config, eval_config, output, exp):
    diff_range=[0,1,2,3,4]
    # diff_range=[3,4]
    
    output_path = os.path.join(output,exp)
    if os.path.exists(output_path):
        os.system(f"rm -r {output_path}")
        # print(f"EXP {output_path} already exist")
        # exit()
        
    os.system(f"mkdir -p {output_path}")
    # os.system(f"source env.sh")
    link = os.path.join(det_root, "det_link")   
    if os.path.exists(link):
        os.system(f"unlink {link}") 
    for d in diff_range:
        target = f"diff{d}"
        print(f"============================ Run {target} ============================")
        os.system(f"ln -sf ./{target} {link}")
        os.system(f"python3 main.py --config ./configs/{config}")
        os.system(f"python3 ./scripts/KITTI/evaluate.py --config ./eval_configs/{eval_config}")
        os.system(f"mv {output}/exp {output_path}/{target}")
        os.system(f"unlink {link}")
    
if __name__ == "__main__":
    main()
