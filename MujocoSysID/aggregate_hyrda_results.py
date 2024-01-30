from pathlib import Path
from argparse import ArgumentParser
from print_colors import PrintColors
import pandas as pd
import shutil
import os
import yaml


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def main(env_name, override, partition, date, algs_override=None):
    if algs_override is not None:
        algs = [algs_override]
    else:
        algs = ["sysid", "mbpo", "lamps"]
    target_entries = 15

    for alg in algs:
        base_dir = Path("exp", alg, partition, env_name)
        csv_results_dir = Path("csv_results", env_name)
        csv_results_dir.mkdir(exist_ok=True, parents=True)
        if override:
            # clear the csv_results directory
            for f in csv_results_dir.glob("*"):
                if f.is_file() and alg in f.name:
                    f.unlink()

        # loop through date/run_id/multi_run_num
        for subdir in base_dir.glob(f"{date}/*"):
            if not subdir.is_dir():
                continue
            results_file = Path(subdir, "results.csv")
            if not results_file.exists():
                continue

            # Make sure results.csv has 150 entries
            try:
                df = pd.read_csv(results_file)
            except Exception as e:
                print(
                    f"{PrintColors.FAIL}Error reading {results_file}: {e}{PrintColors.ENDC}"
                )
                continue

            if len(df) < target_entries - 20 or len(df) < 4:
                print(
                    f"{PrintColors.FAIL}Skipping {results_file} because it has {len(df)} entries{PrintColors.ENDC}"
                )
                continue
            elif len(df) < target_entries:
                print(
                    f"{PrintColors.WARNING}Note: {results_file} has {len(df)} entries, which is less than {target_entries}{PrintColors.ENDC}"
                )

            # Look into .hydra directory for the hydra.yml file
            config_yaml = Path(subdir, ".hydra", "config.yaml")
            if not config_yaml.exists():
                print(f"{PrintColors.FAIL}Config file does not exist{PrintColors.ENDC}")
                continue
            # Read the seed number from the hydra.yml file
            with open(config_yaml, "r") as stream:
                try:
                    hydra_yml = yaml.safe_load(stream)
                    seed = hydra_yml.get("seed")
                    shaky = hydra_yml.get("shaky")
                    model_lr = hydra_yml.get("overrides").get("model_lr")
                    if model_lr == 0.001:
                        # sysid params
                        continue
                except yaml.YAMLError as exc:
                    print(exc)

            # Copy the csv file to the csv_results directory

            # for walker only
            suffix = ""
            if "walker" in env_name.lower():
                if model_lr == 0.001:
                    suffix = "-ogparams"
                else:
                    suffix = "-sysparams"

            if shaky:
                new_file_path = Path(
                    csv_results_dir,
                    f"{alg}_s{seed}_shaky_{partition.replace('_', '-')}{suffix}.csv",
                )
            else:
                new_file_path = Path(
                    csv_results_dir,
                    f"{alg}_s{seed}_{partition.replace('_', '-')}{suffix}.csv",
                )

            if new_file_path.exists():
                new_file_path = Path(
                    new_file_path.parent,
                    f"{alg}_s{seed+1}_shaky_{partition.replace('_', '-')}{suffix}.csv",
                )
                # continue
            assert new_file_path.parent.exists()
            shutil.copy(results_file, new_file_path)
            assert new_file_path.exists
            print(f"{results_file} ==> {new_file_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        choices=["ant", "hc", "hop", "hum", "walk", "div", "play"],
        help="Name of the environment",
    )
    parser.add_argument("-o", "--override", action="store_true", default=False)
    parser.add_argument("-a", type=str, default=None)
    parser.add_argument("-p", type=str, default="result")
    parser.add_argument("-d", type=str, required=True)
    args = parser.parse_args()

    env_abbr = args.env_name
    if env_abbr == "ant":
        env_name = "ant_truncated_obs"
    elif env_abbr == "hc":
        env_name = "HalfCheetah-v3"
    elif env_abbr == "hop":
        env_name = "Hopper-v3"
    elif env_abbr == "hum":
        env_name = "humanoid_truncated_obs"
    elif env_abbr == "walk":
        env_name = "Walker2d-v3"
    elif env_abbr == "div":
        env_name = "antmaze-large-diverse-v2"
    elif env_abbr == "play":
        env_name = "antmaze-large-play-v2"
    if "truncated" not in env_name:
        env_name = f"gym___{env_name}"
    main(env_name, args.override, args.p, args.d, args.a)
