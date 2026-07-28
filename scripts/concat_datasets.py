import argparse
import warnings

# Suppress Zarr v3 warnings early
warnings.filterwarnings("ignore", message=r".*does not have a Zarr V3 specification.*")
warnings.filterwarnings("ignore", message=r".*is currently not part in the Zarr format 3 specification.*")

import ray

from t2.io.schema import concat_zarr_stores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_paths", type=str, nargs="+")
    parser.add_argument("--to_path", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=0)
    parser.add_argument(
        "--flattened_keys",
        type=str,
        nargs="+",
        default=[
            "rollout/metric",
            "rollout/done",
            "rollout/track_link_obs",
            "rollout/actuator_obs",
            "rollout/dyna_joint_obs",
            "rollout/free_link_obs",
            "rollout/target_pose",
            "rollout/ctrl",
            "hardware/dyna_joint",
            "hardware/link",
            "hardware/fixed_joint",
            "hardware/actuator",
            "hardware_meta/seed",
            "hardware_meta/fixed_joint/ends",
            "hardware_meta/link/ends",
            "hardware_meta/actuator/ends",
            "hardware_meta/dyna_joint/ends",
            "rollout_meta/seed",
            "rollout_meta/ends",
            "rollout_meta/hardware_id",
        ],
    )
    args = parser.parse_args()
    flattened_keys = args.flattened_keys
    if args.num_processes == 0:
        concat_zarr_stores(
            from_paths=args.from_paths,
            to_path=args.to_path,
            root_metadata={"from_paths": args.from_paths},
            flattened_keys=flattened_keys,
        )
    else:
        ray.init(num_cpus=args.num_processes)
        fn = ray.remote(concat_zarr_stores)
        tasks = [
            fn.remote(
                from_paths=args.from_paths,
                to_path=args.to_path,
                root_metadata={"from_paths": args.from_paths},
                flattened_keys=[flattened_key],
                overwrite=True,
            )
            for flattened_key in flattened_keys
        ]
        ray.get(tasks)
