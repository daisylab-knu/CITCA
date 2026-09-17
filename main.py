import argparse
import json
import os
from datetime import datetime


def configure_gpu() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    args, _ = parser.parse_known_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))


configure_gpu()

from setting import get_args


def main() -> None:
    args = get_args()
    from train import Experiment
    from utils import set_seed

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    experiment = Experiment(args, output_dir=args.output_dir)
    metrics = experiment.train()

    result_path = os.path.join(
        args.output_dir,
        f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump({"config": vars(args), "metrics": metrics}, file, indent=2)
    print(f"Training metrics saved to {result_path}")


if __name__ == "__main__":
    main()
