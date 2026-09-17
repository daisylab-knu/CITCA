import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the proposed CITCA model."
    )
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--text_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_fusion", type=float, default=1.4e-4)
    parser.add_argument("--lr_clf", type=float, default=5.5e-4)
    parser.add_argument("--wd", type=float, default=1.5e-2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.414)
    parser.add_argument("--clf_hidden_mult", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.19)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--vit_depth", type=int, default=1)
    parser.add_argument("--vit_mlp_dim", type=int, default=64)
    parser.add_argument("--qformer_depth", type=int, default=2)
    parser.add_argument(
        "--readout_type",
        choices=["query_readout", "cls", "gap"],
        default="query_readout",
    )
    parser.add_argument("--use_ocread", type=str2bool, default=False)
    parser.add_argument("--ocread_num_clusters", type=int, default=8)
    return parser.parse_args()
