import json


def str2bool(v):
    return v.lower() in ("true", "t")


def add_main_args(parser):
    parser.add_argument("--lit_model", type=str, default="LitResNet")
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--overload_ckpt_cfg", type=int, default=0)

    parser.add_argument("--max_epoch", type=int, default=400)
    parser.add_argument("--step_per_epoch", type=int, default=5)

    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--mode", type=str, choices=("train", "test"), default="train")

    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--logger", nargs="+", type=str, choices=("tb", "wandb"), default=("tb", "wandb"))

    # optimizer
    parser.add_argument("--optim", type=str, choices=("SGD", "Adam", "AdamW", "RAdam"), default="AdamW")
    parser.add_argument("--lr", type=float, default=0.0002)  # 1e-3 ~ 1e-12
    parser.add_argument("--weight_decay", type=float, default=0.01)  # 1e-1 ~ 1e-12
    parser.add_argument("--act_w", type=float, default=1)  # 0~1

    # scheduler
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=("CosineAnnealingWarmRestarts", "OneCycleLR"),
        default="CosineAnnealingWarmRestarts",
    )
    parser.add_argument("--T0", type=int, default=11)  # 11~150
    parser.add_argument("--T_mult", type=int, default=2)  # 1~3
    parser.add_argument("--max_lr", type=float, default=0.1)  # 0.08 ~ 0.2
    parser.add_argument("--T_up", type=int, default=10)  # 1~10
    parser.add_argument("--gamma", type=float, default=0.1)  # 0.5~1

    # data
    parser.add_argument("--dset_path", type=str, default="/workspace/data/data_max_neigh")
    parser.add_argument("--batch_size", type=int, default=1024)  # 1024

    # augmentation
    parser.add_argument("--horizontal_flip_prob", type=float, default=0.5)  # 0~0.5
    parser.add_argument("--noise_prob", type=float, default=0.8)  # 0~1
    parser.add_argument("--max_num_noise", type=int, default=3)  # 2~10
    parser.add_argument("--max_value_pert", type=float, default=0.2)  # 0.2~1
    parser.add_argument("--max_pert", type=float, default=0.5)  # time perturbation. # 0.2~0.5

    # exclude acts
    parser.add_argument("--proh_acts", nargs="+", type=int, default=(88, 99))
    parser.add_argument("--excluded_acts", nargs="+", type=int, default=())

    # train test split
    parser.add_argument("--train_ids", nargs="+", type=int, default=(2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15))
    parser.add_argument("--test_ids_", type=int, default=8)

    # input data, target label
    parser.add_argument("--fog_map", nargs="+", type=int, default=(0, 0, 1, 1))
    parser.add_argument("--act_map", nargs="+", type=int, default=(0, 0, 1, 3, 2, 3, 2, 3, 2))

    # resnet hp
    parser.add_argument("--net", type=str, choices=("res18", "res34"), default="res34")
    parser.add_argument("--block_channel_starts", type=int, default=256)  # 64~512 (res34에선 절반 됨)

    # before intervals
    parser.add_argument("--before_interval_diff1", type=int, default=4)  # 1~4
    parser.add_argument("--before_interval_num1", type=int, default=30)  # 5~30
    parser.add_argument("--before_interval_diff2", type=int, default=50)  # 5~50
    parser.add_argument("--before_interval_num2", type=int, default=10)  # 2~10
    # parser.add_argument("--before_intervals_mode", choices=("square", "lin"), type=str, default="square")

    # target
    # parser.add_argument("--targets", nargs="+", type=str, default=("act",))
    parser.add_argument("--targets_str", type=str, choices=("fog", "act", "fog_act"), default="fog")
    # parser.add_argument("--targets", nargs="+", type=str, default=("fog", "act"))
    # ('fog',), ('act',), ('fog', 'act')
    parser.add_argument("--ip", type=int, default=113)
    parser.add_argument("--swing_neigh", type=str, default="true")

    with open("xli0t5hw.json", "r") as f:
        config = json.load(f)

    result = {}
    for key, value in config.items():
        result[key] = value["value"]
    parser.set_defaults(**result)

    parser.add_argument("--job_type", type=str, default="xli0t5hw_test_change_fog_2cls")
    return parser
