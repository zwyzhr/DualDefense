from __future__ import print_function

import argparse

def args_parser():
    parser = argparse.ArgumentParser(
        description="带有防御机制的联邦学习模拟实验"
    )

    # 联邦学习相关参数
    parser.add_argument(
        "-np",
        "--num_clients",
        type=int,
        default=5,
        help="分布式集群中的客户端数量",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fmnist", "cifar10", "svhn"],
        help="训练使用的数据集",
    )
    parser.add_argument(
        "-pd",
        "--partition_type",
        type=str,
        default="noniid",
        help="数据划分策略（如 iid 或 noniid）",
    )
    parser.add_argument(
        "-pb",
        "--partition_dirichlet_beta",
        type=float,
        default=0.25,
        help="Dirichlet分布参数，用于数据非独立划分",
    )
    parser.add_argument(
        "-f",
        "--fusion",
        choices=[
            "average",
            "fedavg",
            "krum",
            "median",
            "clipping_median",
            "trimmed_mean",
            "cos_defense",
            "dual_defense",
        ],
        type=str,
        default=0.5,
        help="聚合方法",
    )
    parser.add_argument(
        "-dm",
        "--dir_model",
        type=str,
        required=False,
        default="./models/",
        help="模型存放目录",
    )
    parser.add_argument(
        "-dd",
        "--dir_data",
        type=str,
        required=False,
        default="./data/",
        help="数据存放目录",
    )

    # 超参数设置
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.01,
        help="学习率（默认0.01）",
    )
    parser.add_argument(
        "-le", "--local_epochs", type=int, default=1, help="每个客户端本地训练的轮数"
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="训练时输入的批次大小",
    )
    parser.add_argument(
        "-tr",
        "--training_round",
        type=int,
        default=100,
        help="最大通信轮数",
    )
    parser.add_argument(
        "-re",
        "--regularization",
        type=float,
        default=1e-5,
        help="L2正则化系数",
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "amsgrad"],
        help="训练过程所用优化器",
    )

    # 攻击与防御相关设置
    parser.add_argument(
        "--attacker_ratio",
        type=float,
        default=0.2,
        required=False,
        help="恶意客户端的比例",
    )
    parser.add_argument(
        "--attacker_strategy",
        type=str,
        default="none",
        required=False,
        choices=[
            "none",
            "model_poisoning_ipm",
            "model_poisoning_scaling",
            "model_poisoning_alie",
        ],
        help="恶意客户端采用的攻击策略",
    )
    parser.add_argument(
        "--attack_start_round",
        type=int,
        default=-1,
        required=False,
        help="攻击开始的轮数（-1表示不启动攻击）",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        required=False,
        help="dual_defense方法中使用的差分隐私参数epsilon",
    )
    # 其它如触发器等相关参数（暂注释掉）
    # parser.add_argument('--trigger_label', type=int, default=1, help='触发器标签编号')
    # parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='触发器图片路径')
    # parser.add_argument('--trigger_size', type=int, default=5, help='触发器尺寸')

    # 其它设置
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        required=False,
        choices=["cpu", "mps", "cuda"],
        help="PyTorch训练使用的设备",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7890,
        required=False,
        help="随机种子",
    )

    return parser.parse_args()  # 返回解析后的参数对象

