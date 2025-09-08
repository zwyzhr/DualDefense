import os
import datetime

from args import args_parser  # 导入自定义的参数解析函数

if __name__ == "__main__":
    # 解析命令行参数，返回一个包含所有超参数的对象
    args = args_parser()
    print(f"args: {args}")

    # 组装训练所需的所有参数配置，统一存入config字典，便于后续传递和管理
    config = {
        "num_clients": args.num_clients,                    # 客户端数量
        "dataset": args.dataset,                            # 数据集名称
        "fusion": args.fusion,                              # 聚合方式/算法
        "training_round": args.training_round,              # 总训练轮数
        "local_epochs": args.local_epochs,                  # 每个客户端本地训练的 epoch 数
        "optimizer": args.optimizer,                        # 优化器类型
        "learning_rate": args.learning_rate,                # 学习率
        "batch_size": args.batch_size,                      # 批大小
        "data_dir": args.dir_data,                          # 数据目录
        "partition_type": args.partition_type,              # 数据划分方式
        "partition_dirichlet_beta": args.partition_dirichlet_beta,  # Dirichlet 分布参数
        "regularization": args.regularization,              # 正则化参数
        "attacker_ratio": args.attacker_ratio,              # 恶意客户端比例
        "attacker_strategy": args.attacker_strategy,        # 攻击策略
        "attack_start_round": args.attack_start_round,      # 攻击发起的轮次
        "epsilon": args.epsilon,                            # 隐私参数/扰动参数
        "device": args.device,                              # 使用的设备（如cpu/gpu）
        "seed": args.seed,                                  # 随机种子
    }
    print(f"config: {config}")

    # 日志与Tensorboard相关路径设置
    log_dir = "./log"
    tensorboard_dir = "./log"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")  # 获取当前时间戳

    # 构建日志文件名，包含所有关键实验参数，便于追溯
    log_file = "{}-{}-{}-p{}r{}e{}b{}ar{}as{}-epsilon{}-{}.log".format(
        args.dataset,
        args.fusion,
        args.attacker_strategy,
        args.num_clients,
        args.training_round,
        args.local_epochs,
        args.batch_size,
        args.attacker_ratio,
        args.attack_start_round,
        args.epsilon,
        timestamp,
    )
    # 设置环境变量，供日志记录系统使用
    os.environ["LOG_FILE_NAME"] = os.path.join(log_dir, log_file)

    # 导入Tensorboard日志工具并初始化
    from utils.util_logger import setup_tensorboard

    # 构建Tensorboard文件名（与log_file类似，但无后缀）
    tb_file = "{}-{}-{}-p{}r{}e{}b{}ar{}as{}-epsilon{}-{}".format(
        args.dataset,
        args.fusion,
        args.attacker_strategy,
        args.num_clients,
        args.training_round,
        args.local_epochs,
        args.batch_size,
        args.attacker_ratio,
        args.attack_start_round,
        args.epsilon,
        timestamp,
    )

    # 初始化Tensorboard日志，记录到config中供后续使用
    config["tensorboard"] = setup_tensorboard(tensorboard_dir, tb_file)

    # 导入联邦学习主控模块
    from fl import SimulationFL

    # 创建联邦学习仿真对象，并启动训练流程
    fl = SimulationFL(config)
    fl.start()

