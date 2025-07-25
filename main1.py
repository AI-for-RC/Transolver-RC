import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import optuna
from model_dict import get_model
from utils.testloss import TestLoss

#python main1.py --use_optuna 1 --n_trials 50 --gpu 0 --model Transolver_Structured_Mesh_3D --epochs 300
# 添加Optuna相关参数
parser = argparse.ArgumentParser('Training Transformer with Optuna')

# 原有参数
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='Transolver_Structured_Mesh_3D')
parser.add_argument('--n-hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n-layers', type=int, default=3, help='layers')
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--save_name', type=str, default='RC_Transolver')
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='train_log')
parser.add_argument('--resume_name', type=str, default='')

# Optuna新增参数
parser.add_argument('--use_optuna', type=int, default=0, help='Whether to use optuna for hyperparameter search')
parser.add_argument('--n_trials', type=int, default=50, help='Number of optuna trials')
parser.add_argument('--optuna_dir', type=str, default='./optuna_results', help='Directory to store optuna results')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train_with_params(args, params):
    """使用给定的超参数运行训练过程"""
    # 数据加载
    DATA_PATH = args.data_path

    N = 500
    ntrain = 400
    ntest = 100

    T = 49
    D = 51
    H = 6
    W = 6
    inputBeam_dim = 6
    inputElas_dim = 9
    space_dim = 3
    output_dim = 99
    inputBeam_num = 1836

    input_beam = np.load(DATA_PATH + '/input1.npy')
    input_elas = np.load(DATA_PATH + '/input2.npy')
    output = np.load(DATA_PATH + '/output.npy')

    input_beam = torch.tensor(input_beam, dtype=torch.float)
    input_elas = torch.tensor(input_elas, dtype=torch.float)
    input_beam = input_beam.permute(0, 3, 1, 2).reshape(N * T, -1, inputBeam_dim)
    input_elas = input_elas.permute(0, 3, 1, 2).reshape(N * T, -1, inputElas_dim)
    output = torch.tensor(output, dtype=torch.float)
    output = output.permute(0, 3, 1, 2).reshape(N * T, -1, output_dim)

    x1_train = input_beam[:ntrain * T]
    x2_train = input_elas[:ntrain * T]
    y_train = output[:ntrain * T]
    x1_test = input_beam[ntrain * T:ntrain * T + ntest * T]
    x2_test = input_elas[ntrain * T:ntrain * T + ntest * T]
    y_test = output[ntrain * T:ntrain * T + ntest * T]

    # 使用params中的batch_size
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x1_train, x2_train, y_train),
        batch_size=params.get('batch_size', args.batch_size),
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x1_test, x2_test, y_test),
        batch_size=params.get('batch_size', args.batch_size),
        shuffle=False
    )

    print("Dataloading is over.")

    # 创建模型，使用params中的超参数
    model = get_model(args).Model(
        space_dim=space_dim,
        n_layers=params.get('n_layers', args.n_layers),
        n_hidden=params.get('n_hidden', args.n_hidden),
        dropout=params.get('dropout', args.dropout),
        n_head=params.get('n_heads', args.n_heads),
        mlp_ratio=params.get('mlp_ratio', args.mlp_ratio),
        fun1_dim=inputBeam_dim - space_dim,
        fun2_dim=inputElas_dim - space_dim,
        out_dim=output_dim,
        slice_num=args.slice_num,
        D=D, H=H, W=W,
        inputBeam_num=inputBeam_num
    ).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get('lr', args.lr),
        weight_decay=params.get('weight_decay', args.weight_decay)
    )

    print(args)
    print(model)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get('lr', args.lr),
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )

    myloss = TestLoss()

    # 训练过程（不使用resume，因为每次试验都需要从头开始）
    for ep in range(args.epochs):
        model.train()
        train_loss = 0

        for x1, x2, y in train_loader:
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x1, x2).squeeze(-1)
            loss = myloss(out, y)
            loss.backward()

            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()

        # 验证
        model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for x1, x2, y in test_loader:
                x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
                out = model(x1, x2).squeeze(-1)
                tl = myloss(out, y).item()
                rel_err += tl

        rel_err /= ntest  # 计算平均测试误差

        # 打印进度（减少频率以避免过多输出）
        if (ep + 1) % 10 == 0:
            print(
                f"Epoch {ep + 1}/{args.epochs} - Train Loss: {train_loss / len(train_loader):.6f}, Test Error: {rel_err:.6f}")

    # 返回最终测试误差作为优化目标
    return rel_err


def objective(trial):
    """Optuna的目标函数，定义超参数搜索空间并评估性能"""
    # 定义要优化的超参数
    params = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-3),
        'n_hidden': trial.suggest_int('n_hidden', [128, 256]),
        'n_layers': trial.suggest_int('n_layers', [4, 8]),
        'n_heads': trial.suggest_int('n_heads', [4, 8]),
        'dropout': trial.suggest_uniform('dropout', [0.0, 0.1, 0.2, 0.3]),
        'mlp_ratio': trial.suggest_int('mlp_ratio', [1, 2, 3, 4]),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.01, 0.5, log=True)
    }

    # 打印当前试验的超参数
    print(f"Trial {trial.number} params: {params}")

    # 执行训练并返回测试误差
    test_error = train_with_params(args, params)

    # 记录超参数和结果到文件
    if not os.path.exists(args.optuna_dir):
        os.makedirs(args.optuna_dir)

    with open(os.path.join(args.optuna_dir, 'trials.log'), 'a') as f:
        f.write(f"Trial {trial.number}: {params} -> Test Error: {test_error}\n")

    return test_error


def main():
    if args.use_optuna:
        # 创建Optuna存储目录
        if not os.path.exists(args.optuna_dir):
            os.makedirs(args.optuna_dir)

        # 创建研究对象
        study = optuna.create_study(
            study_name='transformer_hyperparam_search',
            direction='minimize',  # 我们希望最小化测试误差
            storage=f'sqlite:///{os.path.join(args.optuna_dir, "optuna.db")}',
            load_if_exists=True
        )

        # 开始优化
        print(f"Starting Optuna hyperparameter optimization with {args.n_trials} trials...")
        start_time = time.time()

        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[lambda study, trial: print(f"Trial {trial.number} completed with value: {trial.value}")]
        )

        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")

        # 输出最佳参数
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.value}")
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # 保存最佳参数
        with open(os.path.join(args.optuna_dir, 'best_params.txt'), 'w') as f:
            f.write(f"Best Test Error: {best_trial.value}\n")
            for key, value in best_trial.params.items():
                f.write(f"{key}: {value}\n")

        # 可视化结果（需要plotly）
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(os.path.join(args.optuna_dir, 'param_importances.html'))

            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(os.path.join(args.optuna_dir, 'optimization_history.html'))
        except Exception as e:
            print(f"Visualization failed: {e}. Install plotly to generate visualizations.")

    else:
        # 原始训练逻辑（不使用Optuna）
        # 创建一个包含默认参数的字典
        default_params = {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'n_hidden': args.n_hidden,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'dropout': args.dropout,
            'mlp_ratio': args.mlp_ratio
        }

        if args.eval:
            # 评估模式
            DATA_PATH = args.data_path

            N = 500
            ntest = 100
            T = 49

            input_beam = np.load(DATA_PATH + '/input1.npy')
            input_elas = np.load(DATA_PATH + '/input2.npy')
            output = np.load(DATA_PATH + '/output.npy')

            input_beam = torch.tensor(input_beam, dtype=torch.float)
            input_elas = torch.tensor(input_elas, dtype=torch.float)
            input_beam = input_beam.permute(0, 3, 1, 2).reshape(N * T, -1, 6)
            input_elas = input_elas.permute(0, 3, 1, 2).reshape(N * T, -1, 9)
            output = torch.tensor(output, dtype=torch.float)
            output = output.permute(0, 3, 1, 2).reshape(N * T, -1, 99)

            ntrain = 400
            x1_test = input_beam[ntrain * T:ntrain * T + ntest * T]
            x2_test = input_elas[ntrain * T:ntrain * T + ntest * T]
            y_test = output[ntrain * T:ntrain * T + ntest * T]

            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x1_test, x2_test, y_test),
                batch_size=args.batch_size,
                shuffle=False
            )

            model = get_model(args).Model(
                space_dim=3,
                n_layers=args.n_layers,
                n_hidden=args.n_hidden,
                dropout=args.dropout,
                n_head=args.n_heads,
                mlp_ratio=args.mlp_ratio,
                fun1_dim=6 - 3,
                fun2_dim=9 - 3,
                out_dim=99,
                slice_num=args.slice_num,
                D=51, H=6, W=6,
                inputBeam_num=1836
            ).cuda()

            model.load_state_dict(torch.load("./checkpoints/" + args.save_name + ".pt")['model_state_dict'],
                                  weights_only=True)
            model.eval()

            myloss = TestLoss()
            rel_err = 0.0
            with torch.no_grad():
                for x1, x2, y in test_loader:
                    x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
                    out = model(x1, x2).squeeze(-1)
                    tl = myloss(out, y).item()
                    rel_err += tl
            rel_err /= ntest
            print(f"Test Error: {rel_err:.6f}")
        else:
            # 常规训练
            test_error = train_with_params(args, default_params)
            print(f"Final Test Error: {test_error:.6f}")


if __name__ == "__main__":
    main()