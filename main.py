import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import *
from model_dict import get_model
from utils.testloss import TestLoss

# --gpu 0 --model Transolver_Structured_Mesh_3D --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --max_grad_norm 0.1 --batch-size 4 --slice_num 64 --eval 0 --save_name RC_Transolver

parser = argparse.ArgumentParser('Training Transformer')

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
parser.add_argument('--save_name', type=str, default='RC_Transolver')
parser.add_argument('--data_path', type=str, default='./data')
args = parser.parse_args()
eval = args.eval
save_name = args.save_name
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():
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

    # input_beam: (N, D * H * W + E, C, T) (500, 1836, 3 + 3, 49)
    # input_elas: (N, D * H * W + E, C, T) (500, 54, 3 + 6, 49)
    input_beam = torch.tensor(input_beam, dtype=torch.float)
    input_elas = torch.tensor(input_elas, dtype=torch.float)
    # input_beam: (N * T, D * H * W + E, C) (24500, 1836, 6)
    # input_elas: (N * T, D * H * W + E, C) (24500, 54, 9)
    input_beam = input_beam.permute(0, 3, 1, 2).reshape(N * T, -1, inputBeam_dim)
    input_elas = input_elas.permute(0, 3, 1, 2).reshape(N * T, -1, inputElas_dim)
    
    # output: (N, D * H * W + E, C, T) (500, 1890, 99, 49)
    output = torch.tensor(output, dtype=torch.float)
    # (N * T, D * H * W + E, C) (24500, 1890, 99)
    output = output.permute(0, 3, 1, 2).reshape(N * T, -1, output_dim)

    print(input_beam.shape)
    print(input_elas.shape)
    print(output.shape)

    x1_train = input_beam[:ntrain * T]
    x2_train = input_elas[:ntrain * T]
    y_train = output[:ntrain * T]
    x1_test = input_beam[ntrain * T:ntrain * T + ntest * T]
    x2_test = input_elas[ntrain * T:ntrain * T + ntest * T]
    y_test = output[ntrain * T:ntrain * T + ntest * T]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x1_train, x2_train, y_train),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x1_test, x2_test, y_test),
                                              batch_size=args.batch_size,
                                              shuffle=False)
    
    print("Dataloading is over.")

    model = get_model(args).Model(space_dim=space_dim,
                                  n_layers=args.n_layers,
                                  n_hidden=args.n_hidden,
                                  dropout=args.dropout,
                                  n_head=args.n_heads,
                                  mlp_ratio=args.mlp_ratio,
                                  fun1_dim=inputBeam_dim - space_dim,
                                  fun2_dim=inputElas_dim - space_dim,
                                  out_dim=output_dim,
                                  slice_num=args.slice_num,
                                  D=D, H=H, W=W,
                                  inputBeam_num=inputBeam_num).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args)
    print(model)
    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"))
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        rel_err = 0.0
        showcase = 10
        id = 0

        with torch.no_grad():
            for x1, x2, y in test_loader:
                id += 1
                x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
                out = model(x1, x2).squeeze(-1)
                tl = myloss(out, y).item()
                rel_err += tl
                if id < showcase:
                    print(id)
                    # save and display the output

        rel_err /= ntest
        print(f"Test Error: {rel_err:.6f}")
    else:
        epoch_pbar = tqdm(range(args.epochs), desc="Training Progress", position=0)
        
        for ep in epoch_pbar:
            model.train()
            train_loss = 0

            batch_pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{args.epochs}", 
                             position=1, leave=False)
            
            for x1, x2, y in batch_pbar:
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

                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{train_loss/(batch_pbar.n+1):.6f}'
                })

            model.eval()
            rel_err = 0.0
            with torch.no_grad():
                for x1, x2, y in test_loader:
                    x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
                    out = model(x1, x2).squeeze(-1)

                    tl = myloss(out, y).item()
                    rel_err += tl

            rel_err /= ntest

            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss/len(train_loader):.6f}',
                'Test Error': f'{rel_err:.6f}'
            })
            
            print(f"\nEpoch {ep+1}/{args.epochs} - Train Loss: {train_loss/len(train_loader):.6f}, Test Error: {rel_err:.6f}")

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

if __name__ == "__main__":
    main()