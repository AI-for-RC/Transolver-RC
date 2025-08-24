import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import *
from model_dict import get_model
from utils.testloss import TestLoss
import pyvista as pv
from scipy.interpolate import griddata

# --gpu 0 --model Transolver_Structured_Mesh_3D --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --epochs 10 --max_grad_norm 0.1 --batch-size 16 --slice_num 64 --eval 0 --resume 0 --save_name 15 --log_name 15
# --gpu 0 --model Transolver_Structured_Mesh_3D --n-hidden 128 --n-heads 8 --n-layers 8 --lr 0.001 --epochs 300 --max_grad_norm 0.1 --batch-size 16 --slice_num 64 --eval 1 --save_name 15 --log_name 15

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
parser.add_argument('--log_name', type=str, default='train_log.txt')

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
    inputBeam_dim = 5
    inputElas_dim = 5
    space_dim = 3
    output_dim = 3
    inputBeam_num = 1836

    input_beam = torch.tensor(np.load(DATA_PATH + '/input_beam.npy'), dtype=torch.float32)
    input_elas = torch.tensor(np.load(DATA_PATH + '/input_elas.npy'), dtype=torch.float32)
    output = torch.tensor(np.load(DATA_PATH + '/output_dis.npy'), dtype=torch.float32)

    # # input_beam: (N, D * H * W + E, C, T) (500, 1836, 3 + 3, 49)
    # # input_elas: (N, D * H * W + E, C, T) (500, 54, 3 + 6, 49)
    # input_beam = torch.tensor(input_beam, dtype=torch.float)
    # input_elas = torch.tensor(input_elas, dtype=torch.float)
    # # input_beam: (N * T, D * H * W + E, C) (24500, 1836, 6)
    # # input_elas: (N * T, D * H * W + E, C) (24500, 54, 9)
    # input_beam = input_beam.permute(0, 3, 1, 2).reshape(N * T, -1, inputBeam_dim)
    # input_elas = input_elas.permute(0, 3, 1, 2).reshape(N * T, -1, inputElas_dim)
    # np.save(DATA_PATH + '/input_beam.npy', input_beam.numpy())
    # np.save(DATA_PATH + '/input_elas.npy', input_elas.numpy())
    
    # # output: (N, D * H * W + E, C, T) (500, 1890, 99, 49)
    # output = torch.tensor(output, dtype=torch.float)
    # # (N * T, D * H * W + E, C) (24500, 1890, 99)
    # output = output.permute(0, 3, 1, 2).reshape(N * T, -1, 3)
    # output = output[:, :, :3]

    # input_beam = np.delete(input_beam, 4, axis=2)
    # input_elas = np.delete(input_elas, 7, axis=2)
    # input_elas = np.delete(input_elas, 6, axis=2)
    # input_elas = np.delete(input_elas, 4, axis=2)
    # input_elas = np.delete(input_elas, 3, axis=2)

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
    myloss = TestLoss()

    if eval:
        checkpoint = torch.load("./checkpoints/" + save_name + ".pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        rel_err = 0.0
        showcase = 10

        with torch.no_grad():
            start_id = 19601
            generated = 0
            for x1, x2, y in test_loader:
                batch_size = x1.shape[0]
                for b in range(batch_size):
                    cur_id = start_id + generated
                    coordinates = torch.cat((x1[b, :, :3], x2[b, :, :3]), dim=0).cpu().numpy()
                    x1_b, x2_b, y_b = x1[b:b+1].cuda(), x2[b:b+1].cuda(), y[b:b+1].cuda()
                    out = model(x1_b, x2_b).squeeze(-1)
                    displacements = out[0, :, :].cpu().numpy()
                    real_displacements = y_b[0, :, :].cpu().numpy()
                    tl = myloss(out, y_b).item()
                    rel_err += tl
                    print(f"生成ID: {cur_id}")
                    directions = ['X', 'Y', 'Z']
                    for i, dir_name in enumerate(directions):
                        point_cloud_pred = pv.PolyData(coordinates)
                        point_cloud_pred[f'{dir_name} Predicted Displacements'] = displacements[:, i]
                        plotter_pred = pv.Plotter(off_screen=True, window_size=(1920, 1080))
                        plotter_pred.add_mesh(
                            point_cloud_pred,
                            scalars=f'{dir_name} Predicted Displacements',
                            cmap='viridis',
                            render_points_as_spheres=True,
                            point_size=15,
                        )
                        plotter_pred.add_title(f'Predicted {dir_name.lower()} Displacements ID={cur_id}')
                        plotter_pred.camera.parallel_projection = True
                        bounds = point_cloud_pred.bounds
                        center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
                        max_range = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
                        camera_pos = [center[0], bounds[3]+max_range, center[2]]
                        plotter_pred.camera_position = [
                            camera_pos,
                            center,
                            [0, 0, 1]
                        ]
                        save_path_pred = f'./results/{save_name}/Predicted {dir_name.lower()} Displacements ID={cur_id}.png'
                        plotter_pred.show(screenshot=save_path_pred, window_size=(1920, 1080))
                        
                        point_cloud_real = pv.PolyData(coordinates)
                        point_cloud_real[f'{dir_name} Real Displacements'] = real_displacements[:, i]
                        plotter_real = pv.Plotter(off_screen=True, window_size=(1920, 1080))
                        plotter_real.add_mesh(
                            point_cloud_real,
                            scalars=f'{dir_name} Real Displacements',
                            cmap='viridis',
                            render_points_as_spheres=True,
                            point_size=15,
                        )
                        plotter_real.add_title(f'Real {dir_name.lower()} Displacements ID={cur_id}')
                        plotter_real.camera.parallel_projection = True
                        bounds = point_cloud_real.bounds
                        center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
                        max_range = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
                        camera_pos = [center[0], bounds[3]+max_range, center[2]]
                        plotter_real.camera_position = [
                            camera_pos,
                            center,
                            [0, 0, 1]
                        ]
                        save_path_real = f'./results/{save_name}/Real {dir_name.lower()} Displacements ID={cur_id}.png'
                        plotter_real.show(screenshot=save_path_real, window_size=(1920, 1080))

                        error = displacements[:, i] - real_displacements[:, i]
                        point_cloud_error = pv.PolyData(coordinates)
                        point_cloud_error[f'{dir_name} Error'] = error
                        plotter_error = pv.Plotter(off_screen=True, window_size=(1920, 1080))
                        plotter_error.add_mesh(
                            point_cloud_error,
                            scalars=f'{dir_name} Error',
                            cmap='coolwarm',
                            render_points_as_spheres=True,
                            point_size=15,
                        )
                        plotter_error.add_title(f'{dir_name.lower()} Displacement Error ID={cur_id}')
                        plotter_error.camera.parallel_projection = True
                        bounds = point_cloud_error.bounds
                        center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
                        max_range = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
                        camera_pos = [center[0], bounds[3]+max_range, center[2]]
                        plotter_error.camera_position = [
                            camera_pos,
                            center,
                            [0, 0, 1]
                        ]
                        save_path_error = f'./results/{save_name}/{dir_name.lower()} Displacement Error ID={cur_id}.png'
                        plotter_error.show(screenshot=save_path_error, window_size=(1920, 1080))
                    generated += 1
                    if generated >= showcase:
                        break
                if generated >= showcase:
                    break

        rel_err /= len(test_loader)
        print(f"Test Error: {rel_err:.6f}")
    else:
        epoch_pbar = tqdm(range(args.epochs), desc="Training Progress", position=0)
        
        log_dir = './log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, args.log_name + ".txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
            f.write("\nEpoch\tTrain Loss\tTest Error\n")

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

            rel_err /= len(test_loader)

            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss/len(train_loader):.6f}',
                'Test Error': f'{rel_err:.6f}',
            })
            
            print(f"\nEpoch {ep+1}/{args.epochs} - Train Loss: {train_loss/len(train_loader):.6f}, Test Error: {rel_err:.6f}")

            if (ep + 1) % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join('./checkpoints', save_name + '_' + str(ep + 1) + '.pt'))

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{ep+1}\t{train_loss/len(train_loader):.6f}\t{rel_err:.6f}\n")

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join('./checkpoints', save_name + '.pt'))

if __name__ == "__main__":
    main()