#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import os
import sys
import time
import random
import numpy as np

from multiprocessing import freeze_support

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset, ConcatDataset, DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

from omegaconf import OmegaConf
from numpy.random import choice

# local imports
from layers.Autoformer_EncDec import moving_avg
from models import Autoformer, DLinear, TimeLLM, ST_TimeLLM_1, TEMPO, SALTT
from data_provider.data_factory_tempo import data_provider
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

# --------------------------- Env setup ---------------------------

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# --------------------------- Utils ---------------------------

FIX_SEED = 2021
random.seed(FIX_SEED)
np.random.seed(FIX_SEED)
torch.manual_seed(FIX_SEED)
torch.cuda.manual_seed_all(FIX_SEED)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_norm(x, d='norm'):
    means = x.mean().detach()
    x = x - means
    stdev = torch.sqrt(torch.var(x, unbiased=False) + 1e-5).detach()
    x /= stdev
    return x, means, stdev


def get_init_config(config_path=None):
    return OmegaConf.load(config_path)


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def plot_input(path, sample, saved_file_name):
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(10, 1))
    plt.plot(sample.cpu().detach().numpy(), label='Time Series Sample')
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    output_path = os.path.join(path, saved_file_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --------------------------- Main ---------------------------

def build_parser():
    parser = argparse.ArgumentParser(description='Time-LLM / SALTT Trainer')

    # task
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test')
    parser.add_argument('--model_comment', type=str, default='none')
    parser.add_argument('--model', type=str, default='SALTT',
                        help='[Autoformer, DLinear, TimeLLM, ST_TimeLLM_1, TEMPO, SALTT]')

    # data
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--datasets', type=str, default='ETTh1', help='the name list of the training dataset')
    parser.add_argument('--target_data', type=str, default='ETTh1', help='test dataset')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M', help='[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--loader', type=str, default='modal')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # forecasting lengths
    parser.add_argument('--seq_len', type=int, default=320)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Hourly')

    # model dims
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=16)#reprogamming attention size
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF', help='[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--patch_len', type=int, default=32)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--prompt_domain', type=int, default=0)
    parser.add_argument('--llm_model', type=str, default='GPT2', help='[LLAMA, GPT2, BERT]')
    parser.add_argument('--llm_dim', type=int, default=768)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--align_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='COS', help='[COS, OneCycle]')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=2)

    # options
    parser.add_argument('--output_attn_map', action='store_true', default=False,
                        help='output attention map of patches and prototype tokens')

    parser.add_argument('--config_path', type=str, default='./configs/multiple_datasets.yml')
    parser.add_argument('--electri_multiplier', type=int, default=1)
    parser.add_argument('--traffic_multiplier', type=int, default=1)
    parser.add_argument('--equal', type=int, default=1, help='1: equal sampling, 0: no equal sampling')

    # SALTT hyperparams
    parser.add_argument('--beta_S', type=float, default=2.0,
                        help='temperature (trend) before sigmoid')
    parser.add_argument('--beta_A', type=float, default=2.0,
                        help='temperature (accel) before sigmoid')
    parser.add_argument('--lambda_S', type=float, default=0.1,
                        help='loss weight for trend KL')
    parser.add_argument('--lambda_A', type=float, default=0.1,
                        help='loss weight for accel KL')
    parser.add_argument('--return_attn', type=bool, default=False,
                        help='return the atttention map between patches and labels')
    return parser


def build_setting_name(args):
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
        args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.des
    )


def build_model(args):
    if args.model == 'Autoformer':
        return Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        return DLinear.Model(args).float()
    elif args.model == 'ST_TimeLLM_1':
        return ST_TimeLLM_1.Model(args).float()
    elif args.model == 'TimeLLM':
        return TimeLLM.Model(args).float()
    elif args.model == 'TEMPO':
        return TEMPO.TEMPO(args).float()
    elif args.model == 'SALTT':
        return SALTT.Model(args).float()
    else:
        raise ValueError(f"Unknown model: {args.model}")


def train_one_experiment(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    setting = build_setting_name(args)
    path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    os.makedirs(path, exist_ok=True)

    # -------- build datasets (multi-dataset concat) --------
    train_datas, val_datas = [], []
    min_sample_num = sys.maxsize
    data_list = args.datasets.split(',')

    # pass 1: compute val sets and min_sample_num
    for dataset_name in data_list:
        args.data = config['datasets'][dataset_name].data
        args.root_path = config['datasets'][dataset_name].root_path
        args.data_path = config['datasets'][dataset_name].data_path
        args.data_name = config['datasets'][dataset_name].data_name
        args.features = config['datasets'][dataset_name].features
        args.freq = config['datasets'][dataset_name].freq or 'h'
        args.target = config['datasets'][dataset_name].target

        train_data_tmp, train_loader_tmp = data_provider(args, 'train')
        if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange']:
            min_sample_num = min(min_sample_num, len(train_data_tmp))

        vali_data_tmp, _ = data_provider(args, 'val')
        val_datas.append(vali_data_tmp)

    # pass 2: build train datasets with equal sampling/multipliers
    for dataset_name in data_list:
        args.data = config['datasets'][dataset_name].data
        args.root_path = config['datasets'][dataset_name].root_path
        args.data_path = config['datasets'][dataset_name].data_path
        args.data_name = config['datasets'][dataset_name].data_name
        args.features = config['datasets'][dataset_name].features
        args.freq = config['datasets'][dataset_name].freq or 'h'
        args.target = config['datasets'][dataset_name].target

        train_data_tmp, _ = data_provider(args, 'train')

        if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange'] and args.equal == 1:
            train_data_tmp = Subset(train_data_tmp, choice(len(train_data_tmp), min_sample_num))

        if args.electri_multiplier > 1 and args.equal == 1 and dataset_name in ['electricity']:
            train_data_tmp = Subset(train_data_tmp, choice(len(train_data_tmp),
                                                           int(min_sample_num * args.electri_multiplier)))

        if args.traffic_multiplier > 1 and args.equal == 1 and dataset_name in ['traffic']:
            train_data_tmp = Subset(train_data_tmp, choice(len(train_data_tmp),
                                                           int(min_sample_num * args.traffic_multiplier)))

        train_datas.append(train_data_tmp)

    # concat datasets if multiple
    if len(train_datas) > 1:
        train_data = ConcatDataset([train_datas[0], train_datas[1]])
        vali_data = ConcatDataset([val_datas[0], val_datas[1]])
        for i in range(2, len(train_datas)):
            train_data = ConcatDataset([train_data, train_datas[i]])
            vali_data = ConcatDataset([vali_data, val_datas[i]])
    else:
        train_data = train_datas[0]
        vali_data = val_datas[0]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    vali_loader = DataLoader(vali_data, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # test set from target_data
    target_name = args.target_data
    args.data = config['datasets'][target_name].data
    args.root_path = config['datasets'][target_name].root_path
    args.data_path = config['datasets'][target_name].data_path
    args.data_name = config['datasets'][target_name].data_name
    args.features = config['datasets'][target_name].features
    args.freq = config['datasets'][target_name].freq or 'h'
    args.target = config['datasets'][target_name].target
    test_data, test_loader = data_provider(args, 'test')

    # -------- build model & optimization --------
    model = build_model(args).float()
    args.content = load_content(args)

    # DataParallel
    model = nn.DataParallel(model).to(device)

    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate
        )

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    early_stopping = EarlyStopping(patience=args.patience)

    # -------- train loop --------
    time_now = time.time()
    save_epochs = [0, 2, 5, 10, 100]  # extra checkpoints

    for epoch in range(args.train_epochs):
        model.train()
        iter_count = 0
        train_losses = []
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in tqdm(
                enumerate(train_loader), total=len(train_loader)):

            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            seq_trend = seq_trend.float().to(device)
            seq_seasonal = seq_seasonal.float().to(device)
            seq_resid = seq_resid.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            if args.model == 'SALTT':
                outputs, slope_loss, acc_loss = model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark, seq_trend, seq_seasonal, seq_resid
                )
                # DataParallel may return per-GPU tensor [ngpu], reduce to scalar
                if slope_loss.dim() > 0:
                    slope_loss = slope_loss.mean()
                if acc_loss.dim() > 0:
                    acc_loss = acc_loss.mean()
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, seq_trend, seq_seasonal, seq_resid)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y)
            if args.model == 'SALTT':
                loss = loss + args.lambda_S * slope_loss + args.lambda_A * acc_loss

            train_losses.append(loss.item())

            if (i + 1) % 100 == 0:
                print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * len(train_loader) - i)
                print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                iter_count = 0
                time_now = time.time()

            loss.backward()
            model_optim.step()
            if args.lradj == 'COS':
                # step each iteration for Cosine? usually per-epoch; keep per-iter if desired
                pass
            else:
                scheduler.step()

        clear_memory()
        print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time:.2f}s")

        train_loss = float(np.average(train_losses))
        vali_loss, vali_mae_loss = vali(args, device, model, vali_data, vali_loader, criterion, mae_metric)
        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.7f}  Vali Loss: {vali_loss:.7f}")

        # save periodic checkpoints
        if epoch in save_epochs:
            ckpt_path = os.path.join(path, f"_epoch_{epoch}checkpoint")
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, ckpt_path)

        # early stopping (saves best as 'checkpoint')
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        if args.lradj == 'COS':
            scheduler.step()
            print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

    # -------- test best checkpoint --------
    best_ckpt = os.path.join(path, 'checkpoint')
    if not os.path.exists(best_ckpt):
        # if early stopping never saved, save current
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, best_ckpt)

    # rebuild fresh model and load best
    test_model = build_model(args).float()
    test_model = nn.DataParallel(test_model).to(device)
    state_dict = torch.load(best_ckpt, map_location=device)
    # allow missing prefixes
    try:
        test_model.load_state_dict(state_dict, strict=True)
    except Exception:
        new_state = {'module.' + k if not k.startswith('module.') else k: v for k, v in state_dict.items()}
        test_model.load_state_dict(new_state, strict=False)

    test_loss, test_mae_loss = vali(args, device, test_model, test_data, test_loader, criterion, mae_metric)
    print(f"Best model: MSE: {test_loss:.7f}  MAE: {test_mae_loss:.7f}")

    return test_loss, test_mae_loss, path, setting


def export_attention_maps(args, config, path, setting):
    """
    Loads several saved checkpoints and exports attention heatmaps.
    Assumes the underlying model returns (forecast, attn_maps) when output_attn_map=True in your model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rebuild dataset loaders (val/test) similarly
    train_datas, val_datas = [], []
    min_sample_num = sys.maxsize
    data_list = args.datasets.split(',')

    for dataset_name in data_list:
        args.data = config['datasets'][dataset_name].data
        args.root_path = config['datasets'][dataset_name].root_path
        args.data_path = config['datasets'][dataset_name].data_path
        args.data_name = config['datasets'][dataset_name].data_name
        args.features = config['datasets'][dataset_name].features
        args.freq = config['datasets'][dataset_name].freq or 'h'
        args.target = config['datasets'][dataset_name].target

        train_data_tmp, _ = data_provider(args, 'train')
        if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange']:
            min_sample_num = min(min_sample_num, len(train_data_tmp))

        vali_data_tmp, _ = data_provider(args, 'val')
        val_datas.append(vali_data_tmp)

    for dataset_name in data_list:
        args.data = config['datasets'][dataset_name].data
        args.root_path = config['datasets'][dataset_name].root_path
        args.data_path = config['datasets'][dataset_name].data_path
        args.data_name = config['datasets'][dataset_name].data_name
        args.features = config['datasets'][dataset_name].features
        args.freq = config['datasets'][dataset_name].freq or 'h'
        args.target = config['datasets'][dataset_name].target

        train_data_tmp, _ = data_provider(args, 'train')
        if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange'] and args.equal == 1:
            train_data_tmp = Subset(train_data_tmp, choice(len(train_data_tmp), min_sample_num))
        if args.electri_multiplier > 1 and args.equal == 1 and dataset_name in ['electricity']:
            train_data_tmp = Subset(train_data_tmp, choice(len(train_data_tmp), int(min_sample_num * args.electri_multiplier)))
        if args.traffic_multiplier > 1 and args.equal == 1 and dataset_name in ['traffic']:
            train_data_tmp = Subset(train_data_tmp, choice(len(train_data_tmp), int(min_sample_num * args.traffic_multiplier)))
        train_datas.append(train_data_tmp)

    if len(train_datas) > 1:
        train_data = ConcatDataset([train_datas[0], train_datas[1]])
        vali_data = ConcatDataset([val_datas[0], val_datas[1]])
        for i in range(2, len(train_datas)):
            train_data = ConcatDataset([train_data, train_datas[i]])
            vali_data = ConcatDataset([vali_data, val_datas[i]])
    else:
        train_data = train_datas[0]
        vali_data = val_datas[0]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    vali_loader = DataLoader(vali_data, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # test from target_data
    target_name = args.target_data
    args.data = config['datasets'][target_name].data
    args.root_path = config['datasets'][target_name].root_path
    args.data_path = config['datasets'][target_name].data_path
    args.data_name = config['datasets'][target_name].data_name
    args.features = config['datasets'][target_name].features
    args.freq = config['datasets'][target_name].freq or 'h'
    args.target = config['datasets'][target_name].target
    test_data, test_loader = data_provider(args, 'test')

    # build model for attention maps
    base_model = ST_TimeLLM_1.Model(args).float() if args.model not in ['Autoformer', 'DLinear'] else build_model(args)
    model = nn.DataParallel(base_model).to(device)

    # checkpoints to visualize (include best checkpoint as 100)
    save_epochs = [0, 2, 5, 10, 100]

    for epoch in save_epochs:
        if epoch != 100:
            ckpt_file = os.path.join(path, f"_epoch_{epoch}checkpoint")
        else:
            ckpt_file = os.path.join(path, 'checkpoint')

        if not os.path.exists(ckpt_file):
            print(f"Skip missing checkpoint: {ckpt_file}")
            continue

        state_dict = torch.load(ckpt_file, map_location=device)
        new_state = {'module.' + k if not k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state, strict=False)
        model.eval()

        # single batch for visualization
        batch = next(iter(test_loader))
        batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid = batch

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        seq_trend = seq_trend.float().to(device)
        seq_seasonal = seq_seasonal.float().to(device)
        seq_resid = seq_resid.float().to(device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        # plot some decomposed series
        sample = batch_x[0, :, :]
        sample_trend = seq_trend[0, :, :]
        sample_seasonal = seq_seasonal[0, :, :]
        sample_resid = seq_resid[0, :, :]
        plot_input(path, sample.squeeze(-1), 'input.png')
        plot_input(path, sample_trend.squeeze(-1), 'input_trend.png')
        plot_input(path, sample_seasonal.squeeze(-1), 'input_seasonal.png')
        plot_input(path, sample_resid.squeeze(-1), 'input_resid.png')

        # forward to obtain attention maps
        _, attn_map_list = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, seq_trend, seq_seasonal, seq_resid)

        if args.decomp_level == 1:
            num_attn_map = {'attn_original': attn_map_list[0]}
        elif args.decomp_level == 2:
            num_attn_map = {'attn_seasonal': attn_map_list[0], 'attn_trend': attn_map_list[1]}
        elif args.decomp_level == 3:
            num_attn_map = {
                'attn_seasonal': attn_map_list[0],
                'attn_trend': attn_map_list[1],
                'attn_residual': attn_map_list[2]
            }
        else:
            num_attn_map = {'attn_original': attn_map_list[0]}

        for k, v in num_attn_map.items():
            print("Export attention map:", str(k))
            attn_heads_fused = v.mean(axis=1).mean(axis=0)  # fuse heads & batch as needed
            plt.figure(figsize=(6, 6))
            sns.heatmap(attn_heads_fused.cpu().detach().numpy(), cmap='viridis', linewidths=0)
            plt.gca().set_aspect('auto')
            output_path = os.path.join(path, f"{k}_heatmap_epoch_{epoch}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    print(args)

    mses, maes = [], []
    config = get_init_config(args.config_path)

    if not args.output_attn_map:
        for _ in range(args.itr):
            test_mse, test_mae, path, setting = train_one_experiment(args, config)
            mses.append(test_mse)
            maes.append(test_mae)

        print("predict length:", args.pred_len)
        print("mse_mean = {:.4f}, mse_std = {:.4f}".format(float(np.mean(mses)), float(np.std(mses))))
        print("mae_mean = {:.4f}, mae_std = {:.4f}".format(float(np.mean(maes)), float(np.std(maes))))
    else:
        # if only exporting attention maps
        setting = build_setting_name(args)
        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        os.makedirs(path, exist_ok=True)
        export_attention_maps(args, config, path, setting)


if __name__ == '__main__':
    freeze_support()
    main()
