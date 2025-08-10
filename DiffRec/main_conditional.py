#!/usr/bin/env python3
"""
Main script for conditional DiffRec training with user groups.
This demonstrates how to add and use conditionals in the diffusion model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
from datetime import datetime
import data_utils
from models.DNN import DNN
import models.gaussian_diffusion as gd
import evaluate_utils

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m_clean', help='dataset name')
    parser.add_argument('--data_path', type=str, default='../datasets/', help='data path')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]', help='topN for evaluation')
    parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='save path')
    parser.add_argument('--log_name', type=str, default='log', help='log name')
    parser.add_argument('--round', type=int, default=1, help='round')
    parser.add_argument('--time_type', type=str, default='cat', help='time type')
    parser.add_argument('--dims', type=str, default='[1000]', help='dims')
    parser.add_argument('--norm', action='store_true', help='normalize')
    parser.add_argument('--emb_size', type=int, default=10, help='embedding size')
    parser.add_argument('--mean_type', type=str, default='x0', help='mean type')
    parser.add_argument('--steps', type=int, default=5, help='steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='noise schedule')
    parser.add_argument('--noise_scale', type=float, default=0.0001, help='noise scale')
    parser.add_argument('--noise_min', type=float, default=0.0005, help='noise min')
    parser.add_argument('--noise_max', type=float, default=0.005, help='noise max')
    parser.add_argument('--sampling_noise', action='store_true', help='sampling noise')
    parser.add_argument('--sampling_steps', type=int, default=0, help='sampling steps')
    parser.add_argument('--reweight', action='store_true', help='reweight')
    parser.add_argument('--use_conditionals', action='store_true', help='use user group conditionals')
    
    args = parser.parse_args()
    print("args:", args)
    
    # Set device
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    
    # Load data with conditionals
    train_path = os.path.join(args.data_path, args.dataset, 'train_list.npy')
    valid_path = os.path.join(args.data_path, args.dataset, 'valid_list.npy')
    test_path = os.path.join(args.data_path, args.dataset, 'test_list.npy')
    
    print(f"Loading data from: {args.data_path}")
    train_data, valid_y_data, test_y_data, n_user, n_item, user_groups, group_stats = \
        data_utils.data_load(train_path, valid_path, test_path, create_conditions=args.use_conditionals)
    
    print("data ready.")
    
    # Create datasets with conditionals
    if args.use_conditionals and user_groups is not None:
        print("Creating conditional datasets...")
        train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()), user_groups)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, 
                                shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        
        if args.tst_w_val:
            tv_dataset = data_utils.DataDiffusion(
                torch.FloatTensor(train_data.toarray()) + torch.FloatTensor(valid_y_data.toarray()), 
                user_groups)
            test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        print("Creating standard datasets...")
        train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, 
                                shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        
        if args.tst_w_val:
            tv_dataset = data_utils.DataDiffusion(
                torch.FloatTensor(train_data.toarray()) + torch.FloatTensor(valid_y_data.toarray()))
            test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
    
    mask_tv = train_data + valid_y_data
    
    # Set mean type
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)
    
    # Create diffusion model
    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
            args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)
    
    # Build MLP with conditionals
    out_dims = eval(args.dims) + [n_item]
    in_dims = out_dims[::-1]
    
    # Build model
    if args.use_conditionals:
        print("Building conditional DNN model...")
        model = DNN(in_dims, out_dims, args.emb_size, args.time_type, args.norm, 
                          dropout=0.5, use_conditionals=True, conditional_dim=2).to(device)
    else:
        print("Building standard DNN model...")
        model = DNN(in_dims, out_dims, args.emb_size, args.time_type, args.norm, 
                          dropout=0.5, use_conditionals=False).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("models ready.")
    
    # Count parameters
    param_num = 0
    mlp_num = sum([param.nelement() for param in model.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])
    param_num = mlp_num + diff_num
    print("Number of all parameters:", param_num)
    
    def evaluate(data_loader, data_te, mask_his, topN, user_groups=None):
        model.eval()
        e_idxlist = list(range(mask_his.shape[0]))
        e_N = mask_his.shape[0]

        predict_items = []
        target_items = []
        for i in range(e_N):
            target_items.append(data_te[i, :].nonzero()[1].tolist())
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
                
                if args.use_conditionals and user_groups is not None:
                    # Handle conditional data
                    if isinstance(batch, (list, tuple)):
                        batch_data, batch_conditionals = batch
                        batch_data = batch_data.to(device)
                        batch_conditionals = batch_conditionals.to(device)
                    else:
                        # Fallback if data format is unexpected
                        batch_data = batch.to(device)
                        batch_conditionals = None
                else:
                    batch_data = batch.to(device)
                    batch_conditionals = None
                
                # Generate predictions with or without conditionals
                if args.use_conditionals and batch_conditionals is not None:
                    prediction = diffusion.p_sample(model, batch_data, args.sampling_steps, 
                                                 args.sampling_noise, batch_conditionals)
                else:
                    prediction = diffusion.p_sample(model, batch_data, args.sampling_steps, 
                                                 args.sampling_noise)
                
                prediction[his_data.nonzero()] = -np.inf

                _, indices = torch.topk(prediction, topN[-1])
                indices = indices.cpu().numpy().tolist()
                predict_items.extend(indices)

        test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
        return test_results
    
    # Training loop
    best_recall, best_epoch = -100, 0
    best_results = None
    best_test_results = None
    print("Start training...")
    
    for epoch in range(1, args.epochs + 1):
        if epoch - best_epoch >= 20:
            print('-'*18)
            print('Exiting from training early')
            break

        model.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            batch_count += 1
            optimizer.zero_grad()
            
            if args.use_conditionals and user_groups is not None:
                # Handle conditional data
                if isinstance(batch, (list, tuple)):
                    batch_data, batch_conditionals = batch
                    batch_data = batch_data.to(device)
                    batch_conditionals = batch_conditionals.to(device)
                else:
                    batch_data = batch.to(device)
                    batch_conditionals = None
            else:
                batch_data = batch.to(device)
                batch_conditionals = None
            
            # Calculate loss with or without conditionals
            if args.use_conditionals and batch_conditionals is not None:
                losses = diffusion.training_losses(model, batch_data, args.reweight, batch_conditionals)
            else:
                losses = diffusion.training_losses(model, batch_data, args.reweight)
            
            loss = losses["loss"].mean()
            total_loss += loss
            loss.backward()
            optimizer.step()
        
        # Evaluation
        if epoch % 5 == 0:
            valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN), user_groups)
            if args.tst_w_val:
                test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN), user_groups)
            else:
                test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN), user_groups)
            evaluate_utils.print_results(None, valid_results, test_results)

            if valid_results[1][1] > best_recall:  # recall@20 as selection
                best_recall, best_epoch = valid_results[1][1], epoch
                best_results = valid_results
                best_test_results = test_results

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                
                # Save model with conditional info in filename
                conditional_suffix = "_conditional" if args.use_conditionals else "_standard"
                torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}{}.pth' \
                    .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                    args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name, conditional_suffix))
        
        print("Running Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
        print('---'*18)

    print('==='*18)
    print("End. Best Epoch {:03d} ".format(best_epoch))
    evaluate_utils.print_results(None, best_results, best_test_results)   
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

if __name__ == "__main__":
    main()
