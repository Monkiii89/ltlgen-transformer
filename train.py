# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import os
# from sklearn.metrics import f1_score
from os.path import join
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter

from data_m import TrainSet_generation, TrainSet_instantiation
from model import RNN

parser = ArgumentParser(description='Train SatVSCNet_TG')

parser.add_argument('--device', type=int, default=0, help="GPU number")
parser.add_argument('--tm', type=str, default=None, help="trained model")
parser.add_argument('--td', type=str, default='./data/debug.json', help="training data")

parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--bs', type=int, default=64, help="batch size")
parser.add_argument('--e', type=int, default=102400, help="epochs")

parser.add_argument('--d_model', type=int, default=512, help="transformerdecoder's d_model")
parser.add_argument('--n_head', type=int, default=8, help="transformerdecoder's n_head")
parser.add_argument('--embedding_size', type=int, default=512, help="nn.embedding")
parser.add_argument('--phase', type=str, default='generation', choices=['generation', 'instantiation'], help="the phase to select data process model")
parser.add_argument('--token_types', type=int, default=10, help="the first skeletion generation phase has 10 tokens, the second instantiation phase has 31 tokens", choices=[10,31])
parser.add_argument('--max_length', type=int, default=301, help="the max length of formulae")

parser.add_argument('--ear', type=int, default=10000, help="early stop")

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.cuda.device_count()
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

# para
early = args.ear            # 早停参数，early次评估未更新best model，结束训练
clip_grads = True           # 是否进行梯度截断
if args.phase =='generation':
    data_model = TrainSet_generation
else:
    data_model = TrainSet_instantiation

def train():

    train_dataset = data_model(path=args.td)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name_dir = f'time_{time_str}_dmodel_{args.d_model}_nhead_{args.n_head}_embeddingsize_{args.embedding_size}_bs_{args.bs}'
    os.makedirs(f'log/{name_dir}')
    os.makedirs(f'model/{name_dir}')
    # summaryWriter = SummaryWriter(f'log/{name_dir}')
    model = RNN(
                d_model=args.d_model,
                n_head=args.n_head,
                embedding_size=args.embedding_size,
                token_types=args.token_types,
                max_length = args.max_length,
                device=device).to(device)
    if args.tm is not None:
        print(f"Load model at {args.tm}.")
        model.load_state_dict(torch.load(args.tm, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

    total_step = 0 # 记录总的迭代次数
    # best_loss = float('inf')
    best_acc = 0
    # best_f1 = 0
    epsilon = 0
    epoch = 0
    # iii =0
    # for epoch in range(args.e):
    # while iii<3:
        # iii+=1
    while True:
        print('Epoch %d.' % (epoch + 1))

        # 训练
        model.train()
        loss = 0
        for x_batch, y_pos_batch, y_batch, x_mask_batch in tqdm(train_loader, desc='Training', ncols=80):
            # Batch_size = len(x_batch)
            x_batch = x_batch.to(device)
            y_pos_batch = y_pos_batch.to(device)
            y_batch = y_batch.to(device)
            x_mask_batch = x_mask_batch.to(device)
            optimizer.zero_grad()

            pred_batch = model(x_batch, x_mask_batch, y_pos_batch)
            # pred_batch = torch.argmax(pred_batch, dim=1)

            # compute sv loss
            loss = criterion(pred_batch, y_batch)
            
            # for name, parms in model.named_parameters():	
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:',parms.requires_grad)
            #     print('-->grad_value:',parms.grad)
            #     print("===")
            #     break
            loss.backward()
            
            # y_pred = pred_batch.argmax(dim=1)
            # ypred = torch.where(pred_batch>0.5, 1, pred_batch)
            # ypred = torch.where(ypred<0.5, 0, ypred)
            # ybatch = torch.tensor(y_batch, dtype=torch.long)
            # ypred = torch.tensor(ypred, dtype=torch.long)
            # accs = torch.tensor([torch.equal(ypred[i], ybatch[i]) for i in range(Batch_size)], dtype=torch.bool).sum()/Batch_size
            # train_acc = ((ypred == ybatch).sum())/Batch_size
            
            optimizer.step()
            # for name, parms in model.named_parameters():	
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:',parms.requires_grad)
            #     print('-->grad_value:',parms.grad)
            #     print("===")
            #     break
            # print(loss.cpu().item())
            
            total_step += 1
            # summaryWriter.add_scalar("training loss", loss.cpu().item(), total_step)
            # summaryWriter.add_scalar("training acc", train_acc, total_step)
        ypreds = []
        ybatchs = []
        epoch += 1
        # accs = 0
        with torch.no_grad():
            model.eval()
            for x_batch, y_pos_batch, y_batch, x_mask_batch in tqdm(train_loader, desc='Training', ncols=80):
                # Batch_size = len(x_batch)
                x_batch = x_batch.to(device)
                y_pos_batch = y_pos_batch.to(device)
                y_batch = y_batch.to(device)
                x_mask_batch = x_mask_batch.to(device)
                
                pred_batch = model(x_batch, x_mask_batch, y_pos_batch)
                # pred_batch = torch.argmax(pred_batch, dim=1)
                
                # pred_batch = nn.functional.softmax(pred_batch, dim=1)
                # ypred = torch.where(pred_batch>0.5, 1, pred_batch)
                # ypred = torch.where(ypred<0.5, 0, ypred)
                # ybatch = torch.tensor(y_batch, dtype=torch.long)
                # ypred = torch.tensor(ypred, dtype=torch.long)
                ypreds.append(pred_batch)
                ybatchs.append(y_batch)
                
                # accs = torch.tensor([torch.equal(ypred[i], ybatch[i]) for i in range(Batch_size)], dtype=torch.bool).sum()/Batch_size
                # train_acc = ((pred_batch == y_batch).sum())/Batch_size
                
            # f1score = F1_score(y_batch=ybatchs, y_pred=ypreds)
            ypreds = torch.cat(ypreds)
            ypreds = torch.argmax(ypreds, dim=1)
            ybatchs = torch.cat(ybatchs).reshape(-1)
            accs = (ypreds==ybatchs).sum()/(ybatchs.size(0))
            print(ypreds, accs)
            if accs > best_acc:
                best_acc = accs
                # best_path = join(f'model/{name_dir}', 'step{%d}-lr{%.4f}-early{%d}-loss{%.2f}-acc{%.2f}.pth' % (total_step, args.lr, args.ear, best_loss, accs))
                best_path = join(f'model/{name_dir}', 'step{%d}-lr{%.4f}-early{%d}-acc{%.2f}.pth' % (total_step, args.lr, args.ear, best_acc))
                torch.save(model.state_dict(), best_path)
                print(f"Best model save at {best_path}.")
                epsilon = 0
            else:
                epsilon += 1
            # if f1score > best_f1:
            #     best_f1 = f1score
            #     best_path = join(f'model/{name_dir}', 'step{%d}-lr{%.4f}-early{%d}-f1{%.2f}.pth' % (total_step, args.lr, args.ear, best_f1, ))
            #     torch.save(model.state_dict(), best_path)
            #     print(f"Best model save at {best_path}.")
            #     epsilon = 0
            # else:
            #     epsilon += 1
            # print(f'Training Done. Train f1: {f1score:.6f}. Best f1:{best_f1:.6f}')
            
        # if epsilon >= args.ear:
        #     break

if __name__ == '__main__':

    train()
