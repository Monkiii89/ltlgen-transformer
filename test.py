# -*- coding: utf-8 -*-

import os
import json
import time
import torch
import random
from argparse import ArgumentParser

from data_m import encode_prel_m1, encode_prel_m2
from model import RNN
from utils import LTL2SMV, nuXmv_ic3, Legal_ltl, pre2ltl, Variable_population, ltl2prefix

parser = ArgumentParser(description='Train Transformer for ltl')

parser.add_argument('--device', type=int, default=1, help="GPU number")
parser.add_argument('--tm', type=str, default="./model/test_model/first_step{92609}-lr{0.0001}-early{10000}-acc{0.93}.pth", help="trained model")
parser.add_argument('--tm2', type=str, default="./model/test_model/second_step{285636}-lr{0.0001}-early{10000}-acc{0.99}.pth", help="trained model")
parser.add_argument('--od', type=str, default='./data/amba_1w.json', help="generated data")
parser.add_argument('--n', type=int, default=4000, help="SAT or UNSAT formulae number")

parser.add_argument('--genm', type=int, default=1, help="select generation model, 1 for LTLGen, 2 for f1+random, 3 for random+f2 and 4 for random+random")
parser.add_argument('--d_model', type=int, default=512, help="transformerdecoder's d_model")
parser.add_argument('--n_head', type=int, default=8, help="transformerdecoder's n_head")
parser.add_argument('--embedding_size', type=int, default=512, help="nn.embedding")
parser.add_argument('--token_types1', type=int, default=10, help="op_u + op_s + 'P' ")
parser.add_argument('--token_types2', type=int, default=31, help="op_u + op_s + 'P' ")
parser.add_argument('--max_length', type=int, default=301, help="op_u + op_s + 'P' ")

args = parser.parse_args()
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

if args.genm ==1:
    flag_g = True
    flag_i = True
elif args.genm ==2:
    flag_g = True
    flag_i = False
elif args.genm ==3:
    flag_g = False
    flag_i = True
else:
    flag_g = False
    flag_i = False

def generator():

    model = RNN(
                d_model=args.d_model,
                n_head=args.n_head,
                embedding_size=args.embedding_size,
                token_types=args.token_types1,
                max_length=args.max_length,
                device=device).to(device)
    model2 = RNN(
                d_model=args.d_model,
                n_head=args.n_head,
                embedding_size=args.embedding_size,
                token_types=args.token_types2,
                max_length=args.max_length,
                device=device).to(device)
    assert args.tm is not None
    assert args.tm2 is not None
    print(f"Load first stage model at {args.tm}.\nLoad second stage model at {args.tm2}.")
    model.load_state_dict(torch.load(args.tm, map_location=device))
    model.eval()
    model2.load_state_dict(torch.load(args.tm2, map_location=device))
    model2.eval()
    tokens = ['U', '|', '&', 'F', 'G', 'X', '!', 'P']
    tokens2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
    pid = os.getpid()
    tmp_path = f'./temp/{pid}tempfile'
    UNSATs = []
    SATs = []
    prels = []
    preds = []
    probs = []
    # while len(UNSATs) < args.n or len(SATs) < args.n:
    while len(UNSATs) + len(SATs)< args.n:
        print(f'Len(SAT):{len(SATs)}\t\tLen(UNSAT):{len(UNSATs)}\n')
        prel = ""
        for j in range(140):
            if flag_g:
                if prel not in preds:
                    x, y_pos, x_mask = encode_prel_m1(prel, args.max_length)
                    x = x.reshape(1, -1).to(device)
                    y_pos = y_pos.reshape(1, -1).to(device)
                    x_mask = x_mask.reshape(1,-1).to(device)
                    preds.append(prel)
                    with torch.no_grad():
                        out = model.infv2(x, x_mask, y_pos)
                    out = out.to('cpu')
                    out = out[0][2:]
                    probs.append(out)
                    prel +=random.choices(tokens, list(out), k=1)[0]
                else:
                    out = probs[preds.index(prel)]
                    prel += random.choices(tokens, out, k=1)[0]   # 以out的概率对token进行采样
            else:
                prel += random.choice(tokens)
        ltl, prelr, _ = Legal_ltl(prel)
        prelrs = []
        probls = []
        # 合法公式变量填充
        t_n = prelr.count('P')
        indexes = [i for i, c in enumerate(prelr) if 'P' == c]
        for j in range(t_n):
            if flag_i:
                if prelr not in prelrs:
                    x, y_pos, x_mask = encode_prel_m2(prelr)
                    x = x.reshape(1, -1).to(device)
                    y_pos = y_pos.reshape(1, -1).to(device)
                    x_mask = x_mask.reshape(1,-1).to(device)
                    prelrs.append(prelr)
                    with torch.no_grad():
                        out = model2.infv2(x, x_mask, y_pos)
                    out = out.to('cpu')
                    out = out[0][9:-2]
                    probls.append(out)
                    idx = indexes[j]
                    if idx == len(prelr):
                        end = len(prelr)
                    else:
                        end = idx + 1
                    prelr = prelr[:end-1] + random.choices(tokens2, list(out), k=1)[0] + prelr[end:]
                else:    
                    out = probls[prelrs.index(prelr)]
                    prelr = prelr[:end-1] + random.choices(tokens2, out, k=1)[0] + prelr[end:]   # 以out的概率对token进行采样
            else:
                idx = indexes[j]
                if idx == len(prelr):
                    end = len(prelr)
                else:
                    end = idx + 1
                prelr = prelr[:end-1] + random.choice(tokens2) + prelr[end:]
        # ltl = Variable_population(ltl)
        # _, prelr = ltl2prefix(ltl)
        ltl, _ = pre2ltl(prelr)
        for j in range(len(tokens2)):
            prelr = prelr.replace(tokens2[j],f'P{j}')
            ltl = ltl.replace(tokens2[j],f'P{j}')
        # ltl, prelr, _ = Legal_ltl(prelr)
        if prelr not in prels:
            prels.append(prelr)
            LTL2SMV(ltl, smv_file=tmp_path)
            cur_time = time.time()
            if nuXmv_ic3(tmp_path):
                time_seconds = float(time.time()-cur_time)
                SATs.append({'inorder':ltl, 'preorder':prelr, 'issat':True, 'nuXmv_ic3_time':time_seconds})
            else:
                time_seconds = float(time.time()-cur_time)
                UNSATs.append({'inorder':ltl, 'preorder':prelr, 'issat':False, 'nuXmv_ic3_time':time_seconds})
    # with open("./data/ltl_SAT.json", "w") as write_f:
    #     json.dump(SATs, write_f, indent=4, ensure_ascii=False)
    with open(args.od, "w") as write_f:
        json.dump(UNSATs+SATs, write_f, indent=4, ensure_ascii=False)
if __name__ == '__main__':
    with torch.no_grad():
        generator()
        