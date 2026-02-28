import re
import torch
import random
from collections import Counter
import pandas as pd
from torch.utils.data import Dataset

token = ['U', '|', '&', 'F', 'G', 'X', '!', 'P']
tokens1 = ['U', '|', '&', 'F', 'G', 'X', '!', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'P', 'M']

def encode_prel_m1(prel: str, max_length=301):
    leng = len(prel)
    x = [1] * max_length
    x[0] = 0
    for i in range(leng):
        if prel[i] not in token:
            x[i + 1] = len(token) + 2 - 1
        else:
            x[i + 1] = token.index(prel[i]) + 2
    x_mask = [False] * (leng + 1) + [True] * (max_length - leng - 1)
    y_pos = [0] * max_length
    y_pos[leng + 1] = 1
    return torch.tensor(x, dtype=torch.long), torch.tensor(y_pos, dtype=torch.long), torch.tensor(x_mask, dtype=torch.bool)

def encode_prel_m2(prel: str):
    leng = len(prel)
    index = prel.find('P')
    indexes = [i for i, c in enumerate(prel) if 'P' == c]
    x = [1] * (leng + 1)
    x[0] = 0
    for i in range(leng):
        x[i + 1] = tokens1.index(prel[i]) + 2
    x[index + 1] = 30
    assert indexes[0] == index
    x_mask = [False] * (leng + 1)
    for i in indexes[1:]:
        x_mask[i + 1] = True
    y_pos = [0] * (leng + 1)
    y_pos[index + 1] = 1

    return torch.tensor(x, dtype=torch.long), torch.tensor(y_pos, dtype=torch.long), torch.tensor(x_mask, dtype=torch.bool)

class TrainSet_instantiation(Dataset):
    def __init__(self, path, max_length=300, step=100):
        
        df = pd.read_json(path)
        data = [df.loc[i]['preorder'] for i in range(len(df))]
        atoms = [re.findall("\||U|&|!|X|F|G|p\d+", pre) for pre in data]
        datas = []
        for i in range(len(atoms)):
            if len(atoms[i]) <max_length:
                datas.append(atoms[i])
                continue
            begin = torch.arange(0, len(atoms[i])-max_length, step)
            end = torch.arange(max_length, len(atoms[i]), step)
            for j in range(len(begin)):
                datas.append(atoms[i][begin[j]:end[j]])
        data_type = []
        y_poss = []
        y_pos_ls = []
        for data in datas:
            pre = ''.join(data)
            atoms = Counter(re.findall('p\d+', pre))
            atoms = sorted(atoms.items(), key=lambda x:x[1], reverse=True) # atoms:[('p1',2), ('p2',1)]
            atoms = [[*el] for el in atoms] 
            atoms = [atoms[i][0] for i in range(len(atoms))] # atoms:['p1','p2']
            types = []
            y_pos = []
            for i in range(len(data)):
                if data[i] in atoms:
                    if(atoms.index(data[i]) + 9 <30):
                        ind = atoms.index(data[i]) + 9
                    else:
                        ind = 29
                    y_pos.append(i + 1)
                    types.append(ind)
                else: # 符号变量的索引
                    types.append(tokens1.index(data[i])+2)
            y_pos_ls.append(torch.tensor(len(y_pos))) # 每个序列中命题变量的数量
            y_poss.append(torch.tensor(y_pos))
            data_type.append(torch.tensor([0]+types))
        
        self.data_type = data_type
        leng_list = torch.tensor([len(i) for i in self.data_type])
        self.max_length = max(leng_list)


        self.data_type = torch.nn.utils.rnn.pad_sequence(data_type, batch_first=True, padding_value=1)
        x = [self.data_type[i].repeat(len(y_poss[i]),1) for i in range(len(y_poss))]
        x = torch.cat(x, dim=0)  # x的维度：(total_samples, seq_len)
        y_m = torch.cat(y_poss).reshape(-1).unsqueeze(1) # 所有命题变量位置
        y_m1 = torch.arange(0,len(x),1).unsqueeze(1) # 样本索引
        y_m= torch.tensor(y_m, dtype=torch.long) 
        y_m1= torch.tensor(y_m1, dtype=torch.long)
        y = x[y_m1, y_m].reshape(-1) # y是x中所有命题变量
        x[y_m1, y_m] = torch.tensor(30) # 将x对应位置的值设为 30
        
        x_m = torch.cat([leng_list[i].repeat(len(y_poss[i]),1) for i in range(len(y_poss))], dim=0)
        x_pos = [torch.triu(y_poss[i].repeat(len(y_poss[i]),1),diagonal=1) for i in range(len(y_poss))]
        ################## torch.triu 的使用
        x_mask = torch.zeros((x.size(1)))
        x_mask = [x_mask[:x_m[i]] for i in range(len(x_m))] # 截取有效长度部分
        x_mask = torch.nn.utils.rnn.pad_sequence(x_mask, batch_first=True, padding_value=1)
        count = 0
        for i in range(len(y_pos_ls)):
            for j in range(y_pos_ls[i]):
                x_m_p = torch.nonzero(x_pos[i][j])  # 第i个样本的第j个命题的索引
                x_mask[count,x_pos[i][j][x_m_p]]=1  ################################
                count += 1
        y_pos = torch.zeros(x.shape)
        y_pos[torch.arange(x.size(0)).unsqueeze(-1),y_m] = torch.tensor(1, dtype=torch.float)
        self.y = y
        self.y_pos = y_pos
        self.x = x
        self.x_mask = torch.tensor(x_mask, dtype=bool)
        self.length = len(self.x)

        
    def __getitem__(self, item):
        return self.x[item], self.y_pos[item], self.y[item], self.x_mask[item]

    def __len__(self):
        return self.length
    
class TrainSet_generation(Dataset):
    def __init__(self, path = './data/debug.json', max_length = 300, step=200):
        
        df = pd.read_json(path)
        self.data = [re.sub('p\d+', 'P', df.loc[i]['preorder']) for i in range(len(df))]

        datas = []
        for i in range(len(self.data)):
            if len(self.data[i]) <max_length:
                datas.append(self.data[i])
                continue
            begin = torch.arange(0, len(self.data[i])-max_length, step)
            end = torch.arange(max_length, len(self.data[i]), step)
            for j in range(len(begin)):
                datas.append(self.data[i][begin[j]:end[j]])
        self.data = datas

        

        data_type = []
        leng_list = [len(i)+1 for i in self.data]


        self.max_length = max(leng_list)


        for i in range(len(self.data)):
            leng = leng_list[i]-1
            types = [token.index(self.data[i][j]) +2 for j in range(leng)]
            types = [0] + types
            data_type.append(torch.tensor(types))
        self.data_type = torch.nn.utils.rnn.pad_sequence(data_type,batch_first=True, padding_value=1)
        
        

        mask_list = [torch.tensor(torch.triu(torch.ones(i,self.max_length), diagonal=0)[1:,:], dtype=torch.bool) for i in leng_list]
        self.x_mask = torch.cat(mask_list)
        
        self.x = torch.cat([torch.masked_fill(input=self.data_type[i], mask=mask_list[i], value=1) for i in range(len(mask_list))])
        
        self.y_pos = torch.cat([torch.eye(self.max_length)[1:i] for i in leng_list])

        
        leng_list = torch.tensor(leng_list)
        y_mask = torch.tensor(self.y_pos, dtype=torch.bool)

        self.y = torch.cat([torch.masked_fill(input=self.data_type[i], mask=~y_mask[torch.sum(leng_list[:i])-i:torch.sum(leng_list[:i+1])-i-1], value=0) for i in range(len(leng_list))])
        self.y = torch.masked_select(self.y, self.y>0)

        self.length = len(self.x)

        self.y = torch.tensor(self.y, dtype=torch.long)


    def __getitem__(self, item):
        return self.x[item], self.y_pos[item], self.y[item], self.x_mask[item]

    def __len__(self):
        return self.length

if __name__ == '__main__':
    TG = TrainSet_generation()
    for i in range(TG.__len__()):
        print(TG.__getitem__(i))