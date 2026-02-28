import subprocess
import random
import torch
from ltlf import LTLfParser, LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext, LTLfOr, LTLfEventually, LTLfImplies, LTLfRelease


op_d = ['U', '|', '&']
op_s = ['F', 'G', '!', 'X']
NTO = 3600
VAR_NUM = 60

def F1_score(y_batch, y_pred):
    y_batch = torch.cat(y_batch)
    y_pred = torch.cat(y_pred)
    # print(y_batch, y_pred)
    yba = torch.tensor(y_batch, dtype=torch.bool).reshape(-1)
    ypre = torch.tensor(y_pred, dtype=torch.bool).reshape(-1)
    TP = torch.tensor(torch.logical_and(yba, ypre), dtype=torch.long).sum()+ 1e-7
    # TN = torch.tensor(torch.logical_and(~yba, ~ypre), dtype=torch.long).sum()
    yy = (y_pred-y_batch).reshape(-1)
    FP = torch.where(yy<1, 0,yy).sum() + 1e-7
    yy = (y_batch-y_pred).reshape(-1)
    FN = torch.where(yy<1, 0,yy).sum()+ 1e-7
    P = TP/ (TP+ FP)
    R = TP/ (TP+ FN)
    print(P, R)
    # print(TP, FP, FN, TN)
    F1 = 2*P*R/ (P + R)
    return F1


def preorder(f):
    if isinstance(f, LTLfAtomic):
        return f.s
    if isinstance(f, LTLfAnd) or isinstance(f, LTLfUntil) or isinstance(f, LTLfOr) or isinstance(f, LTLfRelease) or isinstance(f, LTLfImplies):
        result = f.operator_symbol
        # if len(f.formulas) > 2:
            # nf = deepcopy(f)
            # nf.formulas = nf.formulas[:-1]
        tstrs = []
        for f_index in range(len(f.formulas)):
            tstrs.append(preorder(f.formulas[f_index]))
                
            # tstrs = [preorder(nf), preorder(f.formulas[-1])]
        # else:
            # tstrs = [preorder(f.formulas[0]), preorder(f.formulas[1])]
        for i in tstrs:
            result += i
        return result
    if isinstance(f, LTLfNot) or isinstance(f, LTLfNext) or isinstance(f, LTLfAlways) or isinstance(f, LTLfEventually):
        result = f.operator_symbol
        result += preorder(f.f)
        return result
        
def ltl2prefix(ltl: str):
    parser = LTLfParser()
    formula = parser(ltl)
    return str(formula), preorder(formula)

def LTL2SMV(formulae:str, vocab=[f'P{i}' for i in range(VAR_NUM)], smv_file='./temp/1tempfile'):
    content = "MODULE main\nVAR\n"
    for v in vocab:
        content += v + ':boolean;\n'

    content += 'LTLSPEC!(\n' + formulae + ')'

    with open(smv_file, 'w') as smvfile:
        smvfile.write(content)

def nuXmv_ic3(temp_uc_path:str):
    #nuXmv_ic3求解
    cmd = './nuXmv -int'
    # -d is optional
    lines = [
        f"read_model -i {temp_uc_path}\n",
        "flatten_hierarchy\n",
        "encode_variables\n",
        "build_boolean_model\n",
        "check_ltlspec_ic3\n",
        "quit\n"
    ]
    stdin_sat_solver = ''.join(lines).encode()
        #         # endwith
        #     # endif
        #     cur_time = time.time()
    mytask = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
    try:
        mytask.stdin.write((stdin_sat_solver))
        result,err = mytask.communicate(timeout= NTO)
        if(result.decode().find("is false") != -1):
            return True
        else:
            return False
    except Exception as e:
        mytask.kill()
        return False
    
def Legal_ltl(ltl: str):
    if ltl =='':
        return 'P', 'P', -1
    if ltl[0] in op_d:
        op1, pop1, pos1 = Legal_ltl(ltl[1:])
        if pos1 ==-1:
            return f'({op1}) {ltl[0]} P', f'{ltl[0]}{pop1}P', -1
        if pos1+1 >= len(ltl):
            return f'({op1}) {ltl[0]} P', f'{ltl[0]}{pop1}P', -1
        op2, pop2, pos2 = Legal_ltl(ltl[pos1+1:])
        if pos2 == -1:
            return f'({op1}) {ltl[0]} ({op2})', f'{ltl[0]}{pop1}{pop2}', pos2
        return f'({op1}) {ltl[0]} ({op2})', f'{ltl[0]}{pop1}{pop2}', pos1+pos2 +1
    if ltl[0] in op_s:
        op1, pop1, pos1 = Legal_ltl(ltl[1:])
        if pos1 == -1:
            return f'{ltl[0]}({op1})', f'{ltl[0]}{pop1}', pos1
        return f'{ltl[0]}({op1})', f'{ltl[0]}{pop1}', pos1 + 1
    else:
        return ltl[0], ltl[0], 1

def pre2ltl(pre: str):
    if pre[0] in op_d:
        ltl1, pos1 = pre2ltl(pre[1:])
        ltl2, pos2 = pre2ltl(pre[pos1+1:])
        return f'({ltl1}) {pre[0]} ({ltl2})', pos1+pos2 +1
    if pre[0] in op_s:
        ltl1, pos1 = pre2ltl(pre[1:])
        return f'{pre[0]}({ltl1})', pos1 + 1
    else:
        return pre[0], 1

def Variable_population(ltl: str):
    atmoics = ltl.split('P')
    vars = [f'p{i}' for i in range(VAR_NUM)]
    formul = [atmoics[0]]
    for atmoic in atmoics[1:]:
        formul.append(random.choice(vars))
        formul.append(atmoic)
    return ''.join(formul)