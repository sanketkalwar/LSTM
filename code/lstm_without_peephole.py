import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.ion()
dataset = open('../data/input.txt','r').read()
#dataset = open('../data/code.txt','r').read()
len_of_dataset = len(dataset)
print('len of dataset:',len_of_dataset)

vocab = set(dataset)
len_of_vocab = len(vocab)
print('len of vocab:',len_of_vocab)

char_to_idx = {char:idx for idx,char in enumerate(vocab)}
idx_to_char = {idx:char for idx,char in enumerate(vocab)}
print('char_to_idx:',char_to_idx)
print('idx_to_char:',idx_to_char)

start_ptr = 0
lr = 1e-1
time_step = 25
mean =0.0
std =0.01

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

Wi,Ri,bi = np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),np.random.normal(mean,std,(len_of_vocab,1))
Wo,Ro,bo = np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),np.random.normal(mean,std,(len_of_vocab,1))
Wf,Rf,bf = np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),np.random.normal(mean,std,(len_of_vocab,1))
Wz,Rz,bz = np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),np.random.normal(mean,std,(len_of_vocab,1))
Wy,by = np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),np.zeros((len_of_vocab,1))

mWi,mRi,mbi = np.zeros_like(Wi),np.zeros_like(Ri),np.zeros_like(bi)
mWo,mRo,mbo = np.zeros_like(Wo),np.zeros_like(Ro),np.zeros_like(bo)
mWf,mRf,mbf = np.zeros_like(Wf),np.zeros_like(Rf),np.zeros_like(bf)
mWz,mRz,mbz = np.zeros_like(Wz),np.zeros_like(Rz),np.zeros_like(bz)
mWy,mby = np.zeros_like(Wy),np.zeros_like(by)

def sample(y_p,c_p):
    idx = []
    x = np.zeros((len_of_vocab,1))
    x[10,0] = np.random.randint(0,len_of_vocab)
    for t in range(200):
        I = np.dot(Wi,x)+np.dot(Ri,y_p)+bi
        i_g = sigmoid(I)

        O = np.dot(Wo,x)+np.dot(Ro,y_p)+bo
        o_g = sigmoid(O)

        F = np.dot(Wf,x) + np.dot(Rf,y_p)+bf
        f_g = sigmoid(F)

        Z = np.dot(Wz,x) + np.dot(Rz,y_p)+bz
        z_g = np.tanh(Z)

        c_p = i_g*z_g + f_g *c_p
        y_p = o_g * np.tanh(c_p)
        os = np.dot(Wy,y_p)+by
        p = softmax(os)
        id = np.random.choice(len_of_vocab,1,p=p.ravel())[0]
        idx.append(id)
        x = np.zeros((len_of_vocab,1))
        x[id,0]=1
    print(''.join([idx_to_char[c] for c in idx]))

def forward_backward_pass(i,o,y_p,c_p):
    cs = {}
    ys = {}
    i_g = {}
    o_g = {}
    f_g = {}
    z_g = {}
    os = {}
    cs[-1] = np.copy(c_p)
    ys[-1] = np.copy(y_p)
    p = {}
    loss = 0
    for t in range(time_step):
        x = np.zeros((len_of_vocab,1))
        x[i[t],0] = 1

        I = np.dot(Wi,x)+np.dot(Ri,ys[t-1])+bi
        i_g[t] = sigmoid(I)

        O = np.dot(Wo,x)+np.dot(Ro,ys[t-1])+bo
        o_g[t] = sigmoid(O)

        F = np.dot(Wf,x) + np.dot(Rf,ys[t-1])+bf
        f_g[t] = sigmoid(F)

        Z = np.dot(Wz,x) + np.dot(Rz,ys[t-1])+bz
        z_g[t] = np.tanh(Z)

        cs[t] = i_g[t]*z_g[t] + f_g[t] *cs[t-1]
        ys[t] = o_g[t] * np.tanh(cs[t])
        os[t] = np.dot(Wy,ys[t])+by
        p[t] = softmax(os[t])
        loss += -np.log(p[t][o[t],0])
    
    dWi,dRi,dbi = np.zeros_like(Wi),np.zeros_like(Ri),np.zeros_like(bi)
    dWo,dRo,dbo = np.zeros_like(Wo),np.zeros_like(Ro),np.zeros_like(bo)
    dWf,dRf,dbf = np.zeros_like(Wf),np.zeros_like(Rf),np.zeros_like(bf)
    dWz,dRz,dbz = np.zeros_like(Wz),np.zeros_like(Rz),np.zeros_like(bz)
    dWy,dby     = np.zeros_like(Wy),np.zeros_like(by)
    dy_z,dy_f,dy_o,dy_i = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))
    dcs_c = np.zeros((len_of_vocab,1))
    for t in reversed(range(time_step)):
        x = np.zeros((len_of_vocab,1))
        x[i[t],0] = 1
        do = np.copy(p[t])
        do[o[t],0] -= 1

        
        dWy += np.outer(do,ys[t])
        dby += do
        dy = np.dot(Wy,do)
        dy = dy + dy_z + dy_f + dy_i + dy_o
        dcs = o_g[t] * (1-np.tanh(cs[t])*np.tanh(cs[t]))*dy + dcs_c
        dcs_c = f_g[t]*dcs

        dig = z_g[t]*dcs
        dog = np.tanh(cs[t])*dy
        dzg = i_g[t]*dcs
        dfg = cs[t-1]*dcs

        dzg_ = (1-z_g[t]*z_g[t])*dzg
        dWz += np.outer(dzg_,x)
        dRz += np.outer(dzg_,ys[t-1])
        dbz += dzg_
        dy_z = np.dot(Rz.T,dzg_)

        dfg_ = f_g[t] * (1-f_g[t])*dfg
        dWf += np.outer(dfg_,x)
        dRf += np.outer(dfg_,ys[t-1])
        dy_f = np.dot(Rf.T,dfg_)
        dbf += dfg_

        dog_ = o_g[t]*(1-o_g[t])*dog
        dWo += np.outer(dog_,x)
        dRo += np.outer(dog_,ys[t-1])
        dy_o = np.dot(Ro.T,dog_)
        dbo += dog_

        dig_ = i_g[t]*(1-i_g[t])*dig
        dWi += np.outer(dig_,x)
        dRi += np.outer(dig_,ys[t-1])
        dy_i = np.dot(Ri.T,dig_)
        dbi += dig_

    for param in [dWi,dRi,dbi,dWo,dRo,dbo,dWf,dRf,dbf,dWz,dRz,dbz]:
        np.clip(param,-1,1,out=param)

    
    return loss,dWi,dRi,dbi,dWo,dRo,dbo,dWf,dRf,dbf,dWz,dRz,dbz,dWy,dby,ys[time_step-1],cs[time_step-1]








y_prev,c_prev = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))
n = 0
x=[]
y=[]
smooth_loss = 200
while True:
    if start_ptr+time_step>len_of_dataset:
        start_ptr = 0
        y_prev = np.zeros((len_of_vocab,1))
    else:
        input = [char_to_idx[c] for c in dataset[start_ptr:start_ptr+time_step]]
        output = [char_to_idx[c] for c in dataset[start_ptr+1:start_ptr+time_step+1]]
        loss,dWi,dRi,dbi,dWo,dRo,dbo,dWf,dRf,dbf,dWz,dRz,dbz,dWy,dby,y_prev,c_prev=forward_backward_pass(i=input,o=output,y_p=y_prev,c_p=c_prev)
        for params,dparams,mparams in zip([Wi,Ri,bi,Wo,Ro,bo,Wf,Rf,bf,Wz,Rz,bz,Wy,by],\
            [dWi,dRi,dbi,dWo,dRo,dbo,dWf,dRf,dbf,dWz,dRz,dbz,dWy,dby],[mWi,mRi,mbi,mWo,mRo,mbo,mWf,mRf,mbf,mWz,mRz,mbz,mWy,mby]):
            mparams += dparams*dparams
            params += -lr*dparams/np.sqrt(mparams+1e-8)
        smooth_loss = (0.999*smooth_loss)+(0.001*loss)
        x.append(n)
        y.append(smooth_loss)
        
        if n%1000 == 0:
            print('smooth_loss:',loss)
            sample(y_p=y_prev,c_p=c_prev)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.plot(x,y,color='r')
            plt.pause(1e-9)

    n+=1
    start_ptr += time_step
