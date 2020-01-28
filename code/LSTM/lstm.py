import numpy as np
from utils import sigmoid,softmax
from dataloader import idx_to_char,char_to_idx,dataset

class LSTM(object):

    def __init__(self,lr=1e-1,time_steps=25,len_of_vocab=25,mean=0.,std=0.01):
        self.lr = lr
        self.time_steps = time_steps
        self.len_of_vocab = len_of_vocab
        self.mean = mean
        self.std = std
        self.Wi,self.Wf,self.Wz,self.Wo,self.Wout = np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                            np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                            np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                            np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                            np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab))

        self.Ri,self.Rf,self.Rz,self.Ro = np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                        np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                        np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                        np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab))

        self.Pi,self.Pf,self.Po  = np.random.normal(self.mean,self.std,(self.len_of_vocab,1)),\
                    np.random.normal(self.mean,self.std,(self.len_of_vocab,1)),\
                    np.random.normal(self.mean,self.std,(self.len_of_vocab,1))

        self.bi,self.bo,self.bf,self.bz,self.bout = np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1))

        self.mWi,self.mWf,self.mWz,self.mWo,self.mWout = np.zeros_like(self.Wi),\
            np.zeros_like(self.Wf),\
            np.zeros_like(self.Wz),\
            np.zeros_like(self.Wo),\
            np.zeros_like(self.Wout)

        self.mRi,self.mRf,self.mRz,self.mRo = np.zeros_like(self.Ri),np.zeros_like(self.Rf),np.zeros_like(self.Rz),np.zeros_like(self.Ro)

        self.mPi,self.mPf,self.mPo = np.zeros_like(self.Pi),np.zeros_like(self.Pf),np.zeros_like(self.Po)

        self.mbi,self.mbo,self.mbf,self.mbz,self.mbout = np.zeros_like(self.bi),np.zeros_like(self.bo),np.zeros_like(self.bf),np.zeros_like(self.bz),np.zeros_like(self.bout)
        
        self.dWi,self.dWf,self.dWz,self.dWo,self.dWout = np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab))

        self.dRi,self.dRf,self.dRz,self.dRo = np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab))

        self.dPi,self.dPf,self.dPo  = np.zeros((self.len_of_vocab,1)),\
                    np.zeros((self.len_of_vocab,1)),\
                    np.zeros((self.len_of_vocab,1))

        self.dbi,self.dbo,self.dbf,self.dbz,self.dbout = np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1))
    
    def zero_grad(self):
        self.dWi,self.dWf,self.dWz,self.dWo,self.dWout = np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab))

        self.dRi,self.dRf,self.dRz,self.dRo = np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab))

        self.dPi,self.dPf,self.dPo  = np.zeros((self.len_of_vocab,1)),\
                    np.zeros((self.len_of_vocab,1)),\
                    np.zeros((self.len_of_vocab,1))

        self.dbi,self.dbo,self.dbf,self.dbz,self.dbout = np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1))

    def clip_grad(self,clip_val=1):
            for dparam in [self.dWi,self.dWf,self.dWz,self.dWo,self.dWout,self.dRi,self.dRf,self.dRz,self.dRo,self.dPi,self.dPo,self.dPf,self.dbi,self.dbo,self.dbf,self.dbz,self.dbout]:
                np.clip(dparam,-clip_val,clip_val,out=dparam)
        
    def step(self):
        for params,dparam,mparam in zip([self.Wi,self.Wf,self.Wz,self.Wo,self.Wout,self.Ri,self.Rf,self.Rz,self.Ro,self.Pi,self.Po,self.Pf,self.bi,self.bo,self.bf,self.bz,self.bout],\
		[self.dWi,self.dWf,self.dWz,self.dWo,self.dWout,self.dRi,self.dRf,self.dRz,self.dRo,self.dPi,self.dPo,self.dPf,self.dbi,self.dbo,self.dbf,self.dbz,self.dbout],\
		[self.mWi,self.mWf,self.mWz,self.mWo,self.mWout,self.mRi,self.mRf,self.mRz,self.mRo,self.mPi,self.mPo,self.mPf,self.mbi,self.mbo,self.mbf,self.mbz,self.mbout]):
            mparam += dparam*dparam
            params += -self.lr*dparam/np.sqrt(mparam+1e-8)

    def sample(self,h_prev,c_prev,num_char):
        hs = np.copy(h_prev)
        cs = np.copy(c_prev)
        x = np.zeros((self.len_of_vocab,1))
        x[np.random.randint(0,self.len_of_vocab),0] = 1
        idxs = []
        for _ in range(num_char):

            I = np.dot(self.Wi,x) + np.dot(self.Ri,hs) + self.Pi*cs + self.bi
            i_gate = sigmoid(I)

            F = np.dot(self.Wf,x) + np.dot(self.Rf,hs) + self.Pf*cs + self.bo
            f_gate = sigmoid(F)

            Z = np.dot(self.Wz,x) + np.dot(self.Rz,hs) + self.bz
            z = np.tanh(Z)

            cs = i_gate*z + f_gate*cs

            O = np.dot(self.Wo,x) + np.dot(self.Ro,hs) + self.Po*cs +self.bo
            o_gate = sigmoid(O)

            hs = o_gate * np.tanh(cs)

            out = np.dot(self.Wout,hs) + self.bout

            p = softmax(out)
            idx = np.random.choice(self.len_of_vocab,1,p=p.ravel())[0]
            x = np.zeros((self.len_of_vocab,1))
            x[idx,0] = 1
            idxs.append(idx)

        print(''.join(idx_to_char[c] for c in idxs))

    #forward_backward_pass
    def forward_backward_pass(self,input,output,h_prev,c_prev):
        hs={}
        cs={}
        i_gate={}
        f_gate={}
        o_gate={}
        z ={}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        p = {}
        loss = 0
        for t in range(self.time_steps):
            x = np.zeros((self.len_of_vocab,1))
            x[input[t],0] = 1

            I = np.dot(self.Wi,x) + np.dot(self.Ri,hs[t-1]) + self.Pi*cs[t-1] + self.bi
            i_gate[t] = sigmoid(I)

            F = np.dot(self.Wf,x) + np.dot(self.Rf,hs[t-1]) + self.Pf*cs[t-1] + self.bo
            f_gate[t] = sigmoid(F)

            Z = np.dot(self.Wz,x) + np.dot(self.Rz,hs[t-1]) + self.bz
            z[t] = np.tanh(Z)

            cs[t] = i_gate[t]*z[t] + f_gate[t]*cs[t-1]

            O = np.dot(self.Wo,x) + np.dot(self.Ro,hs[t-1]) + self.Po*cs[t] +self.bo
            o_gate[t] = sigmoid(O)

            hs[t] = o_gate[t] * np.tanh(cs[t])

            out = np.dot(self.Wout,hs[t]) + self.bout

            p[t] = softmax(out)

            loss += -np.log(p[t][output[t],0])

        #Backward pass
        dht_z = np.zeros((self.len_of_vocab,1))
        dht_f = np.zeros((self.len_of_vocab,1))
        dht_o = np.zeros((self.len_of_vocab,1))
        dht_i = np.zeros((self.len_of_vocab,1))

        dct_cs = np.zeros((self.len_of_vocab,1))
        dct_f = np.zeros((self.len_of_vocab,1))
        dct_o = np.zeros((self.len_of_vocab,1))
        dct_i = np.zeros((self.len_of_vocab,1))

        for t in reversed(range(self.time_steps)):
            x = np.zeros((self.len_of_vocab,1))
            x[input[t],0] = 1

            dout = np.copy(p[t])
            dout[output[t],0] -= 1
            self.dWout += np.dot(dout,hs[t].T)
            dht = np.dot(self.Wout.T,dout) + dht_z + dht_f + dht_o + dht_i
            self.dbout += dout
    
            dog = np.tanh(cs[t])*dht
            dog_ = o_gate[t]*(1-o_gate[t])*dog
            self.dWo += np.dot(dog_,x.T)
            self.dRo += np.dot(dog_,hs[t-1].T)
            dht_o = np.dot(self.Ro.T,dog_)
            self.dPo += cs[t]*dog_
            dct_o = self.Po * dog_
            self.dbo += dog_

            dct = (1-np.tanh(cs[t])*np.tanh(cs[t]))*o_gate[t]*dht + dct_cs + dct_f + dct_o + dct_i
            dig = z[t] * dct
            dz  = i_gate[t] * dct
            dfg = cs[t-1] * dct
            dct_cs = f_gate[t] * dct

            dz_ = (1-z[t]*z[t])*dz
            self.dWz += np.dot(dz_,x.T)
            self.dRz += np.dot(dz_,hs[t-1].T)
            dht_z = np.dot(self.Rz.T,dz_)
            self.dbz += dz_

            dfg_ = f_gate[t]*(1-f_gate[t])*dfg
            self.dWf += np.dot(dfg_,x.T)
            self.dRf += np.dot(dfg_,hs[t-1].T)
            dht_f = np.dot(self.Rf.T,dfg_)
            self.dPf += cs[t-1] * dfg_
            dct_f  = self.Pf * dfg_
            self.dbf += dfg_

            dig_ = i_gate[t]*(1-i_gate[t])*dig
            self.dWi += np.dot(dig_,x.T)
            self.dRi += np.dot(dig_,hs[t-1].T)
            dht_i = np.dot(self.Ri.T,dig_)
            self.dPi += cs[t-1]*dig_
            dct_i = self.Pi * dig_
            self.dbi += dig_

        return loss,self.dWi,self.dWf,self.dWz,self.dWo,self.dWout,self.dRi,self.dRf,self.dRz,self.dRo,self.dPi,self.dPo,self.dPf,self.dbi,self.dbo,self.dbf,self.dbz,self.dbout,hs[self.time_steps-1],cs[self.time_steps-1]
