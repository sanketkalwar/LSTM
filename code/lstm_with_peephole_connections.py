import numpy as np
import matplotlib.pyplot as plt

plt.ion()
#load dataset 
dataset = open('../data/input.txt','r').read()
len_of_dataset = len(dataset)
print('length of dataset:',len_of_dataset)

vocab = set(dataset)
len_of_vocab = len(vocab)
print('length of vocab:',len_of_vocab)

char_to_idx = {char:idx for idx,char in enumerate(vocab)}
print('char_to_idx:',char_to_idx)

idx_to_char = {idx:char for idx,char in enumerate(vocab)}
print('idx_to_char:',idx_to_char)

#hyperparameter initialization
lr = 1e-1
time_steps = 25
start_ptr = 0
mean = 0.
std = 0.01
epoches = 10000

Wi,Wf,Wz,Wo,Wout = np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),\
					np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),\
					np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),\
					np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),\
					np.random.normal(mean,std,(len_of_vocab,len_of_vocab))

Ri,Rf,Rz,Ro = np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),\
				np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),\
				np.random.normal(mean,std,(len_of_vocab,len_of_vocab)),\
				np.random.normal(mean,std,(len_of_vocab,len_of_vocab))

Pi,Pf,Po  = np.random.normal(mean,std,(len_of_vocab,1)),\
			np.random.normal(mean,std,(len_of_vocab,1)),\
			np.random.normal(mean,std,(len_of_vocab,1))

bi,bo,bf,bz,bout = np.zeros((len_of_vocab,1)),\
					np.zeros((len_of_vocab,1)),\
					np.zeros((len_of_vocab,1)),\
					np.zeros((len_of_vocab,1)),\
					np.zeros((len_of_vocab,1))

mWi,mWf,mWz,mWo,mWout = np.zeros_like(Wi),np.zeros_like(Wf),np.zeros_like(Wz),np.zeros_like(Wo),np.zeros_like(Wout)

mRi,mRf,mRz,mRo = np.zeros_like(Ri),np.zeros_like(Rf),np.zeros_like(Rz),np.zeros_like(Ro)

mPi,mPf,mPo = np.zeros_like(Pi),np.zeros_like(Pf),np.zeros_like(Po)

mbi,mbo,mbf,mbz,mbout = np.zeros_like(bi),np.zeros_like(bo),np.zeros_like(bf),np.zeros_like(bz),np.zeros_like(bout)

#utility functions
def sigmoid(x):
	return 1/(1+np.exp(-x))

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))


def sample(h_prev,c_prev,num_char):
	hs = np.copy(h_prev)
	cs = np.copy(c_prev)
	x = np.zeros((len_of_vocab,1))
	x[np.random.randint(0,len_of_vocab),0] = 1
	idxs = []
	for t in range(num_char):

		I = np.dot(Wi,x) + np.dot(Ri,hs) + Pi*cs + bi
		i_gate = sigmoid(I)

		F = np.dot(Wf,x) + np.dot(Rf,hs) + Pf*cs + bf
		f_gate = sigmoid(F)

		Z = np.dot(Wz,x) + np.dot(Rz,hs) + bz
		z = np.tanh(Z)

		cs = i_gate*z + f_gate*cs

		O = np.dot(Wo,x) + np.dot(Ro,hs) + Po*cs +bo
		o_gate = sigmoid(O)

		hs = o_gate * np.tanh(cs)

		out = np.dot(Wout,hs) + bout

		p = softmax(out)
		idx = np.random.choice(len_of_vocab,1,p=p.ravel())[0]
		x = np.zeros((len_of_vocab,1))
		x[idx,0] = 1
		idxs.append(idx)

	print(''.join(idx_to_char[c] for c in idxs))

#forward_backward_pass
def forward_backward_pass(input,output,h_prev,c_prev):
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
	for t in range(time_steps):
		x = np.zeros((len_of_vocab,1))
		x[input[t],0] = 1

		I = np.dot(Wi,x) + np.dot(Ri,hs[t-1]) + Pi*cs[t-1] + bi
		i_gate[t] = sigmoid(I)

		F = np.dot(Wf,x) + np.dot(Rf,hs[t-1]) + Pf*cs[t-1] + bf
		f_gate[t] = sigmoid(F)

		Z = np.dot(Wz,x) + np.dot(Rz,hs[t-1]) + bz
		z[t] = np.tanh(Z)

		cs[t] = i_gate[t]*z[t] + f_gate[t]*cs[t-1]

		O = np.dot(Wo,x) + np.dot(Ro,hs[t-1]) + Po*cs[t] +bo
		o_gate[t] = sigmoid(O)

		hs[t] = o_gate[t] * np.tanh(cs[t])

		out = np.dot(Wout,hs[t]) + bout

		p[t] = softmax(out)

		loss += -np.log(p[t][output[t],0])
	
	dWi,dWf,dWz,dWo,dWout = np.zeros_like(Wi),np.zeros_like(Wf),np.zeros_like(Wz),np.zeros_like(Wo),np.zeros_like(Wout)
	dRi,dRf,dRz,dRo = np.zeros_like(Ri),np.zeros_like(Rf),np.zeros_like(Rz),np.zeros_like(Ro)
	dPi,dPo,dPf = np.zeros_like(Pi),np.zeros_like(Po),np.zeros_like(Pf)
	dbi,dbo,dbf,dbz,dbout = np.zeros_like(bi),np.zeros_like(bo),np.zeros_like(bf),np.zeros_like(bz),np.zeros_like(bout)

	#Backward pass
	dht_z = np.zeros((len_of_vocab,1))
	dht_f = np.zeros((len_of_vocab,1))
	dht_o = np.zeros((len_of_vocab,1))
	dht_i = np.zeros((len_of_vocab,1))

	dct_cs = np.zeros((len_of_vocab,1))
	dct_f = np.zeros((len_of_vocab,1))
	dct_o = np.zeros((len_of_vocab,1))
	dct_i = np.zeros((len_of_vocab,1))

	for t in reversed(range(time_steps)):
		x = np.zeros((len_of_vocab,1))
		x[input[t],0] = 1

		dout = np.copy(p[t])
		dout[output[t],0] -= 1
		dWout += np.dot(dout,hs[t].T)
		dht = np.dot(Wout.T,dout) + dht_z + dht_f + dht_o + dht_i
		dbout += dout
		
		dog = np.tanh(cs[t])*dht
		dog_ = o_gate[t]*(1-o_gate[t])*dog
		dWo += np.dot(dog_,x.T)
		dRo += np.dot(dog_,hs[t-1].T)
		dht_o = np.dot(Ro.T,dog_)
		dPo += cs[t]*dog_
		dct_o = Po * dog_
		dbo += dog_

		dct = (1-np.tanh(cs[t])*np.tanh(cs[t]))*o_gate[t]*dht + dct_cs + dct_f + dct_o + dct_i
		dig = z[t] * dct
		dz  = i_gate[t] * dct
		dfg = cs[t-1] * dct
		dct_cs = f_gate[t] * dct

		dz_ = (1-z[t]*z[t])*dz
		dWz += np.dot(dz_,x.T)
		dRz += np.dot(dz_,hs[t-1].T)
		dht_z = np.dot(Rz.T,dz_)
		dbz += dz_

		dfg_ = f_gate[t]*(1-f_gate[t])*dfg
		dWf += np.dot(dfg_,x.T)
		dRf += np.dot(dfg_,hs[t-1].T)
		dht_f = np.dot(Rf.T,dfg_)
		dPf += cs[t-1] * dfg_
		dct_f  = Pf * dfg_
		dbf += dfg_

		dig_ = i_gate[t]*(1-i_gate[t])*dig
		dWi += np.dot(dig_,x.T)
		dRi += np.dot(dig_,hs[t-1].T)
		dht_i = np.dot(Ri.T,dig_)
		dPi += cs[t-1]*dig_
		dct_i = Pi * dig_
		dbi += dig_

	for dparam in [dWi,dWf,dWz,dWo,dWout,dRi,dRf,dRz,dRo,dPi,dPo,dPf,dbi,dbo,dbf,dbz,dbout]:
		np.clip(dparam,-1,1,out=dparam)

	return loss,dWi,dWf,dWz,dWo,dWout,dRi,dRf,dRz,dRo,dPi,dPo,dPf,dbi,dbo,dbf,dbz,dbout,hs[time_steps-1],cs[time_steps-1]





x=[]
y=[]
n = 0
smooth_loss = -np.log(1/len_of_vocab)*time_steps
h_prev,c_prev = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))
while n<=10000:
	if start_ptr+time_steps > len_of_dataset:
		start_ptr = 0
		h_prev = np.zeros((len_of_vocab,1))
	else:
		input = [char_to_idx[c] for c in dataset[start_ptr:start_ptr+time_steps]]
		output = [char_to_idx[c] for c in dataset[start_ptr+1:start_ptr+time_steps+1]]
		loss,dWi,dWf,dWz,dWo,dWout,dRi,dRf,dRz,dRo,dPi,dPo,dPf,dbi,dbo,dbf,dbz,dbout,h_prev,c_prev=forward_backward_pass\
																	(input=input,output=output,h_prev=h_prev,c_prev=c_prev)

		smooth_loss = (0.999*smooth_loss)+(0.001*loss)
		x.append(n)
		y.append(smooth_loss)
		if n%epoches==0:
			print('--------------------------------------------')
			print('iter:',n)
			print('smooth_loss:',smooth_loss)
			sample(h_prev=h_prev,c_prev=c_prev,num_char=300)
			print('--------------------------------------------')
			plt.ylabel('Loss')
			plt.xlabel('Epoch')
			plt.plot(x,y,color='r')
			plt.pause(1e-9)

		for params,dparam,mparam in zip([Wi,Wf,Wz,Wo,Wout,Ri,Rf,Rz,Ro,Pi,Po,Pf,bi,bo,bf,bz,bout],\
			[dWi,dWf,dWz,dWo,dWout,dRi,dRf,dRz,dRo,dPi,dPo,dPf,dbi,dbo,dbf,dbz,dbout],\
			[mWi,mWf,mWz,mWo,mWout,mRi,mRf,mRz,mRo,mPi,mPo,mPf,mbi,mbo,mbf,mbz,mbout]):
			mparam += dparam*dparam
			params += -lr*dparam/np.sqrt(mparam+1e-8)
	n+=1
	start_ptr += time_steps

plt.savefig('../Performance/lstm_with_peephole_connection.png')