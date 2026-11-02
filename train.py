import gzip, numpy as np, os
# 사용법: python train.py   (MNIST 다운로드 -> 학습 -> model/mnist_mlp.npz 저장)
import urllib.request
BASE=os.path.dirname(os.path.abspath(__file__))
D=os.path.join(BASE,"data"); out=os.path.join(BASE,"model","mnist_mlp.npz")
URL="https://raw.githubusercontent.com/fgnt/mnist/master/"
os.makedirs(D,exist_ok=True); os.makedirs(os.path.dirname(out),exist_ok=True)
for n in ["train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz","t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz"]:
    p=os.path.join(D,n)
    if not os.path.exists(p):
        print("download",n); urllib.request.urlretrieve(URL+n,p)
def L(n,off): return np.frombuffer(gzip.open(os.path.join(D,n)).read(),np.uint8,offset=off)
Xtr=L('train-images-idx3-ubyte.gz',16).reshape(-1,28,28).astype(np.float32)/255; ytr=L('train-labels-idx1-ubyte.gz',8)
Xte=L('t10k-images-idx3-ubyte.gz',16).reshape(-1,784).astype(np.float32)/255; yte=L('t10k-labels-idx1-ubyte.gz',8)
rng=np.random.default_rng(0)
def aug(b):
    o=np.zeros_like(b)
    for i in range(len(b)):
        dx,dy=rng.integers(-2,3,2); o[i]=np.roll(np.roll(b[i],dy,0),dx,1)
    return o.reshape(len(b),-1)
sizes=[784,512,256,10]
W=[rng.normal(0,np.sqrt(2/a),(a,b)).astype(np.float32) for a,b in zip(sizes,sizes[1:])]
B=[np.zeros(b,np.float32) for b in sizes[1:]]
P=W+B; m=[np.zeros_like(p) for p in P]; v=[np.zeros_like(p) for p in P]; t=0
def fwd(x):
    hs=[x]
    for i in range(3):
        z=hs[-1]@W[i]+B[i]; hs.append(np.maximum(z,0) if i<2 else z)
    return hs
for ep in range(12):
    lr=1e-3*(0.8**ep); idx=rng.permutation(len(Xtr))
    for s in range(0,len(idx),128):
        j=idx[s:s+128]; x=aug(Xtr[j]); y=ytr[j]; hs=fwd(x)
        z=hs[-1]; z=z-z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
        g=p; g[np.arange(len(y)),y]-=1; g/=len(y)
        gW=[None]*3; gB=[None]*3
        for i in (2,1,0):
            gW[i]=hs[i].T@g+1e-5*W[i]; gB[i]=g.sum(0)
            if i: g=(g@W[i].T)*(hs[i]>0)
        t+=1
        for k,(p_,gr) in enumerate(zip(P,gW+gB)):
            m[k]=0.9*m[k]+0.1*gr; v[k]=0.999*v[k]+0.001*gr*gr
            p_-=lr*(m[k]/(1-0.9**t))/(np.sqrt(v[k]/(1-0.999**t))+1e-8)
    acc=(fwd(Xte)[-1].argmax(1)==yte).mean(); print(ep,acc,flush=True)
np.savez_compressed(out,W0=W[0],W1=W[1],W2=W[2],b0=B[0],b1=B[1],b2=B[2],test_acc=acc)
