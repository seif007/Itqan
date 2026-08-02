#!/usr/bin/env python3
"""Probe the GRU parity: replicate the Kotlin TajweedNet GRU math in numpy and
compare to PyTorch for one real clip. If they match, the Kotlin GRU is correct
and the divergence is elsewhere; if they differ, the GRU port is wrong."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch, train as T

DS = os.path.join(HERE, "..", "dataset")

def conv_block(x):
    # x: (inCh, T) numpy
    def conv1d(xin, W, b, k=3, pad=1):
        cin, T = xin.shape
        Tout = T
        padded = np.pad(xin, ((0,0),(pad,pad)), mode="constant")
        out = np.zeros((W.shape[0], Tout), dtype=np.float32)
        for co in range(W.shape[0]):
            for ti in range(Tout):
                for ki in range(k):
                    for ci in range(cin):
                        out[co,ti] += W[co, ci, ki] * padded[ci, ti+ki]
                out[co,ti] += b[co]
        return np.maximum(out, 0.0)
    def maxpool(xin, p=2):
        cin, T = xin.shape
        Tout = T//2
        return np.maximum(xin[:, 0:Tout*2:2], xin[:, 1:Tout*2:2])
    c = x
    c = conv1d(c, W0, b0); c = maxpool(c)
    c = conv1d(c, W1, b1); c = maxpool(c)
    c = conv1d(c, W2, b2); c = maxpool(c)
    return c

def gru_cell(x, h, Wih, Whh, bih, bhh):
    hDim = len(h)
    xDim = len(x)
    ih = Wih @ x + bih
    hh = Whh @ h + bhh
    r = 1.0/(1.0+np.exp(-(ih[0:hDim]+hh[0:hDim])))
    z = 1.0/(1.0+np.exp(-(ih[hDim:2*hDim]+hh[hDim:2*hDim])))
    n = np.tanh(ih[2*hDim:3*hDim] + r*hh[2*hDim:3*hDim])
    return (1-z)*n + z*h

def kotlin_gru(c, Wih, Whh, bih, bhh):
    # c: (channels, time)  -> GRU expects (time, channels)
    C, T = c.shape
    hDim = Whh.shape[1]  # GRU hidden size (64)
    h = np.zeros(hDim); hR = np.zeros(hDim)
    fwd=[]; bwd=[]
    for t in range(T):
        xt = c[:, t]
        nf = gru_cell(xt, h, Wih, Whh, bih, bhh); fwd.append(nf); h = nf
    for t in range(T-1,-1,-1):
        xt = c[:, t]
        nb = gru_cell(xt, hR, Wih, Whh, bih, bhh); bwd.append(nb); hR = nb
    bwd = bwd[::-1]
    fwd = np.stack(fwd); bwd = np.stack(bwd)
    pooled = np.concatenate([fwd.mean(0), bwd.mean(0)])
    return pooled

def main():
    global W0,b0,W1,b1,W2,b2
    model = T.TajweedNet(rules=T.RULES, dropout=0.0)
    sd = torch.load(os.path.join(DS,"model_best.pt"), map_location="cpu")
    model.load_state_dict(sd); model.eval()
    # pull conv weights [outCh,inCh,k] -- Conv1d at indices 0,3,6
    convs = [m for m in model.conv if isinstance(m, torch.nn.Conv1d)]
    assert len(convs) == 3, f"expected 3 convs, got {len(convs)}"
    def cw(i):
        layer = convs[i]
        return layer.weight.data.numpy(), layer.bias.data.numpy()
    W0,b0 = cw(0); W1,b1 = cw(1); W2,b2 = cw(2)
    gru = model.gru
    Wih = gru.weight_ih_l0.data.numpy(); Whh = gru.weight_hh_l0.data.numpy()
    bih = gru.bias_ih_l0.data.numpy(); bhh = gru.bias_hh_l0.data.numpy()

    fixture = json.load(open(os.path.join(DS,"parity_fixture.json")))
    clip = fixture[0]
    lm = np.array(clip["logmel"], dtype=np.float32)   # (40, T)
    c = conv_block(lm)
    pooled_k = kotlin_gru(c, Wih, Whh, bih, bhh)

    # PyTorch reference
    x = torch.from_numpy(lm).unsqueeze(0)
    with torch.no_grad():
        cc = model.conv(x)
        cc = cc.permute(0,2,1)
        out, _ = model.gru(cc)
        pooled_t = out.mean(1).numpy()[0]
    print("pooled Kotlin:", np.round(pooled_k[:6],4))
    print("pooled Torch:", np.round(pooled_t[:6],4))
    print("max abs diff:", np.max(np.abs(pooled_k-pooled_t)))

if __name__=="__main__":
    main()
