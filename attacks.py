import torch
import torch.nn as nn
import torch.nn.functional as F

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()

def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool
    data = dat.to(device)
    ori=data.data
    if rand_start :
        rand=torch.FloatTensor(data.shape).uniform_(-eps, eps)
        data.data=data.data+rand.data
    label = lbl.to(device)
    loss = nn.CrossEntropyLoss()
    
    for i in range(iters) :
 
        data.requires_grad = True
        
        output = model(data)
        model.zero_grad()
        l = loss(output, label).to(device)
        l.backward()
        
        adv=data + alpha*torch.sign(data.grad)
        update = torch.clamp(adv - ori, min=-eps, max=eps)
        data = torch.clamp(ori +update, min=0.0, max=1.0).detach_()
    return data

def FGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float
    data = dat.to(device)
    label = lbl.to(device)
    loss = nn.CrossEntropyLoss()
    
    data.requires_grad = True
    output = model(data)
    model.zero_grad()
    loss = loss(output, label).to(device)
    loss.backward()
    data = torch.clamp(data+eps*torch.sign(data.grad), min=0.0, max=1.0).detach_()
    return data

def MomentumIterative_attack(model, device, dat, lbl, eps, alpha, iters, mu):
    # TODO: Implement the Momentum Iterative Method
    # - dat and lbl are tensors
    # - eps, alpha and mu are floats
    # - iters is an integer
    data = dat.to(device)
    ori=data.data
    label = lbl.to(device)
    loss = nn.CrossEntropyLoss()
    g=0
    for i in range(iters) :
 
        data.requires_grad = True
        
        output = model(data)
        model.zero_grad()
        l = loss(output, label).to(device)
        l.backward()
        grad=data.grad
        l1n=torch.sum(torch.abs(grad))
        g=mu*g+grad/l1n
        adv=data + alpha*torch.sign(g)
        update = torch.clamp(adv - ori, min=-eps, max=eps)
        data = torch.clamp(ori +update, min=0.0, max=1.0).detach_()
    return data