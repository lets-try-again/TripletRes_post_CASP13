import numpy as np 
import os,sys,re,pickle
from subprocess import Popen, PIPE, STDOUT
import torch
import resnet,precision

USEcuda=torch.cuda.is_available()
print('usecuda',USEcuda)

expdir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(expdir,'model')
models=os.listdir(model_path)
models=[amodel for amodel in models ]

msafile=sys.argv[1]
savefile=sys.argv[2]
def getweights_out(msafile,seq_id,outfile):
    exefile=os.path.join(os.path.dirname(__file__),'bin/calNf_ly')
    cmd=exefile+' '+msafile+' '+str(seq_id)+' >'+outfile
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,close_fds=True)
    output,error = p.communicate()
def outpre_(pre):
    [n,d]=pre.shape
    ilist=[]
    jlist=[]
    newpre=[]
    for i in range(n):
        for j in range(i+1,d):
            
            newpre.append(-pre[i,j])
            
            ilist.append(i)
            jlist.append(j)
    index=np.argsort(newpre)
    res=[]
    l=len(index)
    for i in range(l):
        res.append(str(ilist[index[i]]+1)+' '+str(jlist[index[i]]+1)+' '+str(pre[ilist[index[i]],jlist[index[i]]])+'\n')
    return res
def outpre(predicted,outfile):
    outlines=outpre_(predicted)
    woutfile=open(outfile,'w')
    for aline in outlines:
        woutfile.write(aline)
    woutfile.close()    
seq_id=0.8
weightfile=savefile+'.weight'
getweights_out(msafile,seq_id,weightfile)

pre,cov=precision.computepre(msafile,weightfile)
plm=precision.computeplm(msafile,weightfile,savefile)



cov,pre,plm=torch.FloatTensor(cov),torch.FloatTensor(pre),torch.FloatTensor(plm)
if USEcuda:
    cov,pre,plm=cov.cuda(0),pre.cuda(0),plm.cuda(0)
cov,pre,plm=cov.unsqueeze(0),pre.unsqueeze(0),plm.unsqueeze(0)
preds=[]
for modelfile in models:
    model=resnet.resnet86_triple()
    if USEcuda:
        model=model.cuda()
    saved_model=model_path+'/'+modelfile
    model_dic=torch.load(saved_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dic)    
    with torch.set_grad_enabled(False):
        model.eval()
        _,pcon=model(cov,pre,plm)
        pcon=pcon.cpu().data. numpy()[0][0]
        #print(pcon.shape)
        preds.append(pcon)

pred=np.mean(preds,0)
outpre(pred,savefile)