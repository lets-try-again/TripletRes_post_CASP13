# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:48:22 2017

@author: lee
"""
import numpy as np
import scipy.linalg as la 
import numpy.linalg as na
import os
import aaweights
import sys
from subprocess import Popen, PIPE, STDOUT
from io import BytesIO

def ROPE(S, rho):
    p=S.shape[0]
    S=S
    try:
        LM=na.eigh(S)
    except:
        LM=la.eigh(S)
    L=LM[0]
    M=LM[1]
    for i in range(len(L)):
        if L[i]<0:
            L[i]=0
    lamda=2.0/(L+np.sqrt(np.power(L,2)+8*rho))
    indexlamda=np.argsort(-lamda)
    lamda=np.diag(-np.sort(-lamda)[:p])
    hattheta=np.dot(M[:,indexlamda],lamda)
    hattheta=np.dot(hattheta,M[:,indexlamda].transpose())
    return hattheta
def blockshaped(arr,dim=21):
    p=arr.shape[0]//dim
    re=np.zeros([dim*dim,p,p])
    for i in range(p):
        for j in range(p):
            re[:,i,j]=arr[i*dim:i*dim+dim,j*dim:j*dim+dim].flatten()
    return re
def computepre(msafile,weightfile):
    msa=aaweights.read_msa(msafile)
    weights=np.genfromtxt(weightfile).flatten()
    cov=(aaweights.cal_large_matrix1(msa,weights))
    rho2=np.exp((np.arange(80)-60)/5.0)[30]
    pre=ROPE(cov,rho2)
    #print(pre)
    return blockshaped(pre),blockshaped(cov)
  
    
    
def computeccm(msafile,savefile):
    #exefile='bin/ccmpred'
    exefile=os.path.join(os.path.dirname(__file__),'bin/ccmpred')
    cmd=exefile+' -r '+savefile+'.raw'+' '+msafile+' '+savefile+'.del'
    #p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,close_fds=True)
    os.system(cmd)

def readfromsavefile(savefile):
    lines=open(savefile+'.raw').readlines()
    L=0
    for i in range(len(lines)):
        if lines[i][0]=='#':
            L=i
            break
    print(L)
    precision=np.zeros([L*21,L*21])
    #first:
    for i in range(L):
        vec=np.genfromtxt(BytesIO(lines[i].encode()))
        for j in range(21):
            precision[i*21+j,i*21+j]=0
    count=0;
    for i in range(L):
        for k in range(i+1,L):
            count+=1
            for j in range(21):
                vec=np.genfromtxt(BytesIO(lines[L+22*(count-1)+1+j].encode())).reshape([1,-1])
                precision[i*21+j,k*21:k*21+21]=vec
                precision[k*21:k*21+21,i*21+j:i*21+j+1]=np.transpose(vec)
    #cleaning    
    os.remove(savefile+'.del')
    os.remove(savefile+'.raw')
    return blockshaped(precision)
def computeapre(msafile,weightfile,savefile):
    #weight file is useless
    print(msafile)
    #if not os.path.isfile(savefile+'.npy222'):
    computeccm(msafile,savefile)
    pre=readfromsavefile(savefile)
    pre=pre.astype('float16')
    np.save(savefile,pre)
def computeplm(msafile,weightfile,savefile):
    #weight file is useless
    print(msafile)
    computeccm(msafile,savefile)
    pre=readfromsavefile(savefile)
    return pre           
def compute_fasta(fasta,updir,savedir):
    #updir='/oasis/projects/nsf/mia174/liyangum/deepPRE/makealn/'
    #savedir='/oasis/scratch/comet/liyangum/temp_project/pre_compute/'
    lines=open(fasta).readlines()
    pdbids=[lines[2*i][1:].strip() for i in range(len(lines)//2)] 
    #pool = Pool(processes=3)
    for pdbid in pdbids:
        msafile=updir+pdbid+'/'+pdbid+'.aln'
        weightfile=updir+pdbid+'/'+pdbid+'.weight'
        savefile=savedir+pdbid+'.pre'
        computeapre(msafile,weightfile,savefile)
        #pool.apply_async(computeapre,[msafile,weightfile,savefile])
    #pool.close()
    #pool.join()     
if __name__ == "__main__":    
    updir='/oasis/projects/nsf/mia174/liyangum/deepPRE/makealn/'
    savedir= '/oasis/scratch/comet/liyangum/temp_project/pre_compute/plm/'   
    #if not os.path.isdir(savedir):
    #    os.makedirs(savedir)
    inputfasta=sys.argv[1]
    compute_fasta(inputfasta,updir,savedir)