# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:39:40 2017

@author: 74297
"""

import numpy as np
import os
from subprocess import Popen, PIPE, STDOUT
from io import BytesIO
aadic={
        'A':1,
        'B':0,
        'C':2,
        'D':3,
        'E':4,
        
        'F':5,
        'G':6,
        'H':7,
        'I':8,
        'J':0,
        
        'K':9,
        'L':10,
        'M':11,
        'N':12,
        'O':0, 
        
        'P':13,
        'Q':14,
        'R':15,
        'S':16,
        'T':17,
        'U':0,
        
        'V':18,
        'W':19,
        'X':0,
        'Y':20,
        'Z':0,
        '-':0,
        '*':0,
        }



            
def read_msa(file_path):    
    lines=open(file_path).readlines()  
    lines=[line.strip() for line in lines]   
    n=len(lines)
    d=len(lines[0]) #CR AND LF  
    msa=np.zeros([n,d],dtype=int)
    for i in  range(n):
        aline=lines[i]
        for j in range(d):
            msa[i,j]=aadic[aline[j]]
    return msa


def cal_large_matrix1(msa,weight):
    #output:21*l*21*l
    ALPHA=21
    pseudoc=1
    M=msa.shape[0]
    N=msa.shape[1]
    pab=np.zeros((ALPHA,ALPHA))
    pa=np.zeros((N,ALPHA))
    cov=np.zeros([N*ALPHA,N*ALPHA ])
    for i in range(N):
        for aa in range(ALPHA):
            pa[i,aa] = pseudoc
        neff=0.0
        for k in range(M):
            pa[i,msa[k,i]]+=weight[k]
            neff+=weight[k]
        for aa in range(ALPHA):
            pa[i,aa] /=pseudoc * ALPHA * 1.0 + neff
    #print(pab)
    for i in range(N):
        for j in range(i,N):
            for a in range(ALPHA):
                for b in range(ALPHA):
                    if i ==j :
                        if a==b :
                            pab[a,b]=pa[i,a]
                        else:
                            pab[a,b]=0.0
                    else:
                        pab[a,b] = pseudoc *1.0 /ALPHA
            if(i!=j):
                neff2=0;
                for k in range(M):
                    a=msa[k,i]
                    b=msa[k,j]
                    tmp=weight[k]
                    pab[a,b]+=tmp
                    neff2+=tmp
                for a in range(ALPHA):
                    for b in range(ALPHA):
                        pab[a,b] /= pseudoc*ALPHA*1.0 +neff2
            for a in range(ALPHA):
                for b in range(ALPHA):
                    if(i!=j or a==b):
                        if (pab[a][b] > 0.0):
                            cov[i*21+a][j*21+b]=pab[a][b] - pa[i][a] * pa[j][b]
                            cov[j*21+b][i*21+a]=cov[i*21+a][j*21+b]
                            #cov[a*ALPHA+b,i,j,]=pab[a][b] - pa[i][a] * pa[j][b]
                            #cov[a*ALPHA+b,j,i,]=cov[a*ALPHA+b,i,j]
            #cov1=cal_cov(pab,pa[<unsigned int>i],pa[<unsigned int>j])
            #cov[<unsigned int>i,<unsigned int>j]=cov[<unsigned int>j,<unsigned int>i]=cov1
    #print(pab)
    #print(pa)
    return cov 


    