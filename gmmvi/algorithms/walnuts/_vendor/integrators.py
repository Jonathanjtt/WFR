#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 10:22:43 2025

@author: torekleppe
"""
from .constants import __logZero
import numpy as np
import sys

# -------------------------------------
# integrators
# -------------------------------------

# common return object
class integratorReturn:
    def __init__(self,q1,v1,lp1,grad1,nEvalF,nEvalB,If,Ib,c,lwt,igrConst):
        self.q = q1
        self.v = v1
        self.lp = lp1
        self.grad = grad1
        self.nEvalF = nEvalF
        self.nEvalB = nEvalB
        self.If = If
        self.Ib = Ib
        self.c = c
        self.lwt = lwt
        self.igrConst = igrConst
        
    def __str__(self):
        return "dim: " + str(len(self.q)) + " If: " + str(self.If) + " Ib: " + str(self.Ib) + " c: " + str(self.c)


# class used for passing assorted tuning parameters to the integrators
class integratorAuxPar:
    def __init__(self,minC=0,maxC=10,R2Pprob0=2.0/3.0,maxFPiter=30,FPtol=1.0e-8,FPNewton=False,rescaledGradThresh=5.0):
        self.minC=minC
        self.maxC=maxC
        self.R2Pprob0=R2Pprob0
        self.maxFPiter=maxFPiter
        self.FPtol=FPtol
        self.FPNewton=FPNewton
        self.rescaledGradThresh=rescaledGradThresh
    
        

def adaptLeapFrogR2P(q,v,g,Ham0,h,xi,lpFun,delta,auxPar):
    
    nEvalF = 0
    If = auxPar.maxC
    for c in range(auxPar.minC,auxPar.maxC+1):
        nstep = 2**c
        hh = h/nstep
        qq = q
        vv = xi*v
        gg = g
        Hams = np.zeros(nstep+1)
        Hams[0] = Ham0
        
        for i in range(1,nstep+1):
            vh = vv + 0.5*hh*gg
            qq = qq + hh*vh
            fnew,gg = lpFun(qq)
            nEvalF += 1
            vv = vh + 0.5*hh*gg
            Hams[i] = -fnew + 0.5*sum(vv*vv)
        
        
        maxErr = abs(Hams[0]-Hams[-1])
        #print(maxErr)
        
        
        if(all(np.isfinite(Hams)) and maxErr < delta):
            If = c
            break
    
    
    if(np.random.uniform()<auxPar.R2Pprob0):
        # simulation occur at minimal accepted precision
        qOut = qq
        vOut = vv
        fOut = fnew
        gOut = gg
        cSim = If
        igrConst = hh*(np.max(np.abs(np.diff(Hams)))**(-1.0/3.0))
    else:
        # simulation occur at minimal + 1 
        c = If+1
        nstep = 2**c
        hh = h/nstep
        qq = q
        vv = xi*v
        gg = g
        Hams = np.zeros(nstep+1)
        Hams[0] = Ham0
        
        for i in range(1,nstep+1):
            vh = vv + 0.5*hh*gg
            qq = qq + hh*vh
            fnew,gg = lpFun(qq)
            nEvalF += 1
            vv = vh + 0.5*hh*gg
            Hams[i] = -fnew + 0.5*sum(vv*vv)
            
        qOut = qq
        vOut = vv
        fOut = fnew
        gOut = gg
        cSim = If+1
        igrConst = hh*(np.max(np.abs(np.diff(Hams)))**(-1.0/3.0))
        

    # done forward simulation pass, now do backward simulations
    nEvalB = 0
    
    if(cSim==If):
        maxTry = If-1
        Ib = If
        lwtf = np.log(auxPar.R2Pprob0)
    else:
        maxTry = auxPar.maxC
        Ib = auxPar.maxC
        lwtf = np.log(1.0-auxPar.R2Pprob0)
        
        
    if(maxTry>=auxPar.minC):
    
        
        H0b = Hams[-1]
        for c in range(auxPar.minC,maxTry+1):
            nstep = 2**c
            hh = h/nstep
            qq = qOut
            vv = -vOut
            gg = gOut
            Hams = np.zeros(nstep+1)
            Hams[0] = H0b
            
            for i in range(1,nstep+1):
                vh = vv + 0.5*hh*gg
                qq = qq + hh*vh
                fnew,gg = lpFun(qq)
                nEvalB += 1
                vv = vh + 0.5*hh*gg
                Hams[i] = -fnew + 0.5*sum(vv*vv)
            
            maxErr = abs(Hams[0]-Hams[-1])
            if(all(np.isfinite(Hams)) and maxErr < delta):
                Ib = c
                break    
    
    # done backward simulation pass, now work out backward probability
    lwtb = __logZero
    if(cSim==Ib):
        lwtb = np.log(auxPar.R2Pprob0)
    elif(cSim==Ib+1):
        lwtb = np.log(1.0-auxPar.R2Pprob0)    
        
    return integratorReturn(qOut,xi*vOut,fOut,gOut,nEvalF,nEvalB,If,Ib,cSim,
                            lwtb-lwtf,
                            igrConst)

