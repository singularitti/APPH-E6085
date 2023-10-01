#!/usr/bin/env wolframscript
(* ::Package:: *)

ClearAll["Global`*"]
f[a_,b_]:=4 Pi (-1+3 a r-2 a^2 r^3)/r Exp[-(a+b) r^2] r^2
H=ParallelTable[NIntegrate[f[a,b],{r,0,Infinity}],{a,{13,1.96,0.44,0.12}},{b,{13,1.96,0.44,0.12}}]
S=ParallelTable[4Pi NIntegrate[Exp[-(a+b) r^2] r^2,{r,0,Infinity}],{a,{13,1.96,0.44,0.12}},{b,{13,1.96,0.44,0.12}}]
evals=Eigenvalues[{H,S}]
evecs=Eigenvectors[{H,S}]
