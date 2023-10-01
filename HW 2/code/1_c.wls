#!/usr/bin/env wolframscript
(* ::Package:: *)

ClearAll["Global`*"]
f[a_,b_]:=4 Pi (-1+3 a r-2 a^2 r^3)/r Exp[-(a+b) r^2] r^2
H=ParallelTable[NIntegrate[f[a,b],{r,0,Infinity}],{a,{13,1.96,0.44,0.12}},{b,{13,1.96,0.44,0.12}}]
S=ParallelTable[4Pi NIntegrate[Exp[-(a+b) r^2] r^2,{r,0,Infinity}],{a,{13,1.96,0.44,0.12}},{b,{13,1.96,0.44,0.12}}]
evals=Eigenvalues[{H,S}]
evecs=Eigenvectors[{H,S}]
renormalize[evec_]:=If[Length[Select[evec,#<0&]]>=3,-evec,evec](*\:5982\:679c\:67093\:4e2a\:5750\:6807\:662f\:8d1f\:6570\:5c31\:53cd\:5411\:8be5\:5411\:91cf*)
evecs=renormalize/@evecs
g=Table[evecs[[i]] . {Exp[-13r^2],Exp[-1.96r^2],Exp[-0.44r^2],Exp[-0.12r^2]},{i,1,4}]
Plot[g,{r,0,3},PlotRange->{0,0.7}]
