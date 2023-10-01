#!/usr/bin/env wolframscript
(* ::Package:: *)

(*First calculate 4 phi integral then contract.*)
Clear["Global`*"]
trial={0.08704173,0.52509737,0.51920929,0.31233737};
phi[r_]:={Exp[-0.298073r^2],Exp[-1.242567r^2],Exp[-5.782948r^2],Exp[-38.474970r^2]};
t2=TensorProduct[phi[r2],phi[r2]]//Expand;
t1=TensorProduct[phi[r1],phi[r1]]//Expand;
hartreeTerm=8Pi^2 NIntegrate[r1^2 r2^2 Sin[theta]*TensorProduct[t2,t1]*1/Sqrt[r1^2+r2^2-2r1 r2 Cos[theta]],{theta,0,Pi},{r1,0,Infinity},{r2,0,Infinity}];
Sum[(TensorProduct[trial,trial]*hartreeTerm)[[i,j]],{i,1,4},{j,1,4}](*Contract 16x16 matrix to 4x4 matrix*)
