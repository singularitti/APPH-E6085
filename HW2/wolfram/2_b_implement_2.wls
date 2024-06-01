#!/usr/bin/env wolframscript
(* ::Package:: *)

Clear["Global`*"];
l={0.298073,1.242567,5.782948,38.474970};
H=Table[Integrate[4*Pi*(-2+3*l[[i]]*rr-2*l[[i]]^2*rr^3)/rr*Exp[-(l[[i]]+l[[j]])*rr^2]*rr^2,{rr,0,Infinity}],{i,1,4},{j,1,4}];
S=Table[Integrate[4*Pi*Exp[-(l[[i]]+l[[j]])*rr^2]*rr^2,{rr,0,Infinity}],{i,1,4},{j,1,4}];
c={0.08704173,0.52509737,0.51920929,0.31233737};
t2=Sum[c[[i]]c[[j]]Exp[-l[[i]]r2^2]Exp[-l[[j]]r2^2],{i,1,4},{j,1,4}];
K=NIntegrate[Table[8Pi^2 r1^2 r2^2 Sin[theta]t2 Exp[-l[[i]]r1^2]Exp[-l[[j]]r1^2]1/Sqrt[r1^2+r2^2-2 r1 r2 Cos[theta]],{i,1,4},{j,1,4}],{theta,0,Pi},{r1,0,Infinity},{r2,0,Infinity}];
Eigenvalues[{H+K,S}]
