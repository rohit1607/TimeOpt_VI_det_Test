# from TimeOpt_Det_VI_multist.extract_velocity_field import *
import matplotlib.pyplot as plt
from Revanth_comparison.TimeOpt_Det_VI_multistate_for_comparison.grid_world_multistate_index import *
from custom_functions import *
import numpy as np

nt=40
xs=np.arange(0,10)
ys=xs
vStream_x=np.zeros((nt,len(ys),len(xs)))
vStream_y=np.zeros((nt,len(ys),len(xs)))
t_list=np.arange(0,nt)

#print(P)
vStream_x[:,4:5,:]=0.1
vStream_x[:,5:6,:]=0.5
vStream_x[:,6:7,:]=0.9


# vStream_x[:,6:8, :]=2

X,Y=my_meshgrid(xs,ys)
print(X.shape, Y.shape, vStream_x.shape, vStream_y.shape)

#set up grid
#start and endpos are tuples of !!indices!!
g=timeOpt_grid(t_list, xs, ys, (3,9), (9,3))

policy={}
#f=open('pol_50x50x50x8.txt','r')
f=open('output.txt','r')
if f.mode == 'r':
    policy = eval(f.read())
f.close()


traj,(t,i,j),val=plot_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y, fname='stateApx')
# traje,(te,ie,je),vale=plot_exact_trajectory(g, policy, xs, ys, X, Y, vStream_x,vStream_y, fname='noApx')

print("Time-steps requirred for optimal path")
print("state-trajectory", t)
# print("position-trajectory", te)

#print("Value function", val)

