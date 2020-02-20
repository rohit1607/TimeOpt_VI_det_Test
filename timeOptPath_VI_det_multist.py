from Revanth_comparison.TimeOpt_Det_VI_multistate_for_comparison.grid_world_multistate_index import *
import random
import matplotlib.pyplot as plt
import time
import math
from custom_functions import *
#from Revanth_comparison.TimeOpt_Det_VI_multistate_for_comparison.extract_velocity_field import *


threshold=1e-3

#set up stream velocity
#vStream_x, vStream_y, xs, ys, t_list = velocity_field()
nt=25
xs=np.arange(0, 20)
ys=xs
vStream_x=np.zeros((nt,len(ys),len(xs)))
vStream_y=np.zeros((nt,len(ys),len(xs)))
t_list=np.arange(0,nt)

#print(P)
# vStream_x[1,4:5,:]=0.1
# vStream_x[1,5:6,:]=0.5
# vStream_x[1,6:7,:]=0.9
#
# vStream_x[2,4:5,:]=0.2
# vStream_x[2,5:6,:]=0.7
# vStream_x[2,6:7,:]=0.5

vStream_x[:, 7:13, :]=2

X,Y=my_meshgrid(xs,ys)
print(X.shape, Y.shape, vStream_x.shape, vStream_y.shape)

#set up grid
#start and endpos are tuples of !!indices!!
g=timeOpt_grid(t_list, xs, ys, (18,9), (2,9))
# g=timeOpt_grid(t_list, xs, ys, (22,2), (2,22))

# for t in range(g.nt):
# 	for x in range(g.nx):
# 		for y in range(g.ny):
#
# 			print((t,x,y), vStream_x[t,x,y], vStream_y[t,x,y])

#initialise value functions and policy
policy={}
V={}
action_state_space=g.ac_state_space()
for s in g.state_space():
    V[s]=0
    policy[s]=None

for s in action_state_space:
    policy[s]=random.choice(g.actions[s])

#Set up transition probabilities
# print("----initial policy-----")
# print_policy(policy,g)

countb=0
start=time.time()

#print(action_state_space)

while True:

    countb+=1
    delV=-10
    print("iter: ",countb)

    for s in action_state_space:
        old_V=V[s]
        g.set_state(s)
        t,i,j=g.current_state()
        best_val=-float('inf')
        new_val=-float('inf')
        for a in g.actions[s]:
            g.set_state(s)
            vx=vStream_x[t,i,j]
            vy=vStream_y[t,i,j]
            r=g.move(a,vx,vy) #for reward. for state update based on only action(thrust,dir)
            #print(s, a, g.current_state())
            if g.if_within_grid():
                new_val=r+V[g.current_state()]
            if new_val>best_val:
                best_val=new_val

        V[s]=best_val
        delV=max(delV, np.abs(V[s]-old_V))

    if delV<threshold:
        print("Peval iters=", countb)
        break


#find policy

for s in action_state_space:
    g.set_state(s)
    t,i,j = g.current_state()
    #print(t,x,y)
    best_val = -float('inf')
    new_val = -float('inf')
    new_a=None
    for a in g.actions[s]:
        g.set_state(s)
        #print(t,x,y)
        vx = vStream_x[t,i,j]
        vy = vStream_y[t,i,j]
        r = g.move(a,vx,vy)
        #print(s,a,g.current_state())
        if g.if_within_grid():
            new_val = r + V[g.current_state()]
        if new_val>best_val:
            best_val=new_val
            new_a=a
    policy[s]=new_a


end=time.time()


outputfile=open('output.txt','w')
print(policy, file=outputfile)
outputfile.close()

V_outputfile = open('V_outputfile2.txt','w')
print(V,file = V_outputfile)
V_outputfile.close()


traj,(t,i,j),val=plot_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y, fname='stateApx_new2')
# traje,(te,ie,je),vale=plot_exact_trajectory(g, policy, xs, ys, X, Y, vStream_x,vStream_y, fname='noApx')

print("Time-steps requirred for optimal path")
print("state-trajectory", t)
# print("position-trajectory", te)
print("Iterations:", countb)
print("MDP solve time: ", end-start)


# for t in range(len(trajectory)+1):
#     plot_trajectory(trajectory ,vStream_x, vStream_y, xs, ys, t)
# print("trajectory:", trajectory)
# print("time taken:", t*g.dt)
# print("prog exec time", end-start)
# xtr=[]
# ytr=[]
# for a in trajectory:
#     xtr.append(xs[a[0]])
#     ytr.append(ys[a[1]])
# #print(xtr,ytr)
# plt.plot(xtr,ytr)
# plt.grid()
# plt.grid(linewidth=1)
#
# plt.quiver(X,Y,vStream_x[0,:,:],vStream_y[0,:,:])
# plt.show()


