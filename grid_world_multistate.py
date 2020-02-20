import numpy as np
import itertools
import math
#import matplotlib.pyplot as plt

class Grid:
	def __init__(self,tlist, xs, ys, start, end):
		self.nx=len(xs)
		self.ny=len(ys)
		self.nt=len(tlist)
		
		self.dx=xs[1] - xs[0]
		self.dy=ys[1] - ys[0]
		self.dt=tlist[1] - tlist[0]
		
		self.xs=xs
		self.ys=ys
		self.tlist=tlist
	
		# self.sigmaX=SigmaX
		# self.sigmaY=SigmaY
		self.i=start[0]
		self.j=start[1]
		self.t=0
		self.endpos=end
		self.start_state=(start[0],start[1],0)
	
	#Rewards and Actions to be dicitonaries
	def set_AR(self, Actions):
		self.actions= Actions
		# self.rewards= Rewards
	
	#explicitly set state. state is a tuple (m,n)
	def set_state(self, state):
		self.t = state[0]
		self.i = state[1]
		self.j = state[2]

	def current_state(self):
		return (self.t, self.i, self.j)

	def current_pos(self):
		return (self.i, self.j)
	
	#MAY NEED TO CHANGE DEFINITION
	def is_terminal(self):
		#return self.actions[state]==None
		return (self.current_pos()==self.endpos)

	def move(self, action, Vx, Vy):
		r = 0
		if self.is_terminal()==False:
			thrust,angle=action
			xnew=(self.i + (thrust*math.cos(angle)+Vx)*self.dt)
			ynew=(self.j + (thrust*math.sin(angle)+Vy)*self.dt)
			
			# xnew=self.i+(Vx*self.dt)
			# ynew=self.j+(Vy*self.dt)

			#if state happens to go out of of grid, bring it back inside
			if xnew>self.xs[-1]:
				xnew=self.xs[-1]
			elif xnew<self.xs[0]:
				xnew=self.xs[0]

			if ynew>self.ys[-1]:
				ynew=self.ys[-1]
			elif ynew<self.ys[0]:
				ynew=self.ys[0]

			# rounding to prevent invalid keys

			remx = (xnew - self.xs[0]) % self.dx
			remy = (ynew - self.ys[0]) % self.dy
			xind = (xnew - self.xs[0]) // self.dx
			yind = (ynew - self.ys[0]) // self.dy

			if remx >= 0.5 * self.dx and remy >= 0.5 * self.dy:
				xind+=1
				yind+=1
			elif remx >= 0.5 * self.dx and remy < 0.5 * self.dy:
				xind+=1
			elif remx < 0.5 * self.dx and remy >= 0.5 * self.dy:
				yind+=1

			self.i=self.xs[xind]
			self.j=self.ys[yind]

			# err_x = np.abs(self.i - xnew)
			# err_y = np.abs(self.j - ynew)

			self.t=self.t + self.dt

			if self.is_terminal():
				r=1
			else:
				r=-1

		return r

	def ac_state_space(self, time=None):
		a=set()
		if time==None:
			for t in range((self.nt)-1): #does not include the states in the last time stice.
				for y in range(self.ny):
					for x in range(self.nx):
						if ((self.xs[x],self.ys[y])!=self.endpos):# does not include states with pos as endpos
							a.add((self.tlist[t], self.xs[x], self.ys[y]))

		else:
			for y in range(self.ny):
				for x in range(self.nx):
					if ((self.xs[x],self.ys[y])!=self.endpos):
						a.add((time, self.xs[x], self.ys[y]))

		return sorted(a)


	def state_space(self,edge=None):
		a=set()
		if edge==None:
			for t in range(self.nt):
				for y in range(self.ny):
					for x in range(self.nx):
						a.add((self.tlist[t], self.xs[x], self.ys[y]))

		elif edge=='l':
			x=0
			for t in range(self.nt):
				for y in range(self.ny):
					a.add((self.tlist[t], self.xs[x], self.ys[y]))

		elif edge=='d':
			y=0
			for t in range(self.nt):
				for x in range(self.nx):
					a.add((self.tlist[t], self.xs[x], self.ys[y]))

		elif edge=='r':
			x=(self.nx)-1
			for t in range(self.nt):
				for y in range(self.ny):
					a.add((self.tlist[t], self.xs[x], self.ys[y]))

		elif edge=='u':
			y=(self.ny)-1
			for t in range(self.nt):
				for x in range(self.nx):
					a.add((self.tlist[t], self.xs[x], self.ys[y])) 
		
		elif edge=='m':
			for t in range(self.nt):
				for y in range(1, (self.ny)-1):
					for x in range(1, (self.nx)-1):
						a.add((self.tlist[t], self.xs[x], self.ys[y]))

		elif edge=='llc':
			x=0
			y=0
			for t in range(self.nt):
				a.add((self.tlist[t], self.xs[x], self.ys[y]))

		elif edge=='ulc':
			x=0
			y=self.ny-1
			for t in range(self.nt):
				a.add((self.tlist[t], self.xs[x], self.ys[y]))

		elif edge=='lrc':
			x=(self.nx)-1
			y=0
			for t in range(self.nt):
				a.add((self.tlist[t], self.xs[x], self.ys[y]))

		elif edge=='urc':
			x=(self.nx)-1
			y=(self.ny)-1
			for t in range(self.nt):
				a.add((self.tlist[t], self.xs[x], self.ys[y]))

		elif edge=='end':
			x=self.endpos[0]
			y=self.endpos[1]
			for t in range(self.nt):
				a.add((self.tlist[t], x, y))

		return sorted(a)
	

	def if_within_grid(self):
		return (self.t>=0 and self.t<self.nt)


def intersection(lst1, lst2): 
	lst3 = [value for value in lst1 if value in lst2] 
	return lst3 

def timeOpt_grid(tlist, xs, ys, startpos, endpos):

	g = Grid(tlist, xs, ys, startpos, endpos)

	#define actions and rewards
	Pi=math.pi

	speed_list=[0.5, 1, 1.5, 2] #speeds except zero
	angle_list_l=[0, 0.25*Pi, 0.5*Pi, 1.5*Pi, 1.75*Pi]
	angle_list_d=[0, 0.25*Pi, 0.5*Pi, 0.75*Pi, Pi]
	angle_list_u=[0, Pi, 1.25*Pi, 1.5*Pi, 1.75*Pi]
	angle_list_r=[0.5*Pi, 0.75*Pi, Pi, 1.25*Pi, 1.5*Pi]
	angle_list_m=[0, 0.25*Pi, 0.5*Pi, 0.75*Pi, Pi, 1.25*Pi, 1.5*Pi, 1.75*Pi]

	action_list_l= list(itertools.product(speed_list,angle_list_l))
	action_list_l.append((0,0))
	action_list_d= list(itertools.product(speed_list,angle_list_d))
	action_list_d.append((0,0))
	action_list_u= list(itertools.product(speed_list,angle_list_u))
	action_list_u.append((0,0))
	action_list_r= list(itertools.product(speed_list,angle_list_r))
	action_list_r.append((0,0))
	action_list_m= list(itertools.product(speed_list,angle_list_m))
	action_list_m.append((0,0))

	actions={}
	for s in g.state_space():
		actions[s]=action_list_m

	#update action set for grid edges
	for s in g.state_space('l'):
		actions[s]=action_list_l

	for s in g.state_space('u'):
		actions[s]=action_list_u

	for s in g.state_space('r'):
		actions[s]=action_list_r

	for s in g.state_space('d'):
		actions[s]=action_list_d

	#update action set for grid corners
	for s in g.state_space('llc'):
		actions[s]=intersection(action_list_d,action_list_l)

	for s in g.state_space('ulc'):
		actions[s]=intersection(action_list_u,action_list_l)

	for s in g.state_space('lrc'):
		actions[s]=intersection(action_list_d,action_list_r)

	for s in g.state_space('urc'):
		actions[s]=intersection(action_list_u,action_list_r)

	#update action for terminal state
	for s in g.state_space('end'):
		actions[s]=None

	#set action set for states in last time step to none
	for s in g.ac_state_space(g.tlist[-1]):
		actions[s]=None

	#set ations for grid
	g.set_AR(actions)

	return g


def print_values(V, g):
	for t in range(g.nt):
		print('t=',t)
		print("")
		for j in reversed(range(g.ny)):
			print("---------------------------")
			for i in range(g.nx):
				v = V[i,j,t]
				print(v, end=" ")
			
			print("")
		print("")


def print_policy(P, g):
	for t in range(g.nt-g.dt):
		print('t=',t)
		print("")
		for j in reversed(range(g.ny)):
			print("---------------------------")
			for i in range(g.nx):
				if (i,j)!=g.endpos:
					a = P[(i,j,t)]
					if a[0]!=0:
						if a[1]==0:
							a=(a[0],' R')
						elif a[1]==math.pi*0.5:
							a=(a[0],' U')
						elif a[1]==math.pi:
							a=(a[0],' L')
						elif a[1]==math.pi*1.5:
							a=(a[0],' D')
						elif a[1]==math.pi*0.25:
							a=(a[0],'UR')
						elif a[1]==math.pi*0.75:
							a=(a[0],'UL')
						elif a[1]==math.pi*1.25:
							a=(a[0],'DL')
						elif a[1]==math.pi*1.75:
							a=(a[0],'DR')
				else:
					a='None'
				print(a,end=" ")
			print("")
		print("")

def print_test(g):
	for t in range(g.nt):
		for j in reversed(range(g.ny)):
			for i in range(g.nx):
				print(i,j,t),

			print("")