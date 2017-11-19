#!/usr/bin/env python3

from __future__ import print_function
from fenics import *
from mshr import *
import math
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

domain = Circle(Point(0,0),1)
mesh = generate_mesh(domain,40)
V = FunctionSpace(mesh,'P',2)

def boundary(x,on_boundary):
  return on_boundary


u_next = TrialFunction(V)
v = TestFunction(V)

k1 = Constant(str(0.03))
k2 = Constant(str(0.012))
k3 = Constant(str(0.8))

L_amp = 0.02
L_radius = 0.5
D = Constant(str(3600*1.67*1e-7))
beta_min = 125
beta_max = 298
K = 0.0017

f_light = Expression('x[0]*x[0] + x[1]*x[1] < L_rad*L_rad ? (K/(K+L))*(b_max-b_min) + b_min : b_max',degree=1,K= K,L =L_amp,L_rad = L_radius,b_max = beta_max,b_min = beta_min)
#hackdy-hack
u2 = Expression('x[0]*x[0] + x[1]*x[1] < L_rad*L_rad ? k3*((K/(K+L))*(b_max-b_min)) + b_min : k3*b_max',degree=1,K= K,L =L_amp,L_rad = L_radius,b_max = beta_max,b_min = beta_min,k3=k3)


####
def f_lux(u1,u2):
	
	#numbers!
	LuxRtot = 2000
	Ka2KDLuxR = 270000

	#because it's easier to read this way
	A = LuxRtot
	B = Ka2KDLuxR


	t1 = (A+B/(4*u1**2))
	t2 = ((A+B/(4*u1**2))**2 - A**2)**0.5

	outp = 0.5*(t1-t2)

	return outp

def f_cl(u1,u2):
	
	KD_Cl = 5

	A = KD_Cl

	outp = (u2/2.0)+ (1.0/(8*A))*(1-(1+8*A)**0.5)

	return outp

def f_logic(u1,u2):
	c0 = 0.04
	c1 = 0.05
	c2 = 0.011
	n = 1.5

	numerator =(c0+c1*f_lux(u1,u1)) 
 	denominator = (1+c0+c1*f_lux(u1,u2) +c2*f_cl(u1,u2)**n + c1*c2*f_lux(u1,u2)*f_cl(u1,u2)**n)

 	return numerator/denominator
####


T0 = 0.0
T1 = 10.0
dt = 0.01
num_steps = int(math.ceil((T1-T0)/dt))
# num_steps=1000


#initial conditions
u_prev = interpolate(Constant('0'),V)

F = u_next*v*dx - u_prev*v*dx - k1*f_light*v*dt*dx + k2*u_prev*v*dt*dx + D*dot(grad(u_prev), grad(v))*dt*dx
a,l = lhs(F), rhs(F)
t=0
u = Function(V)
for n in range(num_steps):
  t += dt
  print(n)
  solve(a == l, u)
  # plot(u)
  print("max: {0}, min: {1}".format(np.max(u.vector().array()),np.min(u.vector().array())))
  u_prev.assign(u)


####Post process

#get mesh points:
ps = [(v.point().x(),v.point().y()) for v in vertices(mesh)]

# main result
u_1 = u 
u1s = [u_1(p[0],p[1]) for p in ps]
plot(u_1) # plot for sanity

#light 
u_2 = interpolate(u2,V)
u2s = [u_2(p[0],p[1]) for p in ps]

# construct u_3
vs = zip(u1s,u2s) #bundle up u_1,u_2 points
k4 = 1
u3s = [k4*f_logic(u[0],u[1]) for u in vs]

fig,axarr = plt.subplots(3)


X = [p[0] for p in ps]
Y = [p[1] for p in ps]

axarr[0].tricontour(X,Y,u1s)
axarr[0].set_title("$u_1$")
axarr[0].set_xlabel("x [no units]")
axarr[0].set_ylabel("y [no units]")

axarr[1].tricontour(X,Y,u2s)
axarr[1].set_title("$u_2$")
axarr[1].set_xlabel("x [no units]")
axarr[1].set_ylabel("y [no units]")


axarr[2].tricontour(X,Y,u3s)
axarr[2].set_title("$u_3$")
axarr[1].set_xlabel("x [no units]")
axarr[1].set_ylabel("y [no units]")

interactive()
plt.show()