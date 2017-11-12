#!/usr/bin/env python

from __future__ import print_function
from fenics import *
import numpy as np 
from mshr import *
import math

domain = Circle(Point(0,0),1)
mesh = generate_mesh(domain,32)
V = FunctionSpace(mesh,'P',2)

def boundary(x,on_boundary):
  return on_boundary


u_next = TrialFunction(V)
v = TestFunction(V)

k1 = Constant(str(0.03))
k2 = Constant(str(0.012))


L_amp = 0.02
L_radius = 0.5
D = Constant(str(2*3600*1.67*1e-7))
beta_min = 125
beta_max = 298
K = 0.0017

f_light = Expression('x[0]*x[0] + x[1]*x[1] < L_rad*L_rad ? (K/(K+L))*(b_max-b_min) + b_min : b_max',degree=1,K= K,L =L_amp,L_rad = L_radius,b_max = beta_max,b_min = beta_min)

T0 = 0.0
T1 = 24.0
dt = 0.01
num_steps = int(math.ceil((T1-T0)/dt))


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
  plot(u)
  print("max: {0}, min: {1}".format(np.max(u.vector().array()),np.min(u.vector().array())))
  u_prev.assign(u)

plot(u)
interactive()
