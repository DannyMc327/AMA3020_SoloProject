import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

a = 0.1
#a = 5
bOver = 5
bCritical = 2
bUnder = 0.2
#omega = 1+1e-5
omega = 0.5
#C_1 = 1
C_1 = 0.5
#maxT = 0.125
maxT = 60
time = np.linspace(0, maxT, 1000)

def resonantFrequency(b):
    omega = np.sqrt(np.abs(1-(b**2)/4))
    return omega

def x_particular(a, b, omega, t):
    x_particular = (
        (a*np.cos(omega*t))/(1-omega**2+(b**2*omega**2/(1-omega**2)))+
        (a*b*omega*np.sin(omega*t))/(1+b**2*omega**2/(1+omega**2)**2)
    )
    return x_particular

def x_homogeneous(b,t):
    C_2 = (b/2) * C_1
    if(b**2>4):
        #print("overdamped")
        r_1 = (-b + np.sqrt(b**2-4))/2
        r_2 = (-b - np.sqrt(b**2-4))/2
        return C_1 * np.exp(r_1*t) + C_2 * np.exp(r_2*t)
    elif(b**2==4):
        #print("critically damped")
        return (C_1 + C_2*t)*np.exp((-b/2)*t)
    elif(b**2<4):
        #print("underdamped")
        alpha = -b/2
        beta = (np.sqrt(4 - b**2))/2
        r_1 = alpha + beta
        r_2 = alpha - beta
        return C_1 * np.cos(beta * t) * np.exp(alpha * t) + C_2 * np.sin(beta * t) * np.exp(alpha * t)

x_exact1 = []
x_exact2 = []
x_exact3 = []

for t in time:
    x_exact1.append(x_homogeneous(bUnder,t) + x_particular(a,bUnder,omega,t))
    x_exact2.append(x_homogeneous(bCritical,t) + x_particular(a,bCritical,omega,t))
    x_exact3.append(x_homogeneous(bOver,t) + x_particular(a,bOver,omega,t))
#################
#ANALYTICAL PLOTS
#################
#plt.plot(time, x_exact1, label='Underdamped')
#plt.plot(time, x_exact2, label='Critically Damped')
#plt.plot(time, x_exact3, label='Overdamped')

plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')

def du1(time, z):
    return [z[1], a*np.sin(omega*time) - np.sin(z[0]) - bOver*z[1]]

def du2(time, z):
    return [z[1], a*np.sin(omega*time) - np.sin(z[0]) - bCritical*z[1]]

def du3(time, z):
    return [z[1], a*np.sin(omega*time) - np.sin(z[0]) - bUnder*z[1]]

sol1 = solve_ivp(du1, [0, time.max()], [C_1,C_1], method='RK45', max_step = 0.01)
sol2 = solve_ivp(du2, [0, time.max()], [C_1,C_1], method='RK45', max_step = 0.01)
sol3 = solve_ivp(du3, [0, time.max()], [C_1,C_1], method='RK45', max_step = 0.01)
w = []

###########
#RK45 PLOTS
###########
plt.plot(sol1.t, sol1.y[0,:], label='overdamped')
plt.plot(sol2.t, sol2.y[0,:], label='critically damped')
plt.plot(sol3.t, sol3.y[0,:], label='underdamped')

bRange = [0.2, 0.3, 0.5, 1]
omegaRange = np.arange(0.01, 2, 0.01)

plt.title(rf'$a={a},\omega={omega}$')
plt.grid()
plt.legend()

plt.figure()
for b in bRange:
    amplitudes = a / np.sqrt((1 - omegaRange**2)**2 + b**2 * omegaRange**2)
    plt.plot(omegaRange, amplitudes, label=f'$b={b}$')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A$')
plt.grid()
plt.legend()

plt.show()
