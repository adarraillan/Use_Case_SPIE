from __future__ import annotations
from random import randint
from random import uniform
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class FileData:
    n_func: int
    n_obj: int
    max_weight: int
    mat_profits: list[list[int]]
    obj_weights: list[int]

def parse_list(data):
    row = data[:-1].replace('[','').replace(']','').split(', ')
    row = [int(x) for x in row]

    return row

def parse_file(data):
    fdat = FileData()

    fdat.n_func = int(data[0])
    fdat.n_obj = int(data[1])
    fdat.max_weight = int(data[2])

    mat_profits = []
    for i in range(fdat.n_func):
        ind = 3+i

        row = parse_list(data[ind])

        mat_profits.append(row)
    
    fdat.mat_profits = mat_profits
    fdat.obj_weights = parse_list(data[-1])

    return fdat

def f(X: list[int],fdat: FileData,i: int):
    return sum([fdat.mat_profits[i][j] for j in range(fdat.n_obj) if X[j]==1 ])

def F(X: list[int],fdat: FileData):
    return sum( [0 if f(X,fdat,i)==0 else np.log(f(X,fdat,i)) for i in range(fdat.n_func) ] )

def is_X_respecting_criteria(X,fdat: FileData):
    return fdat.max_weight >= sum( [fdat.obj_weights[i] for i in range(fdat.n_obj) if X[i]==1 ] )

def neighbor(X,fdat):
    while True:
        i = randint(0,len(X)-1)
        Xvois = X.copy()
        if X[i] == 0:
            Xvois[i] = 1
        else:
            Xvois[i] = 0

        if is_X_respecting_criteria(Xvois,fdat):
            break
    
    return Xvois

def is_a_dominated_by_b(A,B,fdat: FileData):
    fA_list = [ f(A,fdat,i) for i in range(fdat.n_func) ]
    fB_list = [ f(B,fdat,i) for i in range(fdat.n_func) ]

    dominance_list = [1 if fB_list[i] >= fA_list[i] else 0 for i in range(fdat.n_func) ]

    return sum(dominance_list) == len(dominance_list)

T,D_COEF = 40, 0.1
TMIN = 0.01

file = open("/home/cytech/Documents/S5/4_Optimisation_Metaheuristique/kp/KP/KP_p-3_n-10_ins-5.dat")

data = [str(x).replace('\n','') for x in list(file) if x != '\n']

file.close()

fdat = parse_file(data)

#Random candidate
X = [0 for _ in range(fdat.n_obj)]
fx = F(X,fdat)
archive = [X]
rejected = []

n = 0
while T > TMIN and n<10000:
    cpt = 0
    while cpt < 10000:
        cpt+=1

        Xvois = neighbor(X,fdat)
        fxvois = F(Xvois,fdat)
        deltaF = fxvois - fx

        tmp = float(deltaF)/float(T)
        if abs(tmp) > 100000:
            tmp = (tmp/tmp)*100000

        if deltaF < 0 or uniform(0,1) < np.exp(-tmp):
            X = Xvois
            fx = fxvois

        #dominance test
        is_not_dominated = True
        for Y in archive:
            if is_a_dominated_by_b(X,Y,fdat):
                is_not_dominated = False
                break

        #adding to archive and cleaning archive
        if is_not_dominated:
            for Y in archive:
                if is_a_dominated_by_b(Y,X,fdat):
                    archive.remove(Y)
                    if Y not in rejected:
                        rejected.append(Y)
            archive.append(X)


    T = T*D_COEF
    n+=1

#PLOT
points_front = []
points_rejected = []
for x in archive:
    points_front.append([f(x,fdat,i) for i in range(fdat.n_func)])
for x in rejected:
    points_rejected.append([f(x,fdat,i) for i in range(fdat.n_func)])

x_f,y_f,z_f = zip(*points_front)
x_r,y_r,z_r = zip(*points_rejected)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_f, y_f, z_f, c='r')
ax.scatter3D(x_r, y_r, z_r, c='b')

ax.plot_trisurf(x_f, y_f, z_f, cmap='Oranges')

plt.show()