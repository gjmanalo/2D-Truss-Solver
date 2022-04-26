import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, flash, redirect, render_template, request
from tempfile import mkdtemp
import pandas as pd
import math
from userinput import *
# Configure application
app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True
plt.rcdefaults()

f = open('/workspaces/99926306/~/project/2DTruss_Solver/templates/index.html', 'w')
f.write('{% extends "layout.html" %}\n')

f.write('{% block title %}\n')
f.write('2D Truss Solver\n')
f.write('{% endblock %}\n')
f.write('{% block main %}\n\n')
f.write('<img src=\"{''{url_for(\'static\', filename=\'ExternalForce.png\')}}" id = "container" style = "float:right"/>\n')
f.write('<img src=\"{''{url_for(\'static\', filename=\'DeformedShape.png\')}}" id = "container" style = "float:right"/>\n')
f.write('<h1>Joints Table</h1>\n')
f.close()
headerz = ['number','x','y','bound_x','bound_y','force_x','force_y']
nodesDF = pd.DataFrame(nodes, columns = headerz)
nodesDF.to_html(open('/workspaces/99926306/~/project/2DTruss_Solver/templates/index.html', 'a'), index=False, classes="table table-striped table-dark", justify = 'center')

f = open('/workspaces/99926306/~/project/2DTruss_Solver/templates/index.html', 'a')
f.write('\n\n')
f.write('<h1>Beam Table</h1>\n')
f.close()

headerz2 = ['number','node i','node j','material','Area']
barsDF = pd.DataFrame(bars, columns = headerz2)
barsDF.to_html(open('/workspaces/99926306/~/project/2DTruss_Solver/templates/index.html', 'a'), index=False, classes="table table-striped table-dark", justify = 'center')

f = open('/workspaces/99926306/~/project/2DTruss_Solver/templates/index.html', 'a')
f.write('\n\n')
f.write('<h1>Materials Table</h1>\n')
f.close()

headerz3 = ['number','material']
materialsDF = pd.DataFrame(materials, columns = headerz3)
materialsDF.to_html(open('/workspaces/99926306/~/project/2DTruss_Solver/templates/index.html', 'a'), index=False, classes="table table-striped table-dark", justify = 'center')
f = open('/workspaces/99926306/~/project/2DTruss_Solver/templates/index.html', 'a')
f.write('\n\n')
f.write('<h1>Displacements</h1>\n')


########################
# PLOT STRUCTURE
########################
Nn = len(nodes) # Number of nodes which is the number of rows in the nodes array
Ne = len(bars) # Number of elements which is the number of rows in the bar array

# Plot structure, creates the window size of the plot to scale with the
# size of the truss being analyzed
plt.rcParams.update({'axes.facecolor':'#121212'})
plt.rcParams["figure.figsize"] = (9,9)
#plt.figure(4) # Create an empty figure
#plt.plot(x,y)
#plt.show()

range_x = max(nodes[:,1]) - min(nodes[:,1]) #assigns range of x to the maximum x distance in the column - the smallest x distance value in the column
range_y = max(nodes[:,2]) - min(nodes[:,2]) #assigns range of y to the maximum value in the column - the smallest value in the column
range_ = max(range_x,range_y) #assigns the range to the larger range value between x and y
delta_x = (1.3 * range_ - range_x) #assign delta x to be 1.2* the largest distance - the largest x distance
delta_y = (1.3 * range_ - range_y)  #assign delta y to be 1.2* the largest distance - the largest y distance
fig, ax = plt.subplots(facecolor = '#FFFFFF', frameon=False)
ax.plot(nodes[:,1], nodes[:,2], 'o', color = '#d9534f', markersize = 8, label = 'Joints')  #Plot nodes by indexing all rows of column 2 and assigning them to x, and indexing all rows of column 3 and assigning them to y
plt.title('External Force and Boundry Conditions Diagram', color = '#84C9FB', fontsize = '18')
ax.set_xlabel('X-axis ')
ax.set_ylabel('Y-axis ')
ax.set_facecolor('#121212')
axiscolor = '#45A29E'
ax.xaxis.label.set_color(axiscolor)        #setting up X-axis label color to yellow
ax.yaxis.label.set_color(axiscolor)          #setting up Y-axis label color to blue

ax.tick_params(axis='x', colors=axiscolor)    #setting up X-axis tick color to red
ax.tick_params(axis='y', colors=axiscolor)  #setting up Y-axis tick color to black

ax.spines['left'].set_color(axiscolor)        # setting up Y-axis tick color to red
ax.spines['top'].set_color(axiscolor)
ax.spines['right'].set_color(axiscolor)
ax.spines['bottom'].set_color(axiscolor)
plt.xlim(np.array([np.amin(nodes[:,1]) - 0.5 * delta_x,np.amax(nodes[:,1]) + 0.5 * delta_x]))
#sets x axis to be between the smallest value in the x column - 0.5*delta x
#and the largest x value in the column + 0.5* delta x

plt.ylim(np.array([np.amin(nodes[:,2]) - 0.5 * delta_y,np.amax(nodes[:,2]) + 0.5 * delta_y]))
#sets y axis to be between the smallest value in the y column - 0.5*delta y
#and the largest y value in the column + 0.5* delta y

bar_nodes = np.zeros((Ne,4))
bar_L_S = np.zeros((Ne,2))
for i in range(Ne):
# Compute bar nodes locations, #bar_nodes(i,:)# creates a new matrix
# with i number of rows and : means as many columnms as specified
# (which is four in this case)
# nodes(bars(i,2),2) indexes the ith row of the node_i column of the bar matrix and then
# indexes x distance in the node matrix
    bar_nodes[i,:] = np.array([nodes[bars[i,1],1],nodes[bars[i,1],2],nodes[bars[i,2],1],nodes[bars[i,2],2]])
    # Draw bars
    plt.plot(np.array([bar_nodes[i,0],bar_nodes[i,2]]),np.array([bar_nodes[i,1],bar_nodes[i,3]]),'-',color = '#45A29E', label = 'Beams')
    #plot([a b], [c d]) draws a line from (a,c) to (b,d)
    # a = bar_nodes(i,1),     b = bar_nodes(i,3),     c = bar_nodes(i,2),
    # d = bar_nodes(i,4)
    # Compute bar length and angle

    bar_L_S[i,:] = np.array([np.sqrt((bar_nodes[i,2] - bar_nodes[i,0]) ** 2 + (bar_nodes[i,3] - bar_nodes[i,1]) ** 2), np.arctan((bar_nodes[i,3] - bar_nodes[i,1]) / (bar_nodes[i,2] - bar_nodes[i,0]))])
    #bar length = sqrt((bar_nodes(i,3)-bar_nodes(i,1))^2+(bar_nodes(i,4)-bar_nodes(i,2))^2)
    #bar slope = atan((bar_nodes(i,4)-bar_nodes(i,2))/(bar_nodes(i,3)-bar_nodes(i,1)))
    #bar_L_S(i,1) = lengths
    # bar_L_S(i,2) = angle


#the following will plot the forces and the arrows showing the direction of the force
for i in range(Nn):
    c = nodes[i,6]

    if np.abs(c) > 0:
        start = np.array([nodes[i,1], nodes[i,2]])

        p = 0.075 * (range_x + range_y)
        if nodes[i,6] < 0:
            p = p * -1

        ends = np.array([nodes[i,1], nodes[i,2] + p])
        diff = ends - start
        plt.quiver(start[0],start[1],diff[0],diff[1], color = '#d9534f', headwidth=1.75, headlength = 3, linewidth = 2, zorder = 5, label = 'Force')
        plt.text(ends[0],ends[1], nodes[i,6], color = '#d9534f', fontsize = 12)

for i in range(Nn):
    plt.plot(np.array([nodes[i,1] - 0.03 * nodes[i,3] * range_,nodes[i,1] + 0.03 * nodes[i,3] * range_]),np.array([nodes[i,2],nodes[i,2]]), color = '#FAFAFA', linewidth = 2, label = 'X - Fixed')
    # plot([a b],[c d]) plots a line from (a,c) to (b,d), in this
    #a = the ith row of x - 0.03 * the restraint x * the max distance
    #b = the ith row of x + 0.03 * the restraint x * the max distance
    #c = the ith row of y
    #d = the ith row of y
    plt.plot(np.array([nodes[i,1],nodes[i,1]]),np.array([nodes[i,2] - 0.03 * nodes[i,4] * range_,nodes[i,2] + 0.03 * nodes[i,4] * range_]), color = '#FAFAFA', linewidth = 2, label = 'Y - Fixed')
    # a = ith row of x distance
    # b = ith row of x distance
    # c = ith row of y distance - 0.03 * the y restraint * the max distance
    # d = ith row of y distance + 0.03 * the y restraint * the max distance


#the following will plot the forces and the arrows showing the direction of the force
for i in range(Nn):
    b = nodes[i,5]

    if np.abs(b) > 0:
        start = np.array([nodes[i,1], nodes[i,2]])
        p = 0.075 * range_x

        if nodes[i,5] < 0:
            p = p * -1

        ends = np.array([nodes[i,1] + p, nodes[i,2]])
        diff = ends - start
        plt.quiver(start[0],start[1],diff[0],diff[1], color = '#d9534f', headwidth=1.75, headlength = 3, linewidth = 2, zorder = 5)
        plt.text(ends[0],ends[1], nodes[i,5], color = '#d9534f', fontsize = 12)

saveto = '/workspaces/99926306/~/project/2DTruss_Solver/static/'
plt.savefig(saveto + 'ExternalForce.png')
plt.close()
plt.clf()

plt.rcdefaults()
plt.rcParams.update({'axes.facecolor':'#121212'})
plt.rcParams["figure.figsize"] = (9,9)

plt.plot(nodes[:,1], nodes[:,2], 'ro')  #Plot nodes by indexing all rows of column 2 and assigning them to x, and indexing all rows of column 3 and assigning them to y
fig, ax = plt.subplots(facecolor = '#FFFFFF', frameon=False)
ax.plot(nodes[:,1], nodes[:,2], 'o', color = '#d9534f', markersize = 8, label = 'Joints')  #Plot nodes by indexing all rows of column 2 and assigning them to x, and indexing all rows of column 3 and assigning them to y
plt.title('Deformed Shape Diagram', color = '#84C9FB', fontsize = '24')
ax.set_xlabel('X-axis ')
ax.set_ylabel('Y-axis ')
ax.set_facecolor('#121212')
axiscolor = '#45A29E'
ax.xaxis.label.set_color(axiscolor)        #setting up X-axis label color to yellow
ax.yaxis.label.set_color(axiscolor)          #setting up Y-axis label color to blue

ax.tick_params(axis='x', colors=axiscolor)    #setting up X-axis tick color to red
ax.tick_params(axis='y', colors=axiscolor)  #setting up Y-axis tick color to black

ax.spines['left'].set_color(axiscolor)        # setting up Y-axis tick color to red
ax.spines['top'].set_color(axiscolor)
ax.spines['right'].set_color(axiscolor)
ax.spines['bottom'].set_color(axiscolor)

plt.xlim(np.array([np.amin(nodes[:,1]) - 0.5 * delta_x,np.amax(nodes[:,1]) + 0.5 * delta_x]))
#sets x axis to be between the smallest value in the x column - 0.5*delta x
#and the largest x value in the column + 0.5* delta x

plt.ylim(np.array([np.amin(nodes[:,2]) - 0.5 * delta_y,np.amax(nodes[:,2]) + 0.5 * delta_y]))
#sets y axis to be between the smallest value in the y column - 0.5*delta y
#and the largest y value in the column + 0.5* delta y

for i in range(Ne):
# Compute bar nodes locations, #bar_nodes(i,:)# creates a new matrix
# with i number of rows and : means as many columnms as specified
# (which is four in this case)
# nodes(bars(i,1),1) indexes the ith row of the node_i column of the bar matrix and then
# indexes x distance in the node matrix
    bar_nodes[i,:] = np.array([nodes[bars[i,1],1],nodes[bars[i,1],2],nodes[bars[i,2],1],nodes[bars[i,2],2]])
    # Draw bars
    plt.plot(np.array([bar_nodes[i,0],bar_nodes[i,2]]),np.array([bar_nodes[i,1],bar_nodes[i,3]]),'#45A29E')
    #plot([a b], [c d]) draws a line from (a,c) to (b,d)
    # a = bar_nodes(i,0),     b = bar_nodes(i,2),     c = bar_nodes(i,1),
    # d = bar_nodes(i,3)
    # Compute bar length and angle
    bar_L_S[i,:] = np.array([np.sqrt((bar_nodes[i,2] - bar_nodes[i,0]) ** 2 + (bar_nodes[i,3] - bar_nodes[i,1]) ** 2), np.arctan((bar_nodes[i,3] - bar_nodes[i,1]) / (bar_nodes[i,2] - bar_nodes[i,0]))])

    #bar length = sqrt((bar_nodes(i,2)-bar_nodes(i,0))^2+(bar_nodes(i,3)-bar_nodes(i,1))^2)
    #bar slope = atan((bar_nodes(i,3)-bar_nodes(i,1))/(bar_nodes(i,2)-bar_nodes(i,0)))
    #bar_L_S(i,1) = lengths
    # bar_L_S(i,2) = angle

######################################
#FEM EQUATIONS  (SOVLING DISPLACEMENT)
######################################

## Solving for displacements, so displacement matrix = force_vector \ global_stiffness_matrix
# so need to write force_vector and global stiffness

## for the force vector
#the force matrix must be double the nodes because there is x and y
#directions

F = np.zeros((2*Nn,1))
X = np.zeros((Nn,2))
for i in range(Nn):
    X[i] = [nodes[i,5], nodes[i,6]]
k = 0
for i in X:
    for j in i:
        F[k]= j
        k += 1

#need local matrix first
#need A,E, L, and angles
#need C^2 , S^2, and C*S
#the number of local stiffness matrices must match the number of elements
#aka number of bars
A= np.zeros(shape = (Ne,1))
E= np.zeros(shape = (Ne,1))
L= np.zeros(shape = (Ne,1))
C= np.zeros(shape = (Ne,1))
S= np.zeros(shape = (Ne,1))

for i in range(Ne):
    A[i] = bars[i,4]
    E[i] = materials[bars[i,3],1]
    L[i] = bar_L_S[i,0]
    C[i] = math.cos(bar_L_S[i,1])
    S[i] = math.sin(bar_L_S[i,1])

C2 = C ** 2
S2 = S ** 2
CS = np.multiply(C,S)
# now create local matrices that can be indexed to create global
k_local = np.zeros(shape = (4, 4, Ne))
## next create global matrix by adding proper indexes together
# for the global stiffness
for i in range(Ne):
    k_localss = ((A[i] * E[i]) / (L[i])) * np.array([ [C2[i],        CS[i],      -C2[i],       -CS[i]    ],     #index node "node# = nodes(row, 1)"
                                                      [CS[i],        S2[i],      -CS[i],       -S2[i]    ],     #index force_x "force_x = nodes(row,6)"
                                                      [-C2[i],      -CS[i],       C2[i],        CS[i]    ],     #index force_ "force_y = nodes(row,7)"
                                                      [-CS[i],      -S2[i],       CS[i],        S2[i]]   ])
    k_locals = k_localss.reshape(4,4)
    k_local[:,:,i] = k_locals

K = np.zeros((2*Nn, 2*Nn))
#stiffness matrix should be square matrix twice the size of the number of nodes to account for x and y directions

#now assemble each matrice element by element
for i in range(Ne):
   n = bars[i,1:3]
   K[2*n[0]:2*n[0]+2, 2*n[0]:2*n[0]+2] = K[2*n[0]:2*n[0] + 2,2*n[0]:2*n[0]+2] + k_local[0:2,0:2, i]
   K[2*n[0]:2*n[0]+2, 2*n[1]:2*n[1]+2] = K[2*n[0]:2*n[0] + 2,2*n[1]:2*n[1]+2] + k_local[0:2,2:4, i]
   K[2*n[1]:2*n[1]+2, 2*n[0]:2*n[0]+2] = K[2*n[1]:2*n[1] + 2,2*n[0]:2*n[0]+2] + k_local[2:4,0:2, i]
   K[2*n[1]:2*n[1]+2, 2*n[1]:2*n[1]+2] = K[2*n[1]:2*n[1] + 2,2*n[1]:2*n[1]+2] + k_local[2:4,2:4, i]

d = np.linalg.lstsq(K[0:2,0:2],F[0:2], rcond = -1)
d=d[0]
d=d.reshape(2,1)
###########################
# APPLY BOUNDARY CONDITIONS
total_boundries = sum(sum(nodes[:,3:5]))
K_reduced = np.zeros((2*Nn, 2*Nn))

for i in range(Nn):
    if nodes[i,3] == 0:
        K_reduced[2 * i ,2 * i] = 1
    if nodes[i,4] == 0:
        K_reduced[2 * i + 1,2 * i + 1] = 1


for i in range(len(K_reduced)-total_boundries):
    if sum(K_reduced[:,i]) == 0:
        K_reduced = np.delete(K_reduced,i,axis=1)
if sum(K_reduced[:,0]) == 0:
    K_reduced = np.delete(K_reduced, 0, axis = 1)

Kt = np.transpose(K_reduced)
K = np.dot(Kt, np.dot(K, K_reduced))
F = np.dot(Kt, F)

########################
# SOLVE PROBLEM

d = np.linalg.lstsq(K,F, rcond = -1)
d = d[0]
d = np.dot(K_reduced, d)
d1 = d[0:None:2,:]
d2 = d[1:None:2,:]
d = np.concatenate((d1,d2), axis = 1)

f.close()
headerz4 = ['X-Displacement','Y-displacement']
materialsDF = pd.DataFrame(d, columns = headerz4)
materialsDF.to_html(open('/workspaces/99926306/~/project/2DTruss_Solver/templates/index.html', 'a'), index=False, classes="table table-striped table-dark", justify = 'center')

f = open('/workspaces/99926306/~/project/2DTruss_Solver/templates/index.html', 'a')
f.write('{% endblock %}')
f.close()


########################
# PLOT DEFORMED SHAPE
########################
scale_factor=20
old_nodes = nodes[:,1:3]
new_nodes = old_nodes + d * scale_factor

plt.plot(new_nodes[:,0], new_nodes[:,1], 'co')

for i in range(Ne):
# Compute bar nodes locations, #bar_nodes(i,:)# creates a new matrix
# with i number of rows and : means as many columnms as specified
# (which is four in this case)
# nodes(bars(i,2),2) indexes the ith row of the node_i column of the bar matrix and then
# indexes x distance in the node matrix
    bar_nodes[i,:] = np.array([new_nodes[bars[i,1],0],new_nodes[bars[i,1],1],new_nodes[bars[i,2],0],new_nodes[bars[i,2],1]])
    # Draw bars
    plt.plot(np.array([bar_nodes[i,0],bar_nodes[i,2]]),np.array([bar_nodes[i,1],bar_nodes[i,3]]), linestyle = '--', color = '#45A29E')
    #plot([a b], [c d]) draws a line from (a,c) to (b,d)
    # a = bar_nodes(i,1),     b = bar_nodes(i,3),     c = bar_nodes(i,2),
    # d = bar_nodes(i,4)
    # Compute bar length and angle
    bar_L_S[i,:] = np.array([np.sqrt((bar_nodes[i,2] - bar_nodes[i,0]) ** 2 + (bar_nodes[i,3] - bar_nodes[i,1]) ** 2), np.arctan((bar_nodes[i,3] - bar_nodes[i,1]) / (bar_nodes[i,2] - bar_nodes[i,0]))])

saveto = '/workspaces/99926306/~/project/2DTruss_Solver/static/'
plt.savefig(saveto + 'DeformedShape.png')
plt.close()
plt.clf()

@app.route("/")
def index():
    """Show portfolio of stocks"""
    return render_template("index.html")


@app.route("/example")
def example():
    """Show history of transactions"""
    return render_template("example.html")


