import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, flash, redirect, render_template, request
from tempfile import mkdtemp
import pandas as pd
import math
##################################
#BEGIN USER INPUT MATRICES SECTION            COMMENT OR UN-COMMENT BY HIGHLIGHTING THE TEXT AND USING ("CTRL + /") YOU ARE IN VS CODE, otherwise manually by deleting #
##################################
# Be consistent with units: for example kips, inches, ksi (or N, mm, MPa)


# Material data (number needs to be increasing: 0, 1, 2...)
#                    number     E (modulus of elasticity)
materials = np.array([[0,     20000   ],
                      [1,     2000    ],
                      [2,     4200   ]])

            #Creates a reference-able material matrix for E
            #Reference by indexing          "E = materials(row,2)"
            # "row" value must work for any size matrix

# Truss nodes data (number needs to be increasing: 0, 1, 2...)

# Boundary conditions (0 = unrestrained, 1 = restrained)



#################################################
#Uncomment the below matrices for example 1 (lines 34 - 60)
#################################################

#                   number      x       y     bound_x     bound_y     force_x     force_y

nodes =   np.array([[0,        0,      0,       1,          1,          0,           0   ],     #index node "node# = nodes(row, 0)"
                    [1,      120,      0,       0,          0,          0,         -20   ],     #index force_x "force_x = nodes(row,5)"
                    [2,      240,      0,       0,          0,          0,         -10   ],     #index force_ "force_y = nodes(row,6)"
                    [3,      360,      0,       0,          1,          0,           0   ],     #index boundary conditions
                    [4,       60,     90,       0,          0,          0,         -20   ],     #"bound_x = nodes(row,3)"
                    [5,      180,     90,       0,          0,         10,           0   ],     #"bound_y = nodes(row,4)"
                    [6,      300,     90,       0,          0,         10,           0   ]])    #"AllRows = nodes(:, column)"


# Truss element data (number needs to be increasing: 0, 1, 2...)


#                 number   node i    node j  material   Area

bars = np.array([[  0,       0,        1,        1,      5   ],     #index node i    "something = bars(row, 1)"
                   [1,       1,        2,        1,      5   ],     #index node j    "something = bars(row, 2)"
                   [2,       2,        3,        1,      5   ],     #index bar       "something = bars(row, 3)"
                   [3,       4,        5,        1,      5   ],     #for material    "material = bars(row, 4)"
                   [4,       5,        6,        1,      5   ],     #for area        "area = bars(row,5)"
                   [5,       0,        4,        1,      5   ],
                   [6,       1,        5,        1,      5   ],
                   [7,       2,        6,        1,      5   ],
                   [8,       1,        4,        1,      5   ],
                   [9,       2,        5,        1,      5   ],
                   [10,      3,        6,        1,      5  ]])


######################################################
#Uncomment the below matrices for example 2 (lines 68 - 77)
######################################################


# ##                   number     x        y    bound_x  bound_y  force_x    force_y
# nodes = np.array([[   0,        0,     0,     0,       0,         0,      -10],
#                    [  1,        0,    120,    1,       1,         0,        0],
#                    [  2,       120,   120,    1,       1,         0,        0],
#                    [  3,       120,    0,     1,       1,         0,        0]])

# ##                 number   node i  node j    material  Area
# bars = np.array([[   0,       0,      1,        2,        2   ],
#                  [   1,       0,      2,        2,        2   ],
#                  [   2,       0,      3,        2,        2   ]])



######################################################
#Uncomment the below matrices for example 3 (lines 86 - 98)
######################################################


# ####               number     x         y       bound_x  bound_y  force_x     force_y
# nodes = np.array([[  0,        0,       5*12,     1,       1,         0,        0  ],
#                   [   1,       0,         0,      1,       1,         0,        0  ],
#                   [   2,       30*12,     0,      1,       1,         0,        0  ],
#                   [   3,       30*12,   5*12,     1,       1,         0,        0  ],
#                   [   4,       15*12,   5*12,     0,       0,         0,       -10 ]])

# ###Truss element data (number needs to be increasing: 1, 2, 3...)
# ####                number  node i  node j   material    Area
# bars = np.array([[   0,       0,      4,        1,        10  ],
#                  [   1,       3,      4,        1,        10  ],
#                  [   2,       2,      4,        1,        20 ],
#                  [   3,       1,      4,        1,        20  ]])