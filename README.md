# 2D Truss Solver
#### Video Demo:  <https://youtu.be/sOajSZb3RlY>
#### Description:

Hello, my name is Garrett Manalo and for this CS50 project I created a 2D Truss Solver.
This program will solve any simple 2D truss using matrices and finite element methods.
It is written in python, jinja, html, and css. It requires main app.py, userinput.py, index.html, layout.html, and styles.css to run. As well as the static folder and templates folder. The program runs on flask framework.

The best explanation for how to use is to watch the youtube video, but I will try to explain here as well.
To begin, the user must understand what a 2D truss system is and have a problem in mind they want to solve,
if you are just trying to see how the program works, I have included examples in the userinput.py, simply un-comment the nodes and bars
that you are interested in viewing.
You can also change specific inputs on the given matrices such as the force_x and force_y, the area, and material to see different results for the same truss.

If the user has a new problem they want to implement, you must break the problem apart into its joints(nodes) and its beams(bars).
The joints(nodes) should be assigned incrementally from 0, 1, 2, 3.... and up. Next the x and y coordinates should be assigned.
Start by letting the bottom left most join be (0,0) and then use the known lengths of the beams(you must have these to solve) to determine
the next joints location. You also must know the boundary conditions which simply mean, is this joint fixed in place or can it move in x or move in y or move in both x and y. Simply set 1 or 0 in the appropriate array section (1 if it can move, 0 if it can not). The step is to also apply a force in the x or y direction.

The follow join array shows that their are 4 joints, join 1,2, and 3 are fixed in place and can not move in any direction, but join 0 can move in both directions. There is also a force applied downward in the y direction on joint 0.
The beam(or bar) array described the bars number, what joints is is connecting between, what material its made out of(modulus of elasticity), and its cross sectional area.
# ([[   0,        0,     0,     0,       0,         0,      -10],
# [  1,        0,    120,    1,       1,         0,        0],
# [  2,       120,   120,    1,       1,         0,        0],
# [  3,       120,    0,     1,       1,         0,        0]])


The following beam array described that beam 0 connects between joints 0 and 1, beam 1 connects between joints 0 and 2, and beam 2 connects between joint 0 and 3. It also described that the material is material number 2 in the materials array, and that each bar has a cross sectional area of 2.
number    node i   node j  material    Area
# ([[   0,       0,      1,        2,        2   ],
# [   1,       0,      2,        2,        2   ],
# [   2,       0,      3,        2,        2   ]])

The materials array can be added to as much as needed, simply list the next number of material and its modulus of elasticity.
Material 0 has a modulus of 20000
number     E (modulus of elasticity)
# ([[0,     20000   ],
# [1,     2000    ],
# [2,     4200   ]])

You may be wondering about what units all of these matrices use, and the answer is anything as long as you are consistent, for instance
if you want to you lbs and inches, make sure the x and y coordinates are in terms of inches, the force in terms of lbs, the E- modulus in psi (lbs per square inch) and the area should be in inches squared.

If you wanted lbs, ft then make sure the relevant numbers are converted to ft instead of inches.
If you want Meters and Newtons (metric system), then make sure x and y are in meters, area meters squares, force in newtons, E in newtons per meter.

Once you have correctly interpreted the given 2D truss and placed its data into the matrices properly, you can run the program with "flask run" command
or run it with "python app.py" command.

The program will create and save two images, one called ExternalForce.png and the other called DeformedShape.png, these are the two graphs that are created.
The program will solve for how the truss deforms based on a given force. It will display how much each joint moved in the x an dy direction.
The information entered into the matrices will also be displayed.

The app.py file is the bulk of the code, it contains all the code to write the index.html as well as all the code that does the math, and reders the html files. The theory used to solve for the displacements is called finite element method.

The index.html file contains the tables and images formated using bootstrap and custom style.css file.
The layouts.html file was recycled from week 9 homework and contains the skeleton of the html pages.
The userinput.py is a seperate file used to allow someone to enter their own 2D truss.
Finally, style.css simply customizes the look of the html page.
The design was made to match my homepage design.