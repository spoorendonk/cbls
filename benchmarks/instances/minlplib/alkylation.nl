g3 0 1 0	# problem alkylation
 10 11 1 0 3	# vars, constraints, objectives, ranges, eqns
 6 1	# nonlinear constraints, objectives
 0 0	# network constraints: nonlinear, linear
 7 8 1	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0	# discrete variables: binary, integer, nonlinear (b,c,o)
 31 6	# nonzeros in Jacobian, gradients
 3 3	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
b
0 1e-6 5e3
0 1e-6 2e3
0 1e-6 1.6e4
0 1e-6 1.2e2
0 1e-6 2e3
0 3 12
0 1e-2 4
0 90 95
0 85 93
0 145 162
x10
0 3048
1 1745
2 1.2e4
3 1.1e2
4 1974
5 8
6 3.6
7 92.8
8 89.2
9 145
r
4 0
4 0
2 0
2 0
2 -57.425
2 57.425
4 0
2 -35.82
2 35.82
2 133
2 -133
C0
o16
o3
o2
n9.8e4
v3
o0
o2
v0
v6
o2
n1e3
v3
C1
o16
o3
o0
v2
v4
v1
C2
o2
o54
3
o2
n-6.67e-3
o5
v5
n2
o2
n.13167
v5
n1.12
v1
C3
o16
o2
o54
3
o2
n-6.67e-3
o5
v5
n2
o2
n.13167
v5
n1.12
v1
C4
o0
o2
n-3.8e-2
o5
v5
n2
o2
n1.098
v5
C5
o16
o0
o2
n-3.8e-2
o5
v5
n2
o2
n1.098
v5
C6
n0
C7
n0
C8
n0
C9
n0
C10
n0
O0 1
o2
o2
n6.3e-2
v0
v7
k9
4
8
9
10
12
17
20
24
27
J0 4
0 0
3 0
6 0
8 1
J1 4
1 0
2 0
4 0
5 1
J2 3
0 -.99
1 0
5 0
J3 3
0 1.01010101010101
1 0
5 0
J4 3
5 0
7 -.99
8 .325
J5 3
5 0
7 1.01010101010101
8 -.325
J6 3
0 -1.22
1 1
4 1
J7 2
6 -.9
9 -.222
J8 2
6 1.11111111111111
9 .222
J9 2
7 3
9 -.99
J10 2
7 -3
9 1.01010101010101
G0 6
0 0
1 -5.04
2 -3.5e-2
3 -10
4 -3.36
7 0
