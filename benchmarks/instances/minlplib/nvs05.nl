g3 0 1 0	# problem nvs05
 8 9 1 0 4	# vars, constraints, objectives, ranges, eqns
 8 1	# nonlinear constraints, objectives
 0 0	# network constraints: nonlinear, linear
 8 4 4	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 2 0 0	# discrete variables: binary, integer, nonlinear (b,c,o)
 26 4	# nonzeros in Jacobian, gradients
 3 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
b
0 1e-2 200
0 1e-2 200
0 1 200
0 1 200
3
3
3
3
x8
0 1
1 1
2 1
3 1
4 1
5 1
6 2
7 1
r
4 0
4 0
4 0
4 0
2 -1.36e4
2 -3e4
2 6e3
2 -.25
2 0
C0
o16
o3
n4243.28147100424
o2
v0
v1
C1
o16
o39
o0
o2
n.25
o5
v1
n2
o5
o0
o2
n.5
v2
o2
n.5
v0
n2
C2
o16
o3
o2
o0
n59405.9405940594
o2
n2121.64073550212
v1
v6
o2
o2
v0
v1
o0
o2
n8.33333333333333e-2
o5
v1
n2
o5
o0
o2
n.5
v2
o2
n.5
v0
n2
C3
o16
o3
o2
n.5
v1
v6
C4
o16
o39
o54
3
o5
v4
n2
o2
o2
o2
n2
v4
v5
v7
o5
v5
n2
C5
o16
o3
n5.04e5
o2
o5
v2
n2
v3
C6
o2
o2
n2.04744897959184e-2
o39
o2
o2
o2
o2
n1e15
o5
v3
n3
v2
v2
o5
v3
n3
o0
n1
o2
n-2.82346219657891e-2
v2
C7
o16
o3
n.21952
o2
o5
v2
n3
v3
C8
n0
O0 0
o0
o2
o2
n1.10471
o5
v0
n2
v1
o2
o2
o2
n4.811e-2
v2
v3
o0
n14
v1
k7
4
8
13
17
19
21
24
J0 3
0 0
1 0
4 1
J1 4
0 0
1 0
2 0
6 1
J2 5
0 0
1 0
2 0
5 1
6 0
J3 3
1 0
6 0
7 1
J4 3
4 0
5 0
7 0
J5 2
2 0
3 0
J6 2
2 0
3 0
J7 2
2 0
3 0
J8 2
0 -1
3 1
G0 4
0 0
1 0
2 0
3 0
