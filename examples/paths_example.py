# -*- coding: utf-8 -*-
"""
-----------------
**paths_example**
-----------------

Description
-----------
Example usage

Copyright (C) 2017 Michael W. Ramsey <michael.ramsey@gmail.com>

License
--------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Details
-------
"""


import networkx as nx
import pandas as pd
import asp

#################
## Path Readme ##
#################



# Set parameters
sigma = .2
pdf = 'truncated'

# Create test graph
G, pos, start, end = asp.grid_graph( size = 2, max_weight = 2 )

# Create family of stochastic paths
myfam = asp.path_family( G, start, end, pdf, sigma )
paths = myfam.get_paths( alg = 'k', k = 2 )

# Initialize stochastic shortest path distributions for the family
mydistr = asp.family_distribution( myfam )

# Generate distributions for multiple algorithms
spd_dict = mydistr.get_distr( paths, alg = 'spd' )[0]
nmi_dict = mydistr.get_distr( paths, alg = 'nmi' )[0]
mcs_dict = mydistr.get_distr( paths, alg = 'mcs' )[0]

# Analyze distributions
distrs = mydistr.keys_to_nodes(  [ spd_dict, nmi_dict, mcs_dict ] )
mydistrs_dataframe = mydistr.gather_dicts( distrs, cols = [ 'spd', 'nmi', 'mcs'] )
dist = mydistr.get_col_dists( mydistrs_dataframe )


#################
## Edge Readme ##
#################

#sigma = .2
#pdf = 'truncated'
#
##G = nx.read_edgelist('data/dimacs_nyc_edges.csv', nodetype=int, delimiter = ',', data=(('weight',float),))
##n = pd.read_csv('data/dimacs_nyc_nodes.csv') 
#
#G, pos, start, end = grid_graph( size = 2, max_weight = 2 )
#
#pts = [1,0,3]
#p = {1:.2,0:.3,3:.5}
#
#edge_distr = edge_distribution( G, pts, p, pdf, sigma )
#
#spd_e_distr = edge_distr.get_edge_distr( path_alg = 'k', path_k = 2, path_tol = 5e-1, alg = 'spd', itr = 1, auto = True, N = 1e2, incr = 1e3, m = .1, tol = 5e-2, M = 5e5 )
#spd_auto_e_distr = edge_distr.get_edge_distr( path_alg = 'k', path_k = 2, path_tol = 5e-1, alg = 'spd', itr = 1, auto = True, N = 1e2, m = .1, tol = 5e-2, M = 5e5 )
#nmi_e_distr = edge_distr.get_edge_distr( path_alg = 'all', path_k = 2, path_tol = 5e-1, alg = 'nmi', itr = 1, epsrel = 1e-4, epsabs = 0 )
#mcs_e_distr = edge_distr.get_edge_distr( alg = 'mcs', itr = 1, n = 10000 )
#
## Analyze Means
#distrs = [ spd_e_distr, spd_auto_e_distr, nmi_e_distr, mcs_e_distr ]
#mydistrs_dataframe = edge_distr.ndf_2_1df( distrs, cols = ['spd', 'spd_auto', 'nmi', 'mcs'] )
#dist = edge_distr.get_col_dists( mydistrs_dataframe )