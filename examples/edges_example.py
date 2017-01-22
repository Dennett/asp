# -*- coding: utf-8 -*-
"""
-----------------
**edges_example**
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
## Edge Readme ##
#################

# Set parameters
sigma = .2
pdf = 'truncated'

# Create test graph, path, and node distribution
G, pos, start, end = asp.grid_graph( size = 3, max_weight = 2 )
pts = {1:.2,0:.3,3:.5}

# Initialize stochastic shortest path distributions for the family
edge_distr = asp.edge_distribution( G, pts, pdf, sigma )

# Generate distributions for multiple algorithms
spd_dict = edge_distr.get_edge_distr( path_alg = 'k', path_k = 2, alg = 'spd' )
nmi_dict = edge_distr.get_edge_distr( path_alg = 'k', path_k = 2, alg = 'nmi' )
mcs_dict = edge_distr.get_edge_distr( path_alg = 'k', path_k = 2, alg = 'mcs' )

# Analyze distributions & plot
distrs = [ spd_dict, nmi_dict, mcs_dict ]
mydistrs_dataframe = edge_distr.gather_dfs( distrs, cols = ['spd', 'nmi', 'mcs'] )
dist = edge_distr.get_col_dists( mydistrs_dataframe )
edge_distr.plot( pos=pos, edge_df=mydistrs_dataframe, col='spd', scale=10 )












