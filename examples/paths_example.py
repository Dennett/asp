# -*- coding: utf-8 -*-
"""
-----------------
**paths_example**
-----------------

Description
-----------
Usage Example

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