# -*- coding: utf-8 -*-
"""
--------------
**graphs**
--------------

Description
-----------
Create Networkx test graphs with random edge weights.

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



#####################
## Import Packages ##
#####################

import networkx as nx
import numpy as np



##########################
## Test Graph Generator ##
##########################    

def grid_graph( size, max_weight ):
    """Generates a Networkx lattice graph with random edge weights.

    Parameters
    ----------
    size : integer
        The lattice will have `size` * `size` many nodes.

    max_weight : integer
        Weights will be drawn from a uniform distribution with support [0, `max_weight`].

    Returns
    -------
    path : tuple
        G: Networkx graph
        
        pos: graph layout in lattice configuration

        start: node 0 in `G`

        end: node (`size` * `size`) - 1 in `G`.
                
    Examples
    --------
    >>> G, pos, start, end = grid_graph( size = 2, max_weight = 10 )
    >>> print(nx.info(G))
    Name: grid_2d_graph
    Type: Graph
    Number of nodes: 4
    Number of edges: 4
    Average degree:   2.0000
    >>> pos
    {0: (0, 1), 1: (0, 0), 2: (1, 1), 3: (1, 0)}
    >>> start, end
    (0, 3)
    """    
  
    n = size 
    G = nx.grid_2d_graph( n, n )

    # Generate edge weights.
    for (u, v) in G.edges():
        G.edge[u][v]['weight'] = round( np.random.uniform( 0, max_weight, 1 )[0], 2 )

    # Label according to position.
    labels = dict( ( (i,j), i + (n-1-j) * n ) for i, j in G.nodes() )
    nx.relabel_nodes( G, labels, False )
    inds = labels.keys()
    vals = labels.values()
    inds = [( n - j - 1, n - i - 1 ) for i,j in inds]

    # Set outputs.
    pos = dict( zip( vals, inds ) )
    start = 0
    end = (n*n)-1
    
    return G, pos, start, end