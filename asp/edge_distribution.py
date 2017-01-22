# -*- coding: utf-8 -*-
"""
---------------------
**edge_distribution**
---------------------

Description
-----------
Create a distribution for the probability an edge is along the shortest path.

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
import matplotlib.pyplot as plt
from itertools import combinations
import collections as coll
import pandas as pd
from asp.family_distribution import family_distribution
from asp.path_family import path_family
  
    

########################
## Edge Distributions ##
########################



class edge_distribution:
    """Class for computing and storing edge distributions.

    Parameters
    ----------
    G : Networkx graph
        Graph from which this path will be derived.
        The nodes, edges, and edge weights below must exist in this graph.
        The weights MUST be labeled 'weight'.
        Asp has only be tested on undirected graphs.

    pts : dictionary
        Keys are node IDs from G and values are the probability the node should
        be a start or end point.
        
    pdf : string
        There are two options: 'normal' for the Gaussian Normal Distribution, 
        or 'truncated' for the Truncated Gaussian Normal Distribution. This 
        defines the amount of noise the edge weights in `G` have. Select 
        'normal' if negative edge weights are allowed. Select 'truncted' edge 
        weights must be non-negative. (CURRENTLY ONLY 'truncated' WORKS)   

    sigma : float
        Defines the standard deviation for the noise pdf.

    Notes
    -----
    Extending estimates for :math:`p^{*}` to edge traversal probabilities arises 
    naturally in the simulation algorithm since edge traversal counts can 
    be easily obtained from path traversals. Such accounting does not come 
    easily with Asp. However, the probability of an edge being traversed 
    can be computed from the probabilities of the paths containing it.
    
    For example, suppose there is a distribution over the nodes giving the 
    probability :math:`P( (i,j) = (s,e))` each pair of nodes would be selected as 
    endpoints :math:`(s,e)`. Then, given any edge :math:`(u,v)`, the probability :math:`(u,v)` 
    is traversed could be defined as:

    .. math:: P( (u,v) \\text{ traversed } ) = \\sum_{(u,v) \\in \\pi_{i,j}} p^{*}_{\\pi_{i,j}} P( (i,j) = (s,e))).

    Examples
    --------
    >>> G, pos, start, end = grid_graph( size = 2, max_weight = 2 )
    >>> myedgedistr = edge_distribution( G, pts={1:.2,0:.3,3:.5}, pdf='truncated', sigma=.2 )
    """
    
    def __init__( self, G, pts, pdf, sigma ):

        # Define path attributes.
        self.G = G
        self.pts = pts
        self.pdf = pdf
        self.sigma = sigma   
        

     
    def summary( self ):
        '''Print basic object attributes.
        
        Examples
        --------

        >>> myedgedistr.summary()
        Parent Graph Info:
        Name: grid_2d_graph
        Type: Graph
        Number of nodes: 9
        Number of edges: 12
        Average degree:   2.6667
        Start node = 0
        End node = 8
        Noise PDF = truncated
        Sigma = 0.2
        '''

        print('Parent Graph Info:')
        print(nx.info( self.G ))
        print('Points of Interest', self.pts)
        print('Noise PDF =', self.pdf)
        print('Sigma =', self.sigma)
        
        
        
    def get_edge_distr( self, path_alg = 'auto', path_k = 2, path_tol = 5e-1, alg = 'spd', itr = 1, auto = True, N = 1e2, m = .1, tol = 5e-2, M = 5e5, n = 1000, **kwargs ):
        '''Generate the edge distribution.
   
        Parameters
        ----------
        path_alg : string        
            Choose to generate the family of paths from a three options: 
            'all' (find all paths between endpoints), 'k' (find only the k-shortest paths),
            and 'auto' (find the (i*k)-shortest paths where i is automatically)
            determined. 
            
        path_k : integer, optional - depending on alg
            Paths are selected in chunks of k until a selection criteria is
            met. For path_alg = 'auto' or 'k'.
            
        path_tol : float, optional, optional - depending on alg
            Threshold for selecting k. For path_alg = 'auto'.

        alg : string
            Choose to generate the edge distribution from a three options:
            'spd' (estimate via Sampling Paths' Distributions), 
            'nmi' (estimate via NuMeric Integration), and 
            'mcs' (estimate via Monte Carlo Simulation)

        paths : dict        
            Dictionary of path objects. Path names MUST be consecutive integers
            starting at zero.
            
        itr : integer, optional
            Number of trials to run.
            
        auto : boolean, optional - depending on alg
            Toggles whether or not to continue drawing sample mins until the
            max marginal change is under less than tol.

        N : integer, optional - depending on alg
            Number of samples to draw from each path's length distribution.
            For alg = 'spd'.
            
        m : integer, optional - depending on alg
            Size of the window to compute the max marginal change on drawing 
            more samples. For alg = 'spd' and auto = True.      

        tol : integer, optional - depending on alg
            Threshold for whether or not to continue drawing sample mins until 
            the max marginal change is under less than tol. For alg = 'spd' and 
            auto = True.

        M : integer, optional - depending on alg
            The maximum number of draws to complete even if max marginal change
            is greater than tol. For alg = 'spd' and auto = True.       

        n : integer, optional - depending on alg        
            Number of draws to make for alg = 'mcs'. 
        
        **kwargs : dict
            scipy.integrate.quad kwargs.   

        Returns
        -------
        df : Pandas DataFrame
            index : tuples of nodes representing edges.
            prob : The estimated probability the edge will be in a shortest 
            path.
        
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 2, max_weight = 2 )
        >>> myedgedistr = edge_distribution( G, pts={1:.2,0:.3,3:.5}, pdf='truncated', sigma=.2 )
        >>> spd_e_distr = edge_distr.get_edge_distr( path_alg = 'k', path_k = 2, alg = 'spd', itr = 1, auto = False, N = 1e2 )       
        >>> spd_e_distr.head()
                prob
        (0, 1)  0.21
        (1, 3)  0.25
        '''  

        # Set variables for brevity.
        c = coll.Counter()
        pts = self.pts
        G = self.G
        pdf = self.pdf
        sigma = self.sigma
        pairs = combinations(pts.keys(), 2)
        
        # Loop through nodes and update edge probabilities.
        for v in pairs:
            prob_pair = pts[v[0]]*pts[v[1]]
            myfam = path_family( G, v[0], v[1], pdf, sigma )
            
            paths = myfam.get_paths( alg = path_alg, k = path_k, tol = path_tol )
            
            mydistr = family_distribution( myfam )
            distr, var, err_est, t_est = mydistr.get_distr( paths, alg = alg, itr = itr, auto = auto, N = N, m = m, tol = tol, M = M, n = n , **kwargs )
       
            for path in distr.keys():
                u = dict.fromkeys(path.edges, prob_pair*distr[path])
                c.update( u )
        
        # Format output.
        df = pd.DataFrame.from_dict(c, orient='index', dtype=float)
        df.columns = ['prob']
        
        return df



    def gather_dfs( self, dfs, cols ):
        '''Converts list of dataframes to one dataframe, merged on keys, with 
        the supplied column names.
        
        Parameters
        ----------
        dicts : list
            Iterable of dictionaries.

        cols : list        
            List of column names.

        Returns
        -------
        df : DataFrame
            Columns are values associated with each dictionary, indexed by the
            union of all the dictionaries keys.
            
        Examples
        --------
        >>> # Setup
        >>> G, pos, start, end = grid_graph( size = 2, max_weight = 2 )
        >>> myedgedistr = edge_distribution( G, pts={1:.2,0:.3,3:.5}, pdf='truncated', sigma=.2 )
        >>> # Generate output for multiple algorithms
        >>> spd_e_distr = myedgedistr.get_edge_distr( path_alg = 'k', path_k = 2, alg = 'spd', itr = 1, auto = False, N = 1e2 )          
        >>> spd_auto_e_distr = myedgedistr.get_edge_distr( path_alg = 'k', path_k = 2, path_tol = 5e-1, alg = 'spd', itr = 1, auto = True, N = 1e2, m = .1, tol = 5e-2, M = 5e5 )
        >>> nmi_e_distr = myedgedistr.get_edge_distr( path_alg = 'all', path_k = 2, path_tol = 5e-1, alg = 'nmi', itr = 1, epsrel = 1e-4, epsabs = 0 )
        >>> mcs_e_distr = myedgedistr.get_edge_distr( alg = 'mcs', itr = 1, n = 10000 )
        >>> # Gather distributions
        >>> distrs = [ spd_e_distr, spd_auto_e_distr, nmi_e_distr, mcs_e_distr ]
        >>> mydistrs_dataframe = myedgedistr.gather_dfs( distrs, cols = ['spd', 'spd_auto', 'nmi', 'mcs'] )
        >>> mydistrs_dataframe.head()
                 spd  spd_auto       nmi      mcs
        (0, 1)  0.06      0.06  0.060000  0.09612
        (0, 2)  0.25      0.25  0.249785  0.14327
        (1, 0)  0.10      0.10  0.099785  0.02939
        (1, 3)  0.00      0.00  0.000215  0.10673
        (2, 3)  0.25      0.25  0.249785  0.14327
        '''
        
        # Loop through DataFrames and merge on index.       
        df = None
        for d in dfs:
            if not isinstance(df, pd.DataFrame):              
                df = d
            elif isinstance(df, pd.DataFrame):           
                df = pd.merge(df, d, left_index = True, right_index = True, how = 'outer')             
 
        # Format output.  
        df.columns = cols
        df = df.fillna(0)

        return df



    def get_col_dists( self, distrs_dataframe ):
        '''Computes pairwise vector euclidean distance between columns.
        
        Parameters
        ----------
        distrs_dataframe : list
            Iterable of DataFrames.

        Returns
        -------
        df : DataFrame
            In distance matrix configuration.
            
        Examples
        --------
        >>> # Setup
        >>> G, pos, start, end = grid_graph( size = 2, max_weight = 2 )
        >>> myedgedistr = edge_distribution( G, pts={1:.2,0:.3,3:.5}, pdf='truncated', sigma=.2 )
        >>> # Generate output for multiple algorithms
        >>> spd_e_distr = myedgedistr.get_edge_distr( path_alg = 'k', path_k = 2, alg = 'spd', itr = 1, auto = False, N = 1e2 )          
        >>> spd_auto_e_distr = myedgedistr.get_edge_distr( path_alg = 'k', path_k = 2, path_tol = 5e-1, alg = 'spd', itr = 1, auto = True, N = 1e2, m = .1, tol = 5e-2, M = 5e5 )
        >>> nmi_e_distr = myedgedistr.get_edge_distr( path_alg = 'all', path_k = 2, path_tol = 5e-1, alg = 'nmi', itr = 1, epsrel = 1e-4, epsabs = 0 )
        >>> mcs_e_distr = myedgedistr.get_edge_distr( alg = 'mcs', itr = 1, n = 10000 )
        >>> # Gather distributions
        >>> distrs = [ spd_e_distr, spd_auto_e_distr, nmi_e_distr, mcs_e_distr ]
        >>> mydistrs_dataframe = myedgedistr.gather_dfs( distrs, cols = ['spd', 'spd_auto', 'nmi', 'mcs'] )
        >>> dist = myedgedistr.get_col_dists( mydistrs_dataframe )
        >>> dist.head()
                       spd  spd_auto       nmi       mcs
        spd       0.000000  0.018193  0.014186  0.088918
        spd_auto  0.018193  0.000000  0.004073  0.100639
        nmi       0.014186  0.004073  0.000000  0.098353
        mcs       0.088918  0.100639  0.098353  0.000000
        '''
        
        # Data clean-up.           
        zero_data = distrs_dataframe.fillna(0)
        
        # Define distance function and compute pairwise distances between cols.
        distance = lambda x, y: pd.np.linalg.norm(x - y)
        d = zero_data.apply(lambda col1: zero_data.apply(lambda col2: distance(col1, col2)))
        
        return d


    def plot( self, pos, edge_df, col, scale= 10, labels = True, filename='', **kwargs ):     
        '''Computes pairwise vector euclidean distance between columns.
        
        Parameters
        ----------
        pos : dictionary
           A dictionary with nodes as keys and positions as values.
           Positions should be sequences of length 2.

        Returns
        -------
        df : DataFrame
            In distance matrix configuration.
        
        edge_df : df
            index : edge tuples.
            col : estimated probabilities.
        
        col : string
            Name of column you want to use for the edge width.
        
        scale : integer, optional
            Scale the width of the edges appropriately for graph size.
        
        labels : boolean, optional
            Toggles edge labels.        
        
        filename : string, optional
            Path where you want to same the image.
        
        **kwargs : dict, optional
            kwargs for for drawing edge labels.
            
        
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 2 )
        >>> myedgedistr = edge_distribution( G, pts={1:.2,0:.3,3:.5}, pdf='truncated', sigma=.2 )
        >>> spd_e_distr = myedgedistr.get_edge_distr( path_alg = 'k', path_k = 2, alg = 'spd', itr = 1, auto = False, N = 1e2 )
        >>> distrs = [ spd_e_distr ]
        >>> mydistrs_dataframe = myedgedistr.gather_dfs( distrs, cols = ['spd'] )
        >>> myedgedistr.plot( pos=pos, edge_df=mydistrs_dataframe, col='spd', scale=10 )
        '''
        
        # Define Traversed edges, nodes, length labels, and their probabilities.
        pedges, pweights = edge_df.index.tolist(), edge_df[col].tolist()
        pnodes = list( set( [n for e in pedges for n in e] ) )
        pweights = [x*scale for x in pweights]
        if labels:        
            elength = nx.get_edge_attributes( self.G, 'weight' )
        
        # Setup Figure.
        plt.figure()        
        plt.axis('off')

        # Draw.
        nx.draw_networkx( self.G, pos = pos, with_labels = True, node_size = 400, node_color = 'lightblue' )
        nx.draw_networkx_nodes( self.G, pos, nodelist = pnodes, node_color='r' )
        nx.draw_networkx_edges( self.G, pos, edgelist = pedges, edge_color='r', width=pweights )
        if labels:  
            nx.draw_networkx_edge_labels( self.G, pos, edge_labels = elength )

        # Plot and save.     
        plt.show()      
        if len(filename)>1:     
            plt.savefig(filename, **kwargs)