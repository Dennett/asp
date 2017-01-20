# -*- coding: utf-8 -*-
"""
----------------------------
APPROXIMATE STOCHASTIC PATHS
----------------------------

Description
-----------
path.py - path class Approximate Stochastic Path distributions.
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
"""



#####################
## Import Packages ##
#####################

import networkx as nx
import numpy as np
import scipy.stats as st
import pandas as pd




###########
## Paths ##
###########

class path:
    """ Class for computing on paths.Stores path attributes and computes 
    summary statistics. Generates path length distribution and samples from it.
    """
    
    def __init__( self, G, pi, name, pdf, sigma ):

        ''' Constructor.        
        
        Parameters
        ----------    
        G : Networkx graph
            Graph from which this path will be derived.
            The nodes, edges, and edge weights below must exist in this graph.
            The weights MUST be labeled 'weight'.
            Asp has only be tested on undirected graphs.
    
        pi : tuple 
            Tuple of nodes from G. This defines the path. Nodes should be 
            ordered.
            
        name : integer
            Should be usable as a key for a Python dictionary.
            
        pdf : string
            There are two options:
                1. 'normal' for the Gaussian Normal Distribution, or 
                2. 'truncated' for the Truncated Gaussian Normal Distribution.
            This defines the amount of noise the edge weights in G have.
            Select 'normal' if negative edge weights are allowed.
            Select 'truncted' edge weights must be non-negative.   
    
        sigma : float
            Defines the standard deviation for the noise pdf.
        
        Example
        --------
        >>> G, pos, start, end = grid_graph( size = 2, max_weight = 10 )
        >>> mypath = path( G=G, pi=(0,1,3), name=0, pdf='truncated', sigma=.2 )
        '''

       # Define path attributes.
        self.name = name
        self.pi = pi
        self.G = G
        self.pdf=pdf
        self.sigma=sigma
  
        # Retrieve edge weights from G and edges.
        weights = []
        edges = []
        for (u, v) in zip( pi, pi[1:] ):
            weights.append( G.edge[u][v]['weight'] )
            edges.append((u, v))
        self.weights = weights
        self.edges = edges
        
        # Compute length distribution mean and variance.
        self.len_mean = sum( weights )
        self.len_var = sum( ( np.square( weights )*( sigma**2 ) ) )


    
    def summary( self ):
        '''Print basic object attributes.
        
        Example
        --------
        >>> mypath.summary()
        Path name = 0
        Parent Graph Info:
        Name: grid_2d_graph
        Type: Graph
        Number of nodes: 4
        Number of edges: 4
        Average degree:   2.0000
        Noise PDF = truncated
        Sigma= 0.2
        Length Mean = 3.81
        Length Variance = 0.309924
        '''
        
        print('Path name =',self.name)
        print('Parent Graph Info:')
        print(nx.info( self.G ))
        print('Noise PDF =', self.pdf)
        print('Sigma=', self.sigma)
        print('Length Mean =', self.len_mean)
        print('Length Variance =', self.len_var)
         
            
      
    def get_sample_length( self, size ):        
        '''Generates samples from the path's length distribution.
        
        Parameters
        ---------- 
        size : integer
            Number of samples to draw from the path's length distribution.
        
        Returns
        -------
        lengths: array
            Array has length equal to size. 

        Examples
        --------
        >>> mypath.get_sample_length( size = 3 )
        array([ 3.9308913 ,  2.38708896,  3.76789294])
       '''
       
        # Initialize variables
        m = self.len_mean
        s = np.sqrt( self.len_var )

        # Return sample from Normal Dist.
        if self.pdf == 'normal':
            return np.random.normal( m, s, size )            

        # Return sample from Truncated Normal Dist.
        if self.pdf == 'truncated':
            return st.truncnorm.rvs( ( 0 - m ) / s, ( 2*m - m ) / s, loc = m, scale =  s, size = size)



    def get_length_distribution( self ):
        '''Inializes and returns the path's length distribution
             
        Returns
        =======
        distribution: Scipy distribution object.

        Examples
        --------
        >>> path_distr = mypath.get_length_distribution() 
        >>> path_distr.rvs( size=3 )
        array([ 3.96226882,  3.39493206,  3.22271213])
        '''

        # Initialize variables 
        m = self.len_mean
        s = np.sqrt( self.len_var )
        
        # Return Scipy Normal Dist. object.
        if self.pdf == 'normal':
            return st.norm( m, s )            
 
        # Return Scipy Truncated Normal Dist. object.
        if self.pdf == 'truncated':   
            return st.truncnorm( ( 0 - m ) / s, ( 2*m - m ) / s, loc = m, scale = s )

   