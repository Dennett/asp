# -*- coding: utf-8 -*-
"""
----------------------------
APPROXIMATE STOCHASTIC PATHS
----------------------------

Description
-----------
path_family.py - family of paths class for Approximate Stochastic Path distributions.
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
from itertools import islice
import scipy.stats as st
import pandas as pd
from asp.path import path



#################
## Path Family ##
#################
    
class path_family:
    """Class for computing on a family of paths related by their start and 
    endpoints. Choose to generate the family of paths from a three options:
    1. get_all_paths() : find all paths between endpoints.
    2. get_k_paths() : find only the k-shortest paths.
    3. get_auto_paths() : find the (i*k)-shortest paths where i is 
       automatically determined.
    """
    
    def __init__( self,  G, start, end, pdf, sigma ):
        '''Constructor.     
        
        Parameters
        ----------
        G : Networkx graph
            Graph from which this path will be derived.
            The nodes, edges, and edge weights below must exist in this graph.
            The weights MUST be labeled 'weight'.
            Asp has only be tested on undirected graphs.
    
        start : Networkx node index
            Starting node for path.
    
        end : Networkx node index
            Ending node for path.
    
        pdf : string
            There are two options:
                1. 'normal' for the Gaussian Normal Distribution, or 
                2. 'truncated' for the Truncated Gaussian Normal Distribution.
            This defines the amout of noise the edge weights in G have.
            Select 'normal' if negative edge weights are allowed.
            Select 'truncted' edge weights must be non-negative.   
    
        sigma : float
            Defines the standard deviation for the noise pdf.
        
        Example
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.2 )        
        '''

        # Define path attributes.
        self.G = G
        self.start = start
        self.end = end
        self.pdf = pdf
        self.sigma = sigma   
        

     
    def summary( self ):
        '''Print basic object attributes.
        
        Example
        --------

        >>> myfamily.summary()
        Parent Graph Info:
        Name: grid_2d_graph
        Type: Graph
        Number of nodes: 4
        Number of edges: 4
        Average degree:   2.0000
        Start node = 0
        End node = 3
        Noise PDF = truncated
        Sigma = 0.2
        '''
        print('Parent Graph Info:')
        print(nx.info( self.G ))
        print('Start node =', self.start)
        print('End node =', self.end)
        print('Noise PDF =', self.pdf)
        print('Sigma =', self.sigma)
        
        
        
    def get_k_paths( self, k ):
        '''Generates the family of paths as the k shortest simple paths in G.
        A simple path is a path with no repeated nodes. No negative weights 
        are allowed.      
        
        Parameters
        ----------
        k : integer
            The number of paths to return.
                 
        Returns
        -------
        paths: dictionary
            key: path name.
            value: path object.

        Notes
        -----
        This Networkx [1] algorithm is based on algorithm by Jin Y. Yen [2], 
        which finds the first k paths requires O(kN^3) operations.      
        
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.2 )  
        >>> myfamily.get_k_paths(k=2)
        {0: <__main__.path at 0x21b9e85da90>, 1: <__main__.path at 0x21b9e85dba8>}
        
        References
        ----------
        [1]	https://networkx.github.io/documentation/networkx-1.10       
        [2]	Jin Y. Yen, “Finding the K Shortest Loopless Paths in a Network”, 
           Management Science, Vol. 17, No. 11,
           Theory Series (Jul., 1971), pp. 712-716.
        '''
       
        # Set variables.
        G = self.G
        start = self.start
        end = self.end
        sigma = self.sigma
        pdf = self.pdf
        
        # Initialize shortest path generator.        
        generator = nx.shortest_simple_paths( G, start, end, weight = 'weight' )
        paths = {}    
        
        # Generate first k shortest paths and initialize as path objects.
        for key, p in enumerate( list( islice( generator, 0, k ) ) ):
            paths[key] = path( G, tuple(p), key, pdf, sigma )

        return paths


    
    def get_all_paths( self ):
        '''Generates the family of paths as all simple paths in G. A simple 
        path is a path with no repeated nodes.  
        
        Returns
        -------
        paths: dictionary
            key: path name.
            value: path object.

        Notes
        -----
        This Networkx [1] algorithm uses a modified depth-first search to 
        generate the paths [2]. A single path can be found in O(V+E) time but 
        the number of simple paths in a graph can be very large, e.g. O(n!) in 
        the complete graph of order n.     
        
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.2 )  
        >>> myfamily.get_all_paths()
        {0: <__main__.path at 0x154d7f0>,
         1: <__main__.path at 0x154d390>,
         2: <__main__.path at 0x151e400>,
         3: <__main__.path at 0x151eb38>,
         4: <__main__.path at 0x151e1d0>,
         5: <__main__.path at 0x151e860>,
         6: <__main__.path at 0x151e470>,
         7: <__main__.path at 0x151e438>,
         8: <__main__.path at 0x151e748>,
         9: <__main__.path at 0x151e828>,
         10: <__main__.path at 0x151e6a0>,
         11: <__main__.path at 0x151eda0>}
        
        References
        ----------
        .. [1] https://networkx.github.io/documentation/networkx-1.10
        .. [2] R. Sedgewick, “Algorithms in C, Part 5: Graph Algorithms”, 
               Addison Wesley Professional, 3rd ed., 2001.
        '''
       
        # Set variables.
        G = self.G
        start = self.start
        end = self.end
        sigma = self.sigma
        pdf = self.pdf

        # Initialize the path generator.         
        generator = nx.all_simple_paths( G, start, end )
        paths = {}    

        # Generate first k shortest paths and initialize as path objects.
        for key, p in enumerate( generator ):
            paths[key] = path(  G, tuple(p), key, pdf, sigma  )

        return paths


    
    def upper_bound( self, first, last ):
        '''Estimates the probability that the last path will ever be shorter
        than the first path. This provides an upper bound on the probability
        that the last path will ever be the shortest of all paths.

        Parameters
        ----------
        first : path object
                        
        last : path object  
            
        Returns
        -------
        probability: float
        
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.2 )  
        >>> paths = myfamily.get_k_paths(k=2)
        >>> myfamily.upper_bound( paths[0], paths[1] )
        0.46289082290119177
        '''
       
        # Set variables.
        mu = first.len_mean - last.len_mean
        sigma = np.sqrt( first.len_var + last.len_var )

        # Return probability appropriate for self.pdf.
        if self.pdf == 'normal':

            return st.norm.sf( 0, mu, sigma )

        # Return probability appropriate for self.pdf.
        if self.pdf == 'truncated':
            lower = - 2 * last.len_mean
            upper =   2 * first.len_mean  

            return st.truncnorm.sf( 0, ( lower - mu ) / sigma, ( upper - mu ) / sigma, loc = mu, scale =  sigma )


    
    def get_auto_paths( self, k, tol = 5e-2 ): 
        '''Generates the family of paths as the i*k shortest simple paths in G,
        where i is automatically determined. A simple path is a path with no
        repeated nodes. No negative weights are allowed.  

        Parameters
        ----------
        k : integer
            Paths are selected in chunks of k until a selection criteria is
            met.
            
        tol : float, optional
            Threshold for selecting k, only used when         

        Returns
        -------
        paths: dictionary
            key: path name.
            value: ordered tuple of nodes in the path.

        Notes
        -----
        The algorithm iteratively generates the next k shortest simple paths, 
        checks if the probability that the last path will ever be shorter than 
        the first path, and stops if that probability is less than the supplied
        tolerance.     
        

        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.2 )  
        >>> myfamily.get_auto_paths(k = 2, tol = 5e-3)
        {0: <__main__.path at 0x154d588>,
         1: <__main__.path at 0x154d898>,
         2: <__main__.path at 0x154de80>,
         3: <__main__.path at 0x154dac8>,
         4: <__main__.path at 0x154da20>,
         5: <__main__.path at 0x154de48>,
         6: <__main__.path at 0x154d198>,
         7: <__main__.path at 0x154d828>}
        '''
       
        # Set variables.
        G = self.G
        start = self.start
        end = self.end
        sigma = self.sigma
        pdf = self.pdf

        # Initialize shortest path generator. 
        generator = nx.shortest_simple_paths( G, start, end, weight = 'weight' )
        paths = {}

        # Generate the first k shortest simple paths.
        for key, p in enumerate( list( islice( generator, k ) ) ):
            paths[key] = path( G, tuple(p), key, pdf, sigma )

        # Error Checking
        if k > len(paths):
            print('error: k is larger than the number of paths')

        # Initialize variables.
        first = paths[0]
        last = paths[k-1]
        q = self.upper_bound( first, last )
        key = k-1    

        # Establish stop condition.
        while q > tol:
            new_paths = {}       

            # Generate the next k shortest simple paths.
            for p in list( islice( generator, k ) ):
                key += 1
                new_paths[key] = path( G, tuple(p), key, pdf, sigma )
                paths.update(new_paths)

            # Compute upper bound probability.
            q = self.upper_bound( first, paths[key] )

        return paths



    def get_paths( self, alg, k, tol = 5e-2 ): 
        '''Wrapper to generate the family of paths from a three options:
        1. 'all' : find all paths between endpoints.
        2. 'k' : find only the k-shortest paths.
        3. 'auto' : find the (i*k)-shortest paths where i is automatically 
           determined. 
    
        Parameters
        ----------
        alg : string        
            Wrapper to selects on of the three path generative functions.
            
        k : integer, optional - depending on alg
            Paths are selected in chunks of k until a selection criteria is
            met. For alg = 'k' or alg = 'auto'.
            
        tol : float, optional - depending on alg
            Threshold for selecting k. For alg = 'auto'.       
    
        Returns
        -------
        paths: dictionary
            key: path name.
            value: ordered tuple of nodes in the path.
      
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.2 )  
        >>> myfamily.get_paths( alg = 'auto', k = 2, tol = 5e-3 )
        {0: <__main__.path at 0x15156a0>,
         1: <__main__.path at 0x15152e8>,
         2: <__main__.path at 0x1515ba8>,
         3: <__main__.path at 0x1515080>}        
        ''' 
      
        # Run appropriate algorithm for given input.
        if alg == 'auto':
            return self.get_auto_paths(k, tol)
        
        elif alg == 'k':
            return self.get_k_paths(k)

        elif alg == 'all':
            return self.get_all_paths()

        else:
            print('Not Implemented')


 