# -*- coding: utf-8 -*-
"""
--------------------
**paths**
--------------------

Description
-----------
Create stochastic paths.

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
import scipy.stats as st
import pandas as pd
from itertools import islice



###########
## Paths ##
###########

class path:
    """ Create a stochastic path.
    
    Parameters
    ----------    
    G : Networkx undirected graph
        Graph from which this path will be derived. The weights MUST be labeled
        'weight'.

    pi : tuple 
        Elements are nodes from `G`. This defines the path. Nodes should be 
        ordered.
        
    name : integer
        Should be usable as a key for a Python dictionary.
        
    pdf : string
        There are two options: 'normal' for the Gaussian Normal Distribution, 
        or 'truncated' for the Truncated Gaussian Normal Distribution. This 
        defines the edge weight noise . Select 'normal' if negative edge weights
        are allowed. Select 'truncted' edge weights must be non-negative. 

    sigma : float
        Defines the standard deviation for the noise `pdf`.
    
    Notes
    -----    
    We define a path :math:`\\pi_{i,j}` as an ordered sequence of nodes 
    :math:`(v_{k})_{k=1}^{m}` with :math:`v_{1}=i` and :math:`v_{m}=j` as source and destination
    nodes respectively and :math:`(v_{k},v_{k+1}) \\in E,  \\forall k`. A path is 
    called simple if :math:`v_{k} \\ne v_{l}, \\forall  k,l`. That is, there are no 
    loops or cycles in the path. 
    
    Let :math:`i,j \\in V` and define :math:`\\Pi_{i,j} =  \\{ \\pi_{i,j} \\mid \\pi_{i,j} \\text{ is simple}\\}`. 
    Then, for :math:`\\pi \\in \\Pi_{i,j}` define:
    
    .. math::
        Z_{\\pi} = \\sum_{v_{k} \\in \\pi} Y_{(v_{k},v_{k+1})} 
        
    as the stochastic length for path :math:`\\pi`. Then, the path 
    :math:`\\pi`'s length distribution is either:
    
    .. math:: 
    
        Z_{\\pi} &\\sim N \\left( \\sum_{v_{k} \\in \\pi} c_{(v_{k},v_{k+1})} ,  \\sum_{v_{k} \\in \\pi} c_{(v_{k},v_{k+1})}^{2}\\sigma^{2} \\right) \\\\
        Z_{\\pi} &\\approx TN \\left( \\sum_{v_{k} \\in \\pi} c_{(v_{k},v_{k+1})} ,  \\sum_{v_{k} \\in \\pi} c_{(v_{k},v_{k+1})}^{2}\\sigma^{2}\\right), \\text{ over } \\left(0,\\sum_{v_{k} \\in \\pi}  2c_{(v_{k},v_{k+1})}\\right).
  
    depending on which noise model is used.

    Examples
    --------
    >>> G, pos, start, end = grid_graph( size = 2, max_weight = 10 )
    >>> mypath = path( G=G, pi=(0,1,3), name=0, pdf='truncated', sigma=.2 )
   
    """
    
    def __init__( self, G, pi, name, pdf, sigma ):

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
        
        Examples
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
        lengths : array
            Array has length equal to `size`.   
        
        Examples
        --------
        >>> mypath.get_sample_length( `size` = 3 )
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
        '''Initializes and returns the path's length distribution.
             
        Returns
        -------
        distribution : object
            Scipy distribution object.

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



#################
## Path Family ##
#################
    
class path_family:
    """Create family of stochastic paths related by their start and endpoints.
    
    Parameters
    ----------
    G : Networkx undirected graph
        Graph from which this path will be derived. The weights MUST be labeled
        'weight'.

    start : integer
        Starting node for path. Must be a node in `G`.

    end : integer
        Ending node for path. Must be a node in `G`.

    pdf : string
        There are two options: 'normal' for the Gaussian Normal Distribution, 
        or 'truncated' for the Truncated Gaussian Normal Distribution. This 
        defines the edge weight noise . Select 'normal' if negative edge weights
        are allowed. Select 'truncted' edge weights must be non-negative.   

    sigma : float
        Defines the standard deviation for the noise `pdf`.
   
    Examples
    --------
    >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
    >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.2 )        
    """
    
    def __init__( self,  G, start, end, pdf, sigma ):

        # Define path attributes.
        self.G = G
        self.start = start
        self.end = end
        self.pdf = pdf
        self.sigma = sigma   
        

     
    def summary( self ):
        '''Print basic object attributes.
        
        Examples
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
        '''Generates the family as the k shortest simple paths in G.
        A simple path is a path with no repeated nodes. **No negative weights 
        are allowed**.      
        
        Parameters
        ----------
        k : integer
            The number of paths to return.
                 
        Returns
        -------
        paths : dictionary
            key: path name.
            
            value: path object.

        Notes
        -----
        This Networkx algorithm is based on algorithm by Jin Y. Yen, 
        which finds the first k paths requires O(kN^3) operations.
        
        Yen's algorithm computes the k shortest simple paths for a graph with
        non-negative edge cost. The algorithm employs any shortest path 
        algorithm to find the best path, then proceeds to find k-1 deviations
        of the best path. The  NetworkX implementation uses a bi-directional 
        Dijkstra's algorithm and returns a Python generator allowing iterative 
        queries for the next shortest path.
        
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.2 )  
        >>> myfamily.get_k_paths(k=2)
        {0: <__main__.path at 0x21b9e85da90>, 1: <__main__.path at 0x21b9e85dba8>}
        
        References
        ----------
        .. [1] https://networkx.github.io/documentation/networkx-1.10 
        
        .. [2] Jin Y. Yen, “Finding the K Shortest Loopless Paths in a Network”, Management Science, Vol. 17, No. 11, Theory Series (Jul., 1971), pp. 712-716.
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
        '''Generates the family as all simple paths in G. A simple path is a 
        path with no repeated nodes.  
        
        Returns
        -------
        paths : dictionary
            key: path name.
            
            value: path object.

        Notes
        -----
        Wrapper for a Networkx algorithm that uses a modified depth-first search to 
        generate the paths. A single path can be found in O(V+E) time but 
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
        
        .. [2] R. Sedgewick, “Algorithms in C, Part 5: Graph Algorithms”, Addison Wesley Professional, 3rd ed., 2001.
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
        first : object
            paths.path object
                        
        last : object
            paths.path object
            
        Returns
        -------
        probability : float
        
        Notes
        -----
        See `get_auto_paths` notes.     
        
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
        '''Generates the family as the i*k shortest simple paths in G,
        where i is automatically determined. A simple path is a path with no
        repeated nodes. **No negative weights are allowed**.  

        Parameters
        ----------
        k : integer
            Paths are selected in chunks of k until the selection criteria is
            met.
            
        tol : float, optional
            Threshold for selecting i.         

        Returns
        -------
        paths : dictionary
            key: path name.
            
            value: ordered tuple of nodes in the path.

        Notes
        -----
        `get_auto_paths` uses the Python generator's efficiency to retrieve 
        shortest paths in batches of size `k` and stops when the 
        probability, that the last path retrieved would ever be shorter than 
        the first retrieved, is below a user provided threshold `tol`.  More 
        formally, the stopping condition is :math:`P(Z_{\\pi} < Z_{\\gamma}) <` `tol` 
        where :math:`\\gamma`  and :math:`\\pi` are the first and last path retrieved 
        respectively. This works because :math:`P(Z_{\\pi} < Z_{\\gamma})` is an upper 
        bound on :math:`P(Z_{\\pi}<W)` where 
        :math:`W =  \min_{\\gamma \\in \\Gamma_{\\pi}} Z_{\\gamma}` and can be computed 
        quickly as:
        
        .. math::

            P(Z_{\\pi} < Z_{\\gamma}) &= P(Z_{\\pi} - Z_{\\gamma} < 0) \\\\ 
                                      &= 1 - P(Z_{\\gamma} - Z_{\\pi} \\leq 0) \\\\ 
                                      &= 1 - CDF_{Z_{\\gamma} - Z_{\\pi}}(0) \\\\
                                      &= SF_{Z_{\\gamma} - Z_{\\pi}}(0)

            
        where
        
        .. math::
            Z_{\\gamma} - Z_{\\pi} \\sim N(\\mu_{\\gamma}-\\mu_{\\pi},\\sigma_{\\gamma}^{2}+\\sigma_{\\pi}^{2})
        
        or where

        .. math::
            Z_{\\gamma} - Z_{\\pi} \\approx TN(\\mu_{\\gamma}-\\mu_{\\pi},\\sigma_{\\gamma}^{2}+\\sigma_{\\pi}^{2}), \\text{ over } (-2\\mu_{k},2\\mu_{k}) 
        
        depending on whether the weights must be non-negative or not. 

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
        '''Wrapper to generate the family.
        
        Parameters
        ----------
        alg : string        
            Wrapper to selects on of the three path generative functions.
            
        k : integer, optional for `alg` = 'k' or `alg` = 'auto'.
            Paths are selected in chunks of k until a selection criteria is
            met. 
            
        tol : float, optional for `alg` = 'auto'. 
            Threshold for selecting k.       
    
        Returns
        -------
        paths : dictionary
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