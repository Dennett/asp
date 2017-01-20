# -*- coding: utf-8 -*-
"""
----------------------------
APPROXIMATE STOCHASTIC PATHS
----------------------------

Description
-----------
family_distribution.py - path distribution class for Approximate Stochastic Path distributions.
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
import scipy.integrate as gral
import time
import collections as coll
import pandas as pd
import math
from asp.path import path

 
###############################
## Path Family Distributions ##
###############################

class family_distribution:
    """Class for computing and storing stochastic shortest path
    distributions for a path_family. Choose to estimate the distribution from 
    three options:
    1. get_spd_ssp_distr() : estimate via Sampling Paths' Distributions.
    2. get_nmi_ssp_distr() : estimate via NuMeric Integration.
    3. get_mcs_ssp_distr() : estimate via Monte Carlo Simulation.
    """
    
    def __init__( self, path_family ):
        '''Constructor.     
        
        Parameters
        ----------
        path_family : path_family object
    
        Example
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        '''

        # Define path attributes.        
        self.path_family = path_family


     
    def sample_mins_distr( self, paths, N ):      
        '''Generates N samples of which path has the minimum length from N 
        draws of each paths length distribution. 
    
        Parameters
        ----------
        paths : dict        
            Dictionary of path objects. Path names MUST be consecutive integers
            starting at zero.
            
        N : integer
            Number of samples to draw from each path's length distribution.      
    
        Returns
        -------
        mins: np.array()
             Array of names of the path with the minimum length out of all the
             samples drawn that round.
      
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> myfamdistr.sample_mins_distr( paths, N = 1e1 )
        array([1, 2, 0, 1, 2, 2, 0, 0, 1, 0], dtype=int64)
        '''
        
        # Set variables and check keys are in order.
        names = sorted(paths.keys())
        L = len(names)
        N = int(N)
        assert names == [x for x in range(L)]

        # Draw N samples each path's length distributions, store in np.array().              
        sample = np.zeros( ( N, L ) )
        for name in names:
            sample[:,name] = paths[name].get_sample_length( N )        
        
        # For each sample, find the arg minimum.
        mins = sample.argmin(axis=1)

        return mins    



    def get_marginal_change( self, mins, N, m = .1 ):  
        '''Computes how much the frequencies of the minimums has changed in the
        last m many draws. 
    
        Parameters
        ----------
        mins: np.array()
             Array of names of the path with the minimum length out of all the
             samples drawn that round.
            
        N : integer
            Number of samples to draw from each path's length distribution.      

        m : integer, optional
            Size of the window to compute the max marginal change on drawing 
            more samples.    
            
        Returns
        -------
        ms: float
             The maximum change in the frequency a path is the minimum over the
             paths.
      
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> mins = myfamdistr.sample_mins_distr( paths, N = 1e1 )
        >>> myfamdistr.get_marginal_change( mins, N = 1e1, m = .2 )
        0.025000000000000022
        '''
        
        # Set variable.
        
        t = N - int(math.floor(m*N))
        assert t > 1
        t = int(t)        
        
        # Count occurances of elements (which are path keys).        
        N_dict = coll.Counter(mins)
        t_dict = coll.Counter(mins[:t])

        # Compute all frequencies for N and t, and compute marginal change.
        ms = []
        for key in N_dict.keys():
            
            # Note, it's possible a key doesn't show up until after t.            
            try:
                ms.append( abs( N_dict[key]/N - t_dict[key]/t ) )
            except:
                ms.appned( N_dict[key]/N )
       
        return max( ms )



    def get_spd_ssp_distr( self, paths, auto, N, m = .1, tol = 5e-2, M = 5e5 ):
        '''Estimates the probability each path in paths will be the shortest
        path using the frequency each path is the shortest in random samples. 
    
        Parameters
        ----------
        paths : dict        
            Dictionary of path objects. Path names MUST be consecutive integers
            starting at zero.
            
        auto : boolean
            Toggles whether or not to continue drawing sample mins until the
            max marginal change is under less than tol.

        N : integer
            Number of samples to draw from each path's length distribution. 
            
        m : integer, optional - depending on auto.
            Size of the window to compute the max marginal change on drawing 
            more samples. For auto = True.      

        tol : integer, optional - depending on auto.
            Threshold for whether or not to continue drawing sample mins until 
            the max marginal change is under less than tol. For auto = True.

        M : integer, optional - depending on auto.
            The maximum number of draws to complete even if max marginal change
            is greater than tol. For auto = True.         
            
        Returns
        -------
        f_N_dict: dictionary
            key: path object.
            value: number of times the path was drawn as minimum of all the 
            path lengths, divided by the number of total draws.
        
        max_margin: float or None
             The maximum change in the frequency a path is the minimum over the
             paths. If auto=False, then None is returned.
      
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> mins = myfamdistr.sample_mins_distr( paths, N = 1e1 )
        >>> myfamdistr.get_spd_ssp_distr( paths, auto = True, N = 1e2, m = .1, tol = 5e-2, M = 5e5 )
        ({<__main__.path at 0x14ffeb8>: 0.34,
          <__main__.path at 0x14ffbe0>: 0.33,
          <__main__.path at 0x14ff278>: 0.33},
         0.025555555555555554)
        '''
        
        if auto:
            # Draw sample of minimums and calculate marginal change of N. 
            mins = self.sample_mins_distr( paths, N = N )          
            max_margin = self.get_marginal_change( mins, N = N, m = m )

            # Draw more samples of minimums until marginal change is less than
            # supplied tolerance level.
            while ( max_margin > tol ) and (N < M):
                incr_mins = self.sample_mins_distr( paths, N = N )
                mins = np.concatenate( [ mins, incr_mins ] )
                N = N + N
                max_margin = self.get_marginal_change( mins, N, m )
                     
            # Compute frequency estimate for each path and the max m.
            N_dict = coll.Counter(mins)
            f_N_dict = {}            
            for key in N_dict.keys():            
                f_N_dict[paths[key]] = N_dict[key]/N

            max_margin = self.get_marginal_change( mins, N, m )

            return f_N_dict, max_margin

        else:
            # Draw sample of minimums and calculate marginal change of N.
            mins = self.sample_mins_distr( paths, N = N ) 
            
            # Compute frequency estimate for each path and the max m.            
            N_dict = coll.Counter(mins)
            f_N_dict = {}            
            for key in N_dict.keys():            
                f_N_dict[paths[key]] = N_dict[key]/N

            return f_N_dict, None



    def pdf( self, x, name, paths ):
        '''Function defining the integrand for computing via integration.
        
        Parameters
        ----------
        x : float
            Variable of integration.
        
        name : integer
            Name of the path for which you want to compute the probability it
            is the shortest path out of the paths.
        
        paths : dict        
            Dictionary of path objects. 
        
        Returns
        -------
        prod : float
            Product of the integrand's factors.        
        
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> integrand = lambda x : myfamdistr.pdf( x, name=0, paths=paths )        
        '''
        
        # Get paths' distributions.
        Zpi_pdf = [paths[name].get_length_distribution().pdf(x)]
        Zgamma_prod_sf = [paths[key].get_length_distribution().sf(x) for key in paths.keys() if key != name]

        # Union all the factors of the integrand into one list.
        factors = Zpi_pdf + Zgamma_prod_sf
        
        return np.prod( factors )



    def get_nmi_ssp_distr( self, paths, **kwargs ):
        '''Estimates the probability each path in paths will be the shortest
        path using numerical integration.
        
        Parameters
        ----------
       
        paths : dict        
            Dictionary of path objects. 
        
        **kwargs : dict
            scipy.integrate.quad kwargs.
            
        Returns
        -------
        prob_dict: dictionary
            key: path object.
            value: estimated probability.
        
        abserr: float or None
            Estimated error returned by Scipy's Quad function.      
        
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> myfamdistr.get_nmi_ssp_distr( paths, epsrel = 1e-4, epsabs = 0 )
        ({<__main__.path at 0x1158588>: 0.16525750032721961,
          <__main__.path at 0x11581d0>: 0.7916708098067267,
          <__main__.path at 0x1158fd0>: 0.04307252246910802},
         {<__main__.path at 0x1158588>: 8.308936571721492e-06,
          <__main__.path at 0x11581d0>: 3.313787842624658e-05,
          <__main__.path at 0x1158fd0>: 1.9361401057474414e-06})
        '''

        # Set up distribution dictionary.
        p_dict = {}
        p_dict_abserr = {}
        for name in paths.keys():
            # Define integrand function.
            integrand = lambda x : self.pdf( x, name, paths )

            # Integrate. 
            y, abserr = gral.quad( integrand, -np.inf, np.inf, **kwargs )
            p_dict[paths[name]] = y
            p_dict_abserr[paths[name]] = abserr
            
        return p_dict, p_dict_abserr
        


    def get_mcs_ssp_distr( self, n ):
        '''Estimates the probability each path in paths will be the shortest
        path by repeatedly drawing stochastic graphs and computing shortest
        paths.
        
        Parameters
        ----------
       
        n : integer        
            Number of draws to make. 
            
        Returns
        -------
        freq_dict: dictionary
            key: path object.
            value: estimated probability based on the frequency a path is
            selected as the shortest.
        
        dummy: None
            Exists only to make the output like the others.      
        
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> myfamdistr.get_mcs_ssp_distr( n = 100 )
        ({<__main__.path at 0x1508780>: 0.84,
          <__main__.path at 0x1508240>: 0.01,
          <__main__.path at 0x1508160>: 0.15},
         None)
        '''
        
        # Set variable and base graph B.
        ssp_simulations = []         
        B = self.path_family.G.copy()          

        # Perform n simulations.    
        for i in range(n):
            
            # Initialize empty stochastic graph and find shortest path.
            G_star = B            

            for (u, v) in G_star.edges():
                e = st.truncnorm.rvs( -1 / self.path_family.sigma, 1 / self.path_family.sigma, loc = 0, scale = self.path_family.sigma, size = 1 )        
                w = B.edge[u][v]['weight'] + ( B.edge[u][v]['weight']*(e) )
                G_star.edge[u][v]['weight'] = w

            # Find shortest path and store it.
            shortest_path = nx.dijkstra_path( G_star, source = self.path_family.start, target = self.path_family.end, weight = 'weight' )
            ssp_simulations.append(shortest_path)

        # Tally how often each path was selected.
        ctr = coll.Counter( [tuple(x) for x in ssp_simulations] )
        hstgrm = dict(ctr)  
        sim_hstgrm = {}
        pdf = self.path_family.pdf
        sigma = self.path_family.sigma
        
        # Compute frequencies.       
        for i,key in enumerate(hstgrm.keys()):
            c = hstgrm[key]
            new_key = path( B, key, i, pdf, sigma )            
            sim_hstgrm[new_key] = c/n

        return sim_hstgrm, None



    def get_distr( self, paths, alg, itr, auto, N = 1e3, m = .1, tol = 5e-2, M = 5e5, n = 1000, **kwargs ): 
        '''Wrapper to generate the paths' distribution from a three options:
        1. 'spd' : estimate via Sampling Paths' Distributions.
        2. 'nmi' : estimate via NuMeric Integration.
        3. 'mcs' : estimate via Monte Carlo Simulation.
    
        Parameters
        ----------
        paths : dict        
            Dictionary of path objects. Path names MUST be consecutive integers
            starting at zero.
            
        itr : integer
            Number of trials to run.
            
        auto : boolean
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
        avg_prob : dictionary
            key: path object.
            value: estimated probability.
        
        var : dictionary
            key: path object.
            value: estimated variance over the trials.
            
        avg_err : dictionary
            key: path object.
            value: average estimated error or upperbound on error.    
    
        avg_time : float
            Mean time of the trials.   
      
        Examples
        --------
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> myfamdistr.get_distr( paths, alg='nmi', itr = 1, auto = False, epsrel = 1e-4, epsabs = 0 )
        ({<__main__.path at 0x1508390>: 0.13612039685532,
          <__main__.path at 0x15089e8>: 0.03171784593212658,
          <__main__.path at 0x1508978>: 0.8321459058917322},
         {<__main__.path at 0x1508390>: 0,
          <__main__.path at 0x15089e8>: 0,
          <__main__.path at 0x1508978>: 0},
         {<__main__.path at 0x1508390>: 1.223261904866232e-05,
          <__main__.path at 0x15089e8>: 6.40382637653869e-07,
          <__main__.path at 0x1508978>: 4.224820309950417e-05},
         3.1645328998565674)
        '''  

        # Set up variables.
        results = []
        times = []
        errs = []
        
        # Compute distr 'itr' number of times.
        for i in range(itr):
            
            if alg == 'spd':
                t1 = time.time()
                r,e = self.get_spd_ssp_distr( paths, auto = auto, N = N, m = m, tol = tol, M = M )
                t2 = time.time()               
                results.append( r )
                times.append( t2-t1 )
                errs.append(e)                
                
            elif alg == 'nmi':
                t1 = time.time()                
                r,e = self.get_nmi_ssp_distr( paths, **kwargs ) 
                t2 = time.time()               
                results.append( r )
                times.append( t2-t1 ) 
                errs.append(e)
                
            elif alg == 'mcs':
                t1 = time.time()
                r,e = self.get_mcs_ssp_distr( n = n ) 
                t2 = time.time()               
                results.append( r )
                times.append( t2-t1 )  
                errs.append(e)
                
            else:
                print('Not Implemented')

        # Avg the results and times.        
        if itr == 1:
            avg_result = results[0]
            var_result = dict.fromkeys(results[0], 0)
            avg_time = times[0]
            avg_err = errs[0]
            
        else:
            df = pd.DataFrame(results)
            df = df.fillna(0)
            avg_result = dict(df.mean())
            var_result = dict(df.var())
            avg_time = sum(times)/itr
            try:
                avg_err = sum(errs)/itr
            except:
                avg_err = None
            
        return avg_result, var_result, avg_err, avg_time



    def keys_2_nodes( self, distrs ):
        '''Converts the keys in the distribution dictionary from path names to
        a tuple of path nodes.
        
        Parameters
        ----------       
        distrs : list
            List of distribution objects with names as keys.
            
        Returns
        -------
        distrs : list
            List of distribution objects with node tuple as keys.
    
        Examples
        -------           
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> avg_result, var_result, avg_err, avg_time = myfamdistr.get_distr( paths, alg='nmi', itr = 1, auto = False, epsrel = 1e-4, epsabs = 0 )
        >>> myfamdistr.keys_2_nodes( [ avg_result ] )
        [{(0, 1, 2, 5, 8): 0.3797458089395961,
          (0, 3, 4, 7, 8): 0.3653256469471194,
          (0, 3, 6, 7, 8): 0.2549285441225291}]
        '''

        # Set up variables.       
        new_distrs = []
        
        # Loop through distributions exchanging keys.        
        for i,distr in enumerate(distrs):
            new_distr = {}
            for k in distr.keys():
                new_distr[k.pi] = distr[k]
            new_distrs.append(new_distr)
        
        return new_distrs



    def gather_dicts( self, dicts, cols ):
        '''Converts list of dictionaries to a dataframe, merged on keys, with 
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
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> # Generate output for all the algorithms 
        >>> spd_auto_dict, spd_auto_var, spd_auto_err_est, spd_auto_t_est = myfamdistr.get_distr( paths, itr = 4, alg = 'spd', auto = True, N = 1e2, m = .1, tol = 5e-2, M = 5e5  )
        >>> spd_dict, spd_var, spd_err_est, spd_t_est = myfamdistr.get_distr( paths, itr = 4, auto = False, alg = 'spd' )
        >>> nmi_dict, nmi_var, nmi_err_est, nmi_t_est = myfamdistr.get_distr( paths, itr = 4, auto = False, alg = 'nmi', epsrel = 1e-2, epsabs = 0 )
        >>> mcs_dict, mcs_var, mcs_err_est, mcs_t_est = myfamdistr.get_distr( paths, itr = 4, auto = False, alg = 'mcs', n = 1000 )
        >>> # Gather all the distributions 
        >>> distrs = myfamdistr.keys_2_nodes(  [ spd_auto_dict, spd_dict, nmi_dict, mcs_dict ] )
        >>> mydistrs_dataframe = myfamdistr.gather_dicts( distrs, cols = ['spd_auto', 'spd', 'nmi', 'mcs'] )
        >>> mydistrs_dataframe.head()
                         spd_auto     spd       nmi      mcs
        (0, 1, 4, 5, 8)    0.5375  0.5545  0.550699  0.09750
        (0, 1, 4, 7, 8)    0.3675  0.3630  0.364878  0.01900
        (0, 1, 2, 5, 8)    0.0950  0.0825  0.084425  0.00425
        (0, 3, 4, 5, 8)    0.0000  0.0000  0.000000  0.06250
        (0, 3, 6, 7, 8)    0.0000  0.0000  0.000000  0.00400
        >>> # Gather all the variances 
        >>> variances = myfamdistr.keys_2_nodes(  [ spd_auto_var, spd_var, nmi_var, mcs_var ] )
        >>> myvars_dataframe = myfamdistr.gather_dicts( variances, cols = ['spd_auto', 'spd', 'nmi', 'mcs'] )
        >>> myvars_dataframe.head()
                         spd_auto       spd  nmi       mcs
        (0, 1, 4, 5, 8)  0.001558  0.000051  0.0  0.038025
        (0, 1, 4, 7, 8)  0.002292  0.000261  0.0  0.001444
        (0, 1, 2, 5, 8)  0.000100  0.000084  0.0  0.000072
        (0, 3, 4, 5, 8)  0.000000  0.000000  0.0  0.015625
        (0, 3, 6, 7, 8)  0.000000  0.000000  0.0  0.000064
        '''
        
        # Set up variables.        
        df = None
        
        # Loop through dictionaries, convert to dataframes, and merge.
        for d in dicts:
            if not isinstance(df, pd.DataFrame):              
                df = pd.DataFrame.from_dict(d, orient='index')
            elif isinstance(df, pd.DataFrame):           
                df = pd.merge(df, pd.DataFrame.from_dict(d, orient='index'), left_index = True, right_index = True, how = 'outer')             
   
        # Clean up.       
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
        >>> G, pos, start, end = grid_graph( size = 3, max_weight = 10 )
        >>> myfamily = path_family( G=G, start=start, end=end, pdf='truncated', sigma=.5 )         
        >>> myfamdistr = family_distribution( myfamily )
        >>> paths = myfamily.get_paths( alg = 'k', k = 3 )
        >>> # Generate output for all the algorithms 
        >>> spd_auto_dict, spd_auto_var, spd_auto_err_est, spd_auto_t_est = myfamdistr.get_distr( paths, itr = 4, alg = 'spd', auto = True, N = 1e2, m = .1, tol = 5e-2, M = 5e5  )
        >>> spd_dict, spd_var, spd_err_est, spd_t_est = myfamdistr.get_distr( paths, itr = 4, auto = False, alg = 'spd' )
        >>> nmi_dict, nmi_var, nmi_err_est, nmi_t_est = myfamdistr.get_distr( paths, itr = 4, auto = False, alg = 'nmi', epsrel = 1e-2, epsabs = 0 )
        >>> mcs_dict, mcs_var, mcs_err_est, mcs_t_est = myfamdistr.get_distr( paths, itr = 4, auto = False, alg = 'mcs', n = 1000 )
        >>> # Gather all the distributions 
        >>> distrs = myfamdistr.keys_2_nodes(  [ spd_auto_dict, spd_dict, nmi_dict, mcs_dict ] )
        >>> mydistrs_dataframe = myfamdistr.gather_dicts( distrs, cols = ['spd_auto', 'spd', 'nmi', 'mcs'] )
        >>> dist = myfamdistr.get_col_dists( mydistrs_dataframe )
        >>> dist.head()
                  spd_auto       spd       nmi       mcs
        spd_auto  0.000000  0.055158  0.048413  0.889655
        spd       0.055158  0.000000  0.007579  0.853678
        nmi       0.048413  0.007579  0.000000  0.856840
        mcs       0.889655  0.853678  0.856840  0.000000
        '''
        
        # Data clean-up.
        zero_data = distrs_dataframe.fillna(0)
        
        # Define distance function and compute pairwise distances between cols.
        distance = lambda x, y: pd.np.linalg.norm(x - y)
        d = zero_data.apply(lambda col1: zero_data.apply(lambda col2: distance(col1, col2)))
        
        return d
        
