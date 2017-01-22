.. asp documentation master file, created by
   sphinx-quickstart on Fri Jan 20 17:32:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Approximate Stochastic Paths (ASP)
==================================
ASP is an experimental package for exploring stochastic shortest paths on Networkx weighted graphs.

Install
-------
This package uses setuptools. The use of virtual environments is recommended.

Normal Mode
+++++++++++
>>> git clone git@github.com:dennett/asp.git

>>> pip install asp

Development Mode
++++++++++++++++

>>> git clone git@github.com:dennett/asp.git

>>> pip install -e asp/

Contributing
------------
ASP is still an extremely young project, and I'm happy for any contributions (patches, code, bugfixes, documentation, whatever) to get it to a stable and useful point. Feel free to get in touch with me via email or directly via github.

Development is synchronized via git. To clone this repository, run

>>> git clone git://github.com/dennett/asp.git

License
-------
ASP is licensed under the GPLv3


Definitions
-----------
Consider a weighted graph :math:`G` defined by triple :math:`G = (V, E,C)` where :math:`V = {1, 2, . . . , n}` is a set of nodes, :math:`E`  is the edges subset of :math:`V \times V`, and :math:`C` is the :math:`n \times n` matrix of edge weights or costs :math:`c_{(i,j)} \in {{\mathbb R}}`. Denote edges as :math:`(i,j) \in E` for :math:`i,j \in V`. We define a stochastic weighted graph induced by :math:`G` as :math:`G^{*} = (V, E,C^{*} )` where :math:`C^{*}` is the :math:`n \times n` matrix of edge cost distributions defined by:

.. math:: Y_{(i,j)} = X_{(i,j)}c_{(i,j)} + c_{(i,j)}

where :math:`c_{(i,j)} \in C` and :math:`X_{(i,j)} \sim N(0,\sigma^{2})` i.i.d.
or :math:`X_{(i,j)} \sim TN(0,\sigma^{2})` i.i.d. Thus we perturb the edge 
weights by a multiple of the original edge weight. The variance 
:math:`\sigma^{2}` parametrizes the magnitude.

Implementation
--------------
ASP was developed with Anaconda Python 3.4.
	
.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: graphs
   :members:
   
.. automodule:: paths
   :members:

.. automodule:: distributions
   :members:    

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
