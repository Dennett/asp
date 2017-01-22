## Approximate Stochastic Paths (ASP)
ASP is an experimental package for exploring stochastic shortest paths on Networkx weighted graphs. 

It was developed with Anaconda Python 3.4.

### Website
https://dennett.github.io/asp/

### Technical Paper
[Estimating Stochastic Shortest Paths](tex/stochastic_shortest_path.pdf)

### Usage
Example script to create stochastic shortest path distributions using multiple algorithms.

```python
import networkx as nx
import pandas as pd
import asp

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
```

Example script to create edge distributions using multiple algorithms. The distribution represents the probability that an edge will be traversed in a stochastic shortest path.

```python
import networkx as nx
import pandas as pd
import asp

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
```


### Install
This package uses setuptools. The use of virtual environments is recommended.

### Normal Mode

	git clone git@github.com:dennett/asp.git

	pip install asp

### Development Mode

	git clone git@github.com:dennett/asp.git

	pip install -e asp/

### Contributing
ASP is still an extremely new project, and I'm happy for any contributions (patches, code, bugfixes, documentation, whatever) to get it to a stable and useful point. Feel free to get in touch with me via email or directly via github.

Development is synchronized via git. To clone this repository, run

	git clone git://github.com/dennett/asp.git

### License
ASP is licensed under the GPLv3

### My Gists
https://gist.github.com/Dennett
