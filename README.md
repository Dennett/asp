# asp
## Approximate Stochastic Paths

ASP is an experimental package for exploring stochastic shortest paths on Networkx weighted graphs. 

It was developed with Anaconda Python 3.4.

### Usage

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
spd_dict, spd_var, spd_err_est, spd_t_est = mydistr.get_distr( paths, alg = 'spd'  )
nmi_dict, nmi_var, nmi_err_est, nmi_t_est = mydistr.get_distr( paths, alg = 'nmi' )
mcs_dict, mcs_var, mcs_err_est, mcs_t_est = mydistr.get_distr( paths, alg = 'mcs' )


# Analyze distributions
distrs = mydistr.keys_to_nodes(  [ spd_dict, nmi_dict, mcs_dict ] )
mydistrs_dataframe = mydistr.gather_dicts( distrs, cols = [ 'spd', 'nmi', 'mcs'] )
dist = mydistr.get_col_dists( mydistrs_dataframe )
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