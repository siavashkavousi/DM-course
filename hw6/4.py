import itertools

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from helper import girvan_newman

g = nx.read_gml('karate.gml')
pos = nx.spring_layout(g)
nx.draw_networkx(g, pos)
# plt.show()

k = 4
comp = girvan_newman(g)
limited = itertools.takewhile(lambda c: len(c) <= k, comp)
communities = list(limited)[2]
colors = ['r', 'g', 'b', 'y']
for index, community in enumerate(communities):
    print np.array(community)
    nx.draw_networkx_nodes(g, pos, nodelist=community, node_color=colors[index])
    nx.draw_networkx_edges(g, pos)
plt.show()
