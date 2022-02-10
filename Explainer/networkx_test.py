
import networkx as nx
import numpy as np

nSubTrain = 2
d = np.array([[1,2],[3,4]]) # Not a digraph, 2 gets ignored

# Dense training graph

Gtrain = nx.Graph()

# Edges
for i in range(nSubTrain):
	for j in range(nSubTrain):
		Gtrain.add_edge(i,j,weight=d[i,j])

for (u,v,w) in Gtrain.edges.data('weight'):
	print(f'({u}, {v}, {w})')

print(nx.to_numpy_matrix(Gtrain))

# Nodes
for i in range(nSubTrain):
	Gtrain.add_node(i,feat=np.random.random(3))

print(Gtrain.nodes[0]['feat'])

for i,u in enumerate(Gtrain.nodes()):
	print(f'{i} {Gtrain.nodes[u]["feat"]}')

#print(Gtrain['label'])
#print(Gtrain['feats'])
#print(Gtrain['adj'])
