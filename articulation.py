import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)
sc.setCheckpointDir('checkpoints')

def articulations(g, usegraphframe=False):
	# Get the starting count of connected components
	starting_count = g.connectedComponents().select('component').distinct().count()
	vertices_df = g.vertices
	edges_df = g.edges
	vertices = vertices_df.rdd.map(lambda row: row["id"]).collect()
	edges = edges_df.rdd.map(lambda row: (row["src"], row["dst"])).collect()
	res = []

	# Default version sparkifies the connected components process 
	# and serializes node iteration.
	if usegraphframe:
		# Get vertex list for serial iteration
		# For each vertex, generate a new graphframe missing that vertex
		# and calculate connected component count. Then append count to
		# the output
		for vertex in vertices:
			vertices_df_new = vertices_df.filter(vertices_df["id"] != vertex)
			edges_df_new = edges_df.filter((edges_df["src"] != vertex) & (edges_df["dst"] != vertex))
			graphframe_new = GraphFrame(vertices_df_new, edges_df_new)
			count = graphframe_new.connectedComponents().select('component').distinct().count()
			res.append((vertex, 1 if count > starting_count else 0))

	# Non-default version sparkifies node iteration and uses networkx 
	# for connected components count.
	else:
		nx_graph = nx.Graph()
		nx_graph.add_nodes_from(vertices)
		nx_graph.add_edges_from(edges)

		def get_connected_components(vertex):
			graph = deepcopy(nx_graph)
			graph.remove_node(vertex)
			return nx.number_connected_components(graph)

		res = vertices_df.rdd.map(lambda row: (row["id"], 1 if get_connected_components(row["id"]) > starting_count else 0))

	return sqlContext.createDataFrame(res, ['id', 'articulation'])

filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness 	

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()	

# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

#Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
print("---------------------------")
df.toPandas().to_csv("articulations_out.csv")

#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
