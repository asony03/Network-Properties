from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from pyspark.sql.functions import explode

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def closeness(g):
	# Get list of vertices. We'll generate all the shortest paths at
	# once using this list.
	# first get all the path lengths.
	# Break up the map and group by ID for summing
	# Sum by ID
	# Get the inverses and generate desired dataframe.
	vertices_df = g.vertices
	vertices = vertices_df.rdd.map(lambda row: row["id"]).collect()
	res = g.shortestPaths(landmarks=vertices)
	centrality_res = res.select("id", explode("distances")).rdd.map(lambda x: (x[0], x[2])).reduceByKey(lambda x, y: x + y)
	centrality_res = centrality_res.map(lambda x: (x[0], float(1.0/float(x[1]))))
	return sqlContext.createDataFrame(centrality_res, ['id', 'closeness'])

print("Reading in graph for problem 2.")
graph = sc.parallelize([('A','B'),('A','C'),('A','D'),
	('B','A'),('B','C'),('B','D'),('B','E'),
	('C','A'),('C','B'),('C','D'),('C','F'),('C','H'),
	('D','A'),('D','B'),('D','C'),('D','E'),('D','F'),('D','G'),
	('E','B'),('E','D'),('E','F'),('E','G'),
	('F','C'),('F','D'),('F','E'),('F','G'),('F','H'),
	('G','D'),('G','E'),('G','F'),
	('H','C'),('H','F'),('H','I'),
	('I','H'),('I','J'),
	('J','I')])
	
e = sqlContext.createDataFrame(graph,['src','dst'])
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()
print("Generating GraphFrame.")
g = GraphFrame(v,e)

print("Calculating closeness.")
res = closeness(g).sort('closeness',ascending=False)
res.show()
res.toPandas().to_csv("centrality_out.csv")