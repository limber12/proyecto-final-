import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import scala.io.Source
import java.io._
import org.apache.spark.ml.feature.IndexToString

import org.apache.spark.sql.functions._

import spark.implicits._


import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
//import co.theasi.plotly
//import util.Random

//1
import org.apache.spark.sql.SparkSession
//2
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)//Quita muchos warnings

val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF("x1","x2","x3","x4","x5")
df.show()

val newdf = df.withColumn("label", when(col("x5") === "Iris-setosa",1.0).otherwise(when(col("x5") === "Iris-versicolor", 2.0).otherwise(3.0)))
newdf.show()
val ensamblador = new VectorAssembler().setInputCols(Array("x1", "x2", "x3", "x4")).setOutputCol("features")
val transformado = ensamblador.transform(newdf)
transformado.show()
val kmeans = new KMeans().setK(2).setSeed(1L)

val model = kmeans.fit(transformado)

// Make predictions
val predictions = model.transform(transformado)

// Evaluate clustering by computing Silhouette score
val evaluator = new ClusteringEvaluator()

val silhouette = evaluator.evaluate(predictions)
println(s"Silhouette with squared euclidean distance = $silhouette")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
