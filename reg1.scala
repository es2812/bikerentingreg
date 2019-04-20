/*
 *              Miniproyecto de Regresión Lineal
 *                  Esther Cuervo Fernández
 *
 *   Este script ejecuta los experimentos de validación cruzada para hallar
 *   los valores óptimos para regParam y elasticNetParam para cuatro modelos
 *   de regresión lineal distintos utilizando el dataset day.csv de Bikerenting.
 *
 *   Almacena el mejor modelo de cada tipo como un PipelineModel.
 *
 */

import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}

/*
 *  Constantes
 */

val PATH = "data/day.csv";

/*
 *  Lectura de fichero a Dataframe
 */

val lines = sc.textFile(PATH);
  
//separado por comas
val parsed = lines.map(x=>x.split(","));
  
//primera linea es nombre de columnas
val headers = parsed.first;
val parsedData = parsed.filter(x=>{!(x.contains(headers(0)))});

//Array[Array[String]] a RDD
//Nos saltamos la columna instant y dteday (2 primeras), y casual y registered (2 penúltimas)
//Se resta 1 a aquellas columnas indexadas por 1 en lugar de 0
val dataRDD = parsedData.map(a=>(a(2).toDouble-1, a(3).toDouble, a(4).toDouble-1, a(5).toDouble, a(6).toDouble, a(7).toDouble, a(8).toDouble-1, a(9).toDouble, a(10).toDouble, a(11).toDouble, a(12).toDouble, a(15).toDouble))
val newHeaders = headers.drop(2).dropRight(3) ++ Array("cnt"); //actualizamos la cabecera

val dataDF = dataRDD.toDF(newHeaders: _*); //convertimos a un DF desenrollando el array newHeaders en varias strings

/* 
 *  Añadimos interacciones
 */

val data = dataDF.withColumn("th",$"temp"*$"hum").withColumn("tw",$"temp"*$"windspeed").withColumn("hw",$"hum"*$"windspeed").withColumn("thw",$"temp"*$"hum"*$"windspeed")

/*
 *        Selección de variables:
 *          - Modelo 1: incluimos todas las variables
 *          - Modelo 2: eliminamos las variables "temp" "weekday" "holiday" y "season"
 */

val eliminadas = Array("season","atemp","holiday","workingday")

val headersCategoricos = Array("season","yr","mnth","holiday","weekday","workingday","weathersit").diff(eliminadas)

//por defecto las variables categóricas están expresadas con strings. Lo convertimos a vectores de tamaño n-1. Podemos hacer esto automáticamente con OneHotEncoderEstimator

import org.apache.spark.ml.feature.OneHotEncoderEstimator

val outputOhe = headersCategoricos.map(x=>x+"_ohe")

val ohe = new OneHotEncoderEstimator().setInputCols(headersCategoricos).setOutputCols(outputOhe)

/*
 *        DataFrame a features, label
 */

import org.apache.spark.ml.feature.VectorAssembler

val featuresCols1 = outputOhe++newHeaders.diff(headersCategoricos).diff(Array("cnt")).diff(eliminadas)//las variables categoricas entrarán con _ohe
val featuresCols2 = featuresCols1++Array("th","tw","hw")

val va1 = new VectorAssembler().setInputCols(featuresCols1).setOutputCol("features")
val va2 = new VectorAssembler().setInputCols(featuresCols2).setOutputCol("features")

/*  
 *        Creamos el modelo
 */ 

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression()
lr.setSolver("l-bfgs")
lr.setFitIntercept(true)
lr.setLabelCol("cnt")

/*
 *    Pipeline y Cross Validation
 */

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.Pipeline

val eval = new RegressionEvaluator().setMetricName("r2").setLabelCol("cnt")
val stages1 = Array(ohe,va1,lr)
val stages2 = Array(ohe,va2,lr)

val pipe1 = new Pipeline().setStages(stages1)
val pipe2 = new Pipeline().setStages(stages2)

val params = new ParamGridBuilder().addGrid(lr.regParam,Array(0,0.2,0.4,0.6,0.8,1.0)).addGrid(lr.elasticNetParam,Array(0,0.2,0.4,0.6,0.8,1)).build()

val SEED = 123456
val cv1 = new CrossValidator().setEstimatorParamMaps(params).setEvaluator(eval).setEstimator(pipe1).setNumFolds(2).setSeed(SEED)
val cv2 = new CrossValidator().setEstimatorParamMaps(params).setEvaluator(eval).setEstimator(pipe2).setNumFolds(2).setSeed(SEED)

/*
 *            Modelo 1
 */

//val cvModel1 = cv1.fit(data)

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegressionModel

//val bestPipe = cvModel1.bestModel.asInstanceOf[PipelineModel]

//val lm = bestPipe.stages.last.asInstanceOf[LinearRegressionModel]
//val sum = lm.summary

//println(s"MODELO 1")
//println(s"----------------------")
//println(s"RegParam: ${lm.getRegParam}")
//println(s"ElasticNetParam: ${lm.getElasticNetParam}")

//println(s"Intercept: ${lm.intercept}")
//println(s"Coeficientes:")
//for(i <- 0 until lm.coefficients.size){
//  println(s"[${i}] ${lm.coefficients(i)}")
//}

//println(s"MSE: ${sum.meanSquaredError}")
//println(s"RMSE: ${sum.rootMeanSquaredError}")
//println(s"R2: ${sum.r2}")
//println(s"R2 ajustado: ${sum.r2adj}")

/*
 *  Generamos y guardamos el dataset predichos v. observados
 */

//val predObs = bestPipe.transform(data).select($"cnt",$"prediction",$"cnt"-$"prediction")
//predObs.write.csv("./residuals/residualsDefinitive")

/*  
 *  Guardamos el mejor modelo encontrado
 */

//bestPipe.write.save("./models/modelDefinitive")

/*
 *            Modelo 2
 */

val cvModel2 = cv2.fit(data)

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegressionModel

val bestPipe = cvModel2.bestModel.asInstanceOf[PipelineModel]

val lm = bestPipe.stages.last.asInstanceOf[LinearRegressionModel]
val sum = lm.summary

println(s"MODELO 2")
println(s"----------------------")
println(s"RegParam: ${lm.getRegParam}")
println(s"ElasticNetParam: ${lm.getElasticNetParam}")

println(s"Intercept: ${lm.intercept}")
println(s"Coeficientes:")
for(i <- 0 until lm.coefficients.size){
  println(s"[${i}] ${lm.coefficients(i)}")
}

println(s"MSE: ${sum.meanSquaredError}")
println(s"RMSE: ${sum.rootMeanSquaredError}")
println(s"R2: ${sum.r2}")
println(s"R2 ajustado: ${sum.r2adj}")

/*
 *  Generamos y guardamos el dataset predichos v. observados
 */

val predObs = bestPipe.transform(data).select($"cnt",$"prediction",$"cnt"-$"prediction")
predObs.write.csv("./residuals2/residualsDefinitive")

/*  
 *  Guardamos el mejor modelo encontrado
 */

bestPipe.write.save("./models2/modelDefinitive")

