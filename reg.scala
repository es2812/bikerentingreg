/*
 *              Miniproyecto de Regresión Lineal
 *                  Esther Cuervo Fernández
 *                        09-04-2019
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
val dataRDD = parsedData.map(a=>(a(2).toDouble, a(3).toDouble, a(4).toDouble, a(5).toDouble, a(6).toDouble, a(7).toDouble, a(8).toDouble, a(9).toDouble, a(10).toDouble, a(11).toDouble, a(12).toDouble, a(15).toDouble))
val newHeaders = headers.drop(2).dropRight(3) ++ Array("cnt"); //actualizamos la cabecera

val dataDF = dataRDD.toDF(newHeaders: _*); //convertimos a un DF desernollando el array newHeaders en varias strings

/* 
 *  Añadimos interacciones
 */

import org.apache.spark.sql.functions.col
val dataFull = dataDF.withColumn("atemp*hum*wind",col("atemp")*col("hum")*col("windspeed"))

/*
 *        Conversión de variables categóricas
 */

val eliminadas = Array("temp","weekday","holiday","season")
val headersCategoricos = Array("season","yr","mnth","holiday","weekday","workingday","weathersit").diff(eliminadas)
val outputOhe = headersCategoricos.map(x=>x+"_ohe")

//por defecto las variables categóricas están expresadas con indices desde 0 hasta n. Lo convertimos a n-1 variables. Podemos hacer esto automáticamente con OneHotEncoderEstimator

import org.apache.spark.ml.feature.OneHotEncoderEstimator

val ohe = new OneHotEncoderEstimator().setInputCols(headersCategoricos).setOutputCols(outputOhe)
//val oheModel = ohe.fit(dataFull) //con pipeline no hacemos fit aun
//val data = oheModel.transform(dataFull)


/*
 *        DataFrame a features, label
 */

import org.apache.spark.ml.feature.VectorAssembler

val featuresCols = outputOhe++newHeaders.diff(headersCategoricos).diff(Array("cnt")).diff(eliminadas)++Array("atemp*hum*wind") //las variables categoricas entrarán con _ohe

val va = new VectorAssembler().setInputCols(featuresCols).setOutputCol("features")
//val df = va.transform(data).select("features","cnt")


/*  
 *        Creamos el modelo
 */ 

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression()
lr.setSolver("l-bfgs")
lr.setFitIntercept(true)
lr.setLabelCol("cnt")

//val lm = lr.fit(df)
//val sum = lm.summary

/*
 *    Pipeline y Cross Validation
 */

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.Pipeline

val eval = new RegressionEvaluator().setMetricName("r2").setLabelCol("cnt")
val stages = Array(ohe,va,lr)

val pipe = new Pipeline().setStages(stages)

val params = new ParamGridBuilder().addGrid(lr.regParam,Array(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)).addGrid(lr.elasticNetParam,Array(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)).build()
val SEED = 123456
val cv = new CrossValidator().setEstimatorParamMaps(params).setEvaluator(eval).setEstimator(pipe).setNumFolds(3).setSeed(SEED)

val cvModel = cv.fit(dataFull)

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.LinearRegressionModel

val bestPipe = cvModel.bestModel.asInstanceOf[PipelineModel]
val lm = bestPipe.stages.last.asInstanceOf[LinearRegressionModel]
val sum = lm.summary

println(s"RegParam: ${lm.getRegParam}")
println(s"ElasticNetParam: ${lm.getElasticNetParam}")

println(s"Intercept: ${lm.intercept}")
println(s"Coeficientes:")
for(i <- 0 until featuresCols.length){
  println(s"${featuresCols(i)}: ${lm.coefficients(i)}")
}

/*
println(s"P-values:")
for(i <- 0 until featuresCols.length){
  println(s"${featuresCols(i)}: ${sum.pValues(i)}")
}^
println(s"Intercept: ${sum.pValues(featuresCols.length)}")
*/

println(s"MSE: ${sum.meanSquaredError}")
println(s"RMSE: ${sum.rootMeanSquaredError}")
println(s"R2: ${sum.r2}")
println(s"R2 ajustado: ${sum.r2adj}")

/*
 *  Generamos y guardamos el dataset predichos v. observados
 */

val predObs = bestPipe.transform(dataFull).select($"cnt",$"prediction",$"cnt"-$"prediction")
predObs.write.format("csv").save("./residuals/residuals.csv")

