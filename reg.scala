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
 *        Conversión de variables categóricas
 */

val headersCategoricos = Array("season","yr","mnth","holiday","weekday","workingday","weathersit")
val outputOhe = headersCategoricos.map(x=>x+"_ohe")

//por defecto las variables categóricas están expresadas con indices desde 0 hasta n. Lo convertimos a n-1 variables. Podemos hacer esto automáticamente con OneHotEncoderEstimator

import org.apache.spark.ml.feature.OneHotEncoderEstimator

val ohe = new OneHotEncoderEstimator().setInputCols(headersCategoricos).setOutputCols(outputOhe)
val oheModel = ohe.fit(dataDF)

val data = oheModel.transform(dataDF)

/*
 *        DataFrame a features, label
 */

import org.apache.spark.ml.feature.VectorAssembler

val featuresCols = outputOhe++newHeaders.diff(headersCategoricos).diff(Array("cnt")) //las variables categoricas entrarán con _ohe

val va = new VectorAssembler().setInputCols(featuresCols).setOutputCol("features")
val df = va.transform(data).select("features","cnt").withColumnRenamed("cnt","label")

/*  
 *        Creamos el modelo
 */ 

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression()
lr.setSolver("normal")
lr.setFitIntercept(true)

val lm = lr.fit(df)

val sum = lm.summary

println(s"Intercept: ${lm.intercept}")
println(s"Coeficientes:")
for(i <- 0 until featuresCols.length){
  println(s"${featuresCols(i)}: ${lm.coefficients(i)}")
}

println(s"P-values:")
for(i <- 0 until featuresCols.length){
  println(s"${featuresCols(i)}: ${sum.pValues(i)}")
}
println(s"Intercept: ${sum.pValues(featuresCols.length)}")

println(s"MSE: ${sum.meanSquaredError}")
println(s"RMSE: ${sum.rootMeanSquaredError}")
println(s"R2: ${sum.r2}")
println(s"R2 ajustado: ${sum.r2adj}")
