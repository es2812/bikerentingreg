1. Utilizando todas las variables disponibles, no es posible obtener p-valores, problema con la resolución exacta (creo que porque la matriz tiene demasiados 0s debido a los ohes)

- Intercept: 2779.058406215549
- Coeficientes:
	- season: 0.0
	- yr: -1575.340544153193
	- mnth: -690.6849176864655
	- holiday: -742.5975438300804
	- weekday: -2019.7297790975608
	- workingday: 0.0
	- weathersit: 89.63464801520195
	- temp: 220.6658052385815
	- atemp: 632.4974239427331
	- hum: 540.8804068625128
	- windspeed: 825.2313856570603
- MSE: 568264.9640739373
- RMSE: 753.8335121722417
- R2: 0.8483678327558559
- R2 ajustado: 0.8414162147733164

Aún sin los p-valores es obvio que algunas variables no aportan (por si solas), ya que tienen coeficiente 0 (season, workingday). Las eliminamos

2. Sigue sin mostrar p-valores

- Intercept: 2156.486712644554
- Coeficientes:
	- yr: -2018.3631306260918
	- mnth: 0.0
	- holiday: -896.8605171711041
	- weekday: -772.7164391567712
	- weathersit: -63.091863970678766
	- temp: 412.36704943565945
	- atemp: 658.3774503822478
	- hum: 414.39485591972146
	- windspeed: -118.17030272715806
- MSE: 639005.216410655
- RMSE: 799.3780184685183
- R2: 0.8294919589094111
- R2 ajustado: 0.8229432859230015

coeficiente de mnth es 0. Eliminamos

3. Sigue sin mostrar p-valores

- Intercept: 1306.6593658966833
- Coeficientes:
	- yr: -2021.6292146966932
	- holiday: 667.374760730165
	- weekday: -449.67389534444544
	- weathersit: -242.66275516997223
	- temp: -162.52504868657746
	- atemp: -99.82408215295132
	- hum: -55.32214459729284
	- windspeed: 3.974483304984023
- MSE: 911922.3252066342
- RMSE: 954.9462420506372
- R2: 0.7566685133320838
- R2 ajustado: 0.7515636569684213

gran caida del R2

4. Dejamos solo las variables que no son obviamente correlacionadas (p.e. de atemp y temp dejamos una, atemp; de weekday, holiday y workday, dejamos workday; de season y mnth dejamos season). No hay p-valores

- Intercept: 3248.510214024744
- Coeficientes:
	- season: 0.0
	- yr: -1550.5675484538533
	- workingday: -362.2919808323232
	- weathersit: -561.8959918654595
	- atemp: -2019.6206227344069
	- hum: -178.18004873808266
	- windspeed: 0.0
- MSE: 682261.922715331
- RMSE: 825.9914785972861
- R2: 0.8179496176786607
- R2 ajustado: 0.8149069929044879

mejora del R2, pero perdemos season y windspeed como variables. Metemos mnth en lugar de season.

5. No hay p-valores

- Intercept: 2911.36794694052
- Coeficientes:
	- yr: -2022.9022823541266
	- mnth: 0.0
	- workingday: -931.0043810537016
	- weathersit: -784.2405618509421
	- atemp: -24.169352691468678
	- hum: 443.2487413859901
	- windspeed: 749.2590897344528
- MSE: 669600.0277175192
- RMSE: 818.2909187553771
- R2: 0.8213282362832139
- R2 ajustado: 0.8162952288545721

mejora del R2, pero mnth no tiene coeficiente.

Conclusiones: 

- no se puede aproximar de manera exacta (falla el resolvedor normal y acaba intentandolo con aproximado, sin haber ajustado los parámetros)
- no hemos modelado interacciones.

Solucion:

- CV
- Modelar interacciones entre variables en las que tiene sentido (p.e. atemp, hum y windspeed)

6. Añadida interacción entre atemp, hum y windspeed. Sigue sin haber p-valores

- Intercept: 2608.475705411673
- Coeficientes:
	- yr: -2020.691323030778
	- mnth: 0.0
	- workingday: -929.7273261843345
	- weathersit: -785.4641231792816
	- atemp: -25.68872267986138
	- hum: 447.2410509721976
	- windspeed: 752.61215188339
	- atemp\*hum\*wind: 573.5160678140222
- MSE: 669028.5218366127
- RMSE: 817.9416371823926
- R2: 0.8214807332358562
- R2 ajustado: 0.8161931385926305

Volvemos a perder mnth, R2 se mantiene aprox y RMSE disminuye algo

7. CV para regParam valores 0, 0.1, 0.2, 0.3, ... 1, l-bfgs, 3 folds. Valor de regParam elegido: 1

- Intercept: 2639.367706515705
- Coeficientes:
	- yr: -2019.894779459438
	- mnth: 0.0
	- workingday: -930.7509689585469
	- weathersit: -786.5000845998156
	- atemp: -27.172387421972243
	- hum: 445.4362447751988
	- windspeed: 751.5662732384042
	- atemp\*hum\*wind: 572.1745576039873
- MSE: 669035.6219124837
- RMSE: 817.9459773802203
- R2: 0.8214788386972871
- R2 ajustado: 0.8161911879393788

8. Valores muy parecidos. Entrenamos también elasticNetParam, entre los mismos valores. regParam elegido: 1, elasticNetParam elegido: 1

- Intercept: 2663.0668622231096
- Coeficientes:
	- yr: -2018.4961523581248
	- mnth: 0.0
	- workingday: -933.8557096179521
	- weathersit: -793.0440402986524
	- atemp: -38.802848704148815
	- hum: 422.22613204517455
	- windspeed: 723.4259208350435
	- atemp\*hum\*wind: 539.3414508333672
- MSE: 669106.6514860142
- RMSE: 817.989395705112
- R2: 0.8214598856228346
- R2 ajustado: 0.8161716734903657

R2 algo mejor. Y si hacemos CV por r2?

9. Mismo pero CV con r2. regParam 1, elasticNetParam 1

- Intercept: 2663.0668622231096
- Coeficientes:
	- yr: -2018.4961523581248
	- mnth: 0.0
	- workingday: -933.8557096179521
	- weathersit: -793.0440402986524
	- atemp: -38.802848704148815
	- hum: 422.22613204517455
	- windspeed: 723.4259208350435
	- atemp\*hum\*wind: 539.3414508333672
- MSE: 669106.6514860142
- RMSE: 817.989395705112
- R2: 0.8214598856228346
- R2 ajustado: 0.8161716734903657

Mismo resultado.

Guardamos los residuos para hacer plots y comprobar las hipótesis del modelo.

Modelo 1 (regParam = 1 elasticNetParam = 1 all variables):

- Intercept: 2548.2908671701803
- Coeficientes:
	- season (3): 0.0,-1548.682181052428,-634.6704576980247
	- yr (1): -694.3180214916454
	- mnth (11):         
	- holiday: -646.4116324932043    
	- weekday: -2015.621138794571    
	- workingday: 0.0                
	- weathersit: 27.304893507135933 
	- temp: 152.32744913355435           
	- atemp: 542.5912651327928           
	- hum: 418.3439986061175             
	- windspeed: 703.354683337561        
	- atemp\*hum\*wind: 479.5411072882426  
- MSE:  568202.1271455066
- RMSE: 753.7918327665182
- R2: 0.848384599757595
- R2 ajustado: 0.8412062522568785


Con Y transformada por Y'=ln(Y):

Modelo 1, tras CV, RegParam: 0, ElasticNetParam 0.9:

- Intercept: 7.266242445152862
- Coeficientes:
	- season: 0.0
	- yr: -0.564031294571285
	- mnth: -0.25956731741982797
	- holiday: -0.22023239691245902
	- weekday: -0.46041631999137705
	- workingday: 0.0
	- weathersit: 0.01030410969812086
	- temp: 0.14898840262798632
	- atemp: 0.200087882575682
	- hum: 0.13068889307603218
	- windspeed: 0.15202495288521442
	- atemp\*hum\*wind: 0.0023599870078927165
- MSE: 0.08038778216245684
- RMSE: 0.2835273922612361
- R2:  0.7634538070630084
- R2 ajustado:  0.7522543459913861

Modelo 2, tras CV, regParam 0, ElasticNetParam 0.3:

- Intercept: 7.262596037980402
- Coeficientes:
	- yr (2011): 
	- yr (2012): 
	- mnth(1-12): 
	- workingday: -0.3523715047401349
	- weathersit: -0.20972390803274815
	- atemp: -0.038499789398834236
	- hum: 0.08495481507275257
	- windspeed: 0.11602536588689043
	- atemp\*hum\*wind: 0.003303202978284427
- MSE: 0.09169538900873245
- RMSE: 0.30281246508149634
- R2: 0.7301804503567707
- R2 ajustado: 0.7221886160231913


