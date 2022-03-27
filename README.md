
<!-- github doc -->
<!-- https://bit.ly/3IBkDvG -->

# Regresion Logística <img src='figures/python.svg' align="right" height="89" />

<!-- badges: start -->
<!-- ![](figures/python.svg) -->

![](figures/pandas.svg) ![](figures/numpy.svg)
<!-- ![](figures/r.svg) -->

## Resumen

<!-- </br> -->

-   Implementar regresión logística en Python 3, utilizando el algoritmo
    “descenso de gradiente”.

-   Se proporciona un conjunto de datos de entrenamiento llamado “wine”.

-   Requerimientos

    -   Un solo archivo .py que acepte como parámetros la entrada la
        ubicación de los datos de entrenamiento, la ubicación de los
        datos de evaluación, alpha, numIterations y el umbral.

    -   Intenten encontrar el mejor modelo posible variando la taza de
        aprendizaje, el número de iteraciones y el umbral de decisión,
        utilizando 10-fold cross-validation. Reporten esos tres valores,
        la precisión y el recall.

    -   Utilizando un alpha de 0.1, 1000 iteraciones y un umbral de 0.5,
        utilizando 10-fold crossvalidation, reporten la precision y el
        recall.

    -   Se utilizó un modelo determinístico, así que se debería obtener
        más o menos los mismos resultados.

-   Análisis

    -   Correr código con los datos de entrenamiento y verificar los
        resultados del punto 3.
    -   Verificar resultados del punto 2
    -   Sin calificar estilo de código.

## Diseño

1.  ¿Qué queremos obtener?

Un modelo que nos prediga la calidad del vino en función de las
variables proporcionadas. Nos interesa responder la siguiente pregunta:

<span style="background-color: #FFFF00">**¿Las características químicas
del vino influyen en su calidad (buena frente a mala)?**</span>

2.  ¿Qué tipo de datos tenemos y como se ven estas instancias?

| variable             | tipo    |
|----------------------|---------|
| fixed acidity        | float64 |
| volatile acidity     | float64 |
| citric acid          | float64 |
| residual sugar       | float64 |
| chlorides            | float64 |
| free sulfur dioxide  | float64 |
| total sulfur dioxide | float64 |
| density              | float64 |
| pH                   | float64 |
| sulphates            | float64 |
| alcohol              | float64 |
| quality              | object  |

3.  ¿Qué significa tener éxito? (métricas)

Utilizaremos las métricas **precision** y **recall** para determinar la
calidad del modelo, por encima del error.

4.  ¿Cómo lo vamos a medir? (función de pérdida)
