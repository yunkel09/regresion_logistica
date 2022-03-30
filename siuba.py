
import pandas as pd
import numpy as np
import random
from siuba import *
from siuba.dply.verbs import *

vinos = pd.read_csv("wine.csv")


vinos >> \
    group_by("quality") >> \
    summarize(
        media = np.mean(_.alcohol))

vinos >> \
    group_by(_.alcohol) >> \
    mutate(loc = _.density - _.pH)
    
vinos >> \
    filter(_.quality == "Bad")


vinos >> \
 select("density", "alcohol")
