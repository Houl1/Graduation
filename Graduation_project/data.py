import numpy as np
data =  [4, 5, 5, 4, 5, 5, 4, 4, 0, 0, 4, 4, 5, 0, 4, 0, 5, 2, 0, 4, 4, 5, 5, 4, 4, 4, 5, 4, 5, 4, 0, 5, 0, 5, 5, 0, 5, 0, 2, 4, 5, 4, 5, 4, 0, 5, 4, 7, 4, 4, 4, 5, 0, 5, 2, 0, 4, 0, 0, 5, 4, 7, 5, 0, 5, 0, 5, 5, 4, 5, 7, 4, 5, 4, 5, 7, 0, 0, 5, 0, 7, 0, 4, 4, 7, 5, 5, 0, 4, 4, 0, 0, 5, 4, 5, 0, 7, 0, 7, 4, 4, 7, 0, 4, 7, 4, 5, 7, 4, 4, 4, 5, 4, 5, 0, 5, 5, 4, 7, 4, 5, 5, 0, 5, 4, 0, 0, 0, 4, 4, 0, 5, 5, 4, 5, 4, 4, 4, 4, 2, 7, 5, 5, 4, 4, 0, 5, 4, 0, 5, 4, 5, 5, 4, 4, 5, 4, 4, 5, 4, 4, 4, 4, 4, 5, 5, 4, 5, 5, 5, 4, 4, 4, 4, 4, 0, 0, 5, 4, 4, 5, 4, 5, 5, 4, 4, 4, 4, 5, 0, 4, 0, 5, 7, 5, 5, 4, 0, 4, 5, 0, 0, 0, 4, 5, 5, 5, 0, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 4, 7, 4, 5, 4, 0, 5, 4, 4, 7, 0, 4, 1, 5, 5, 5, 5, 4, 0, 4, 5, 4, 0, 5, 4, 4, 5, 2, 0, 0, 0, 4, 0, 0, 4, 5, 4, 0, 4, 5, 0, 2, 4, 4, 0, 4, 5, 5, 0, 5, 4, 5, 5, 4, 5, 7, 4, 5, 4, 4, 0, 4, 0, 0, 0, 4, 4, 4, 5, 5, 5, 4, 5, 7, 0, 0, 0, 4, 4, 4, 0, 2, 5, 4, 5, 5, 4, 5, 4, 0, 4, 4, 4, 5, 5, 5, 5, 0, 2, 0, 4, 4, 5, 5, 5, 0, 4, 5, 0, 0, 4, 5, 5, 5, 4, 5, 4, 5, 4, 4, 5, 0, 4, 5, 4, 0, 4, 5, 5, 5, 0, 4, 4, 5, 4, 2, 4, 0, 4, 4, 5, 4, 4, 4, 5, 5, 7, 0, 4, 4, 4, 0, 4, 0, 5, 4, 4, 2, 5, 4, 5, 4, 0, 5, 4, 4, 4, 5, 2, 4, 5, 4, 4, 4, 5, 4, 4, 7, 4, 4, 0, 4, 4, 4, 7, 7, 5, 5, 7, 4, 4, 5, 7, 4, 4, 2, 0, 5, 5, 5, 1, 4, 5, 5, 4, 4, 0, 5, 0, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 7, 5, 5, 5, 0, 5, 0, 4, 7, 5, 5, 7, 5, 4, 4, 5, 5, 5, 5, 5, 7, 0, 5, 5, 7, 5, 4, 7, 0, 4, 0, 5, 4, 0, 4, 4, 5, 4, 7, 4, 5, 4, 5, 0, 5, 5, 5, 0, 4, 4, 4, 0, 5, 5, 5, 0, 4, 0, 4, 0, 4, 4, 4, 5, 4, 5, 5, 4, 4, 5, 5, 0, 4, 4, 4, 5, 0, 5, 7, 0, 4, 4, 4, 7, 2, 5, 5, 5, 4, 5, 0, 4, 0, 0, 7, 4, 5, 4, 0, 5, 5, 5, 0, 4, 4, 5, 4, 4, 5, 4, 7, 4, 7, 4, 4, 4, 4, 0, 4, 0, 0, 0, 4, 4, 4, 7, 4, 4, 2, 4, 5, 4, 4, 5, 5, 4, 4, 0, 0, 4, 2, 0, 0, 0, 5, 5, 5, 4, 5, 2, 4, 5, 4, 5, 5, 4, 5, 4, 5, 2, 0, 4, 4, 4, 0, 5, 4, 5, 4, 4, 4, 5, 4, 4, 0, 2, 0, 4, 0, 0, 4, 0, 4, 0, 0, 5, 4, 5, 4, 0, 5, 4, 5, 5, 4, 4, 0, 4, 0, 2, 5, 4, 0, 4, 5, 4, 0, 4, 0, 0, 0, 5, 5, 7, 0, 5, 4, 0, 5, 5, 4, 0, 5, 4, 4, 5, 5, 0, 5, 0, 5, 4, 0, 5, 0, 4, 4, 4, 4, 0, 4, 5, 4, 0, 0, 4, 4, 5, 0, 5, 0, 4, 5, 4, 5, 4, 4, 5, 5, 0, 0, 5, 0, 0, 4, 4, 0, 0, 4, 5, 5, 0, 5, 1, 4, 4, 4, 4, 5, 0, 5, 4, 5, 5, 4, 4, 5, 4, 5, 4, 0, 5, 4, 0, 4, 5, 5, 4, 4, 4, 4, 5, 5, 5, 4, 4, 0, 2, 4, 5, 0, 5, 4, 5, 0, 5, 4, 5, 0, 5, 0, 5, 5, 7, 5, 5, 5, 0, 4, 4, 5, 5, 4, 4, 0, 5, 2, 0, 5, 4, 4, 4, 0, 5, 5, 4, 4, 4, 0, 4, 5, 0, 4, 5, 0, 5, 2, 0, 0, 4, 2, 4, 5, 4, 0, 0, 0, 0, 0, 0, 5, 7, 4, 0, 5, 5, 0, 4, 5, 4, 4, 5, 2, 5, 0, 0, 5, 4, 5, 4, 4, 4, 4, 4, 0, 5, 0, 5, 4, 7, 5, 4, 4, 0, 4, 0, 4, 5, 0, 4, 2, 4, 7, 5, 5, 0, 2, 4, 5, 4, 4, 7, 5, 0, 4, 0, 5, 5, 4, 4, 5, 0, 4, 5, 5, 7, 5, 1, 4, 5, 4, 5, 0, 0, 5, 2, 4, 0, 4, 5, 5, 4, 4, 7, 2, 4, 5, 7, 5, 4, 5, 4, 0, 5, 2, 1, 2, 5, 5, 0, 5, 4, 5, 0, 4, 0, 5, 5, 0, 5, 4, 4, 2, 0, 4, 4, 4, 4, 4, 0, 5, 5, 5, 1, 0, 5, 5, 5, 5, 5, 0, 5, 0, 5, 4, 5, 5, 4, 4, 5, 5, 0, 5, 4, 4, 0, 4, 4, 0, 4, 4, 7, 5, 0, 5, 5, 0, 0, 5, 5, 0, 7, 5, 4, 5, 4, 5, 4, 5, 4, 2, 4, 0, 4, 4, 5, 4, 2, 4, 0, 7, 5, 5, 5, 5, 4, 5, 5, 0, 0, 5, 4, 5, 4, 0, 2, 5, 4, 4, 4, 5, 0, 5, 0, 4, 4, 6, 7, 6, 6, 6, 6, 6, 7, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 5, 6, 5, 5, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 5, 5, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 5, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 1, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 7, 0, 6, 5, 6, 7, 6, 6, 6, 6, 7, 6, 6, 7, 5, 6, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 0, 6, 6, 6, 5, 6, 6, 7, 5, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 5, 6, 6, 6, 7, 7, 6, 6, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 0, 6, 7, 6, 6, 7, 5, 6, 6, 5, 6, 6, 6, 6, 6, 6, 5, 6, 7, 6, 6, 6, 6, 6, 6, 6, 1, 6, 5, 6, 7, 6, 6, 6, 6, 7, 0, 6, 5, 6, 6, 6, 6, 6, 6, 6, 5, 6, 7, 6, 6, 7, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 5, 6, 7, 6, 6, 6, 5, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 7, 7, 7, 1, 1, 7, 1, 1, 1, 7, 7, 7, 7, 1, 1, 7, 1, 7, 7, 1, 7, 1, 7, 7, 1, 7, 7, 7, 1, 1, 1, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 1, 1, 1, 7, 7, 1, 1, 7, 7, 7, 1, 7, 1, 7, 7, 1, 7, 1, 7, 1, 1, 7, 1, 7, 7, 1, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 5, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 1, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 1, 7, 1, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 1, 1, 1, 7, 7, 1, 1, 1, 7, 7, 7, 7, 1, 1, 1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 1, 1, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 1, 7, 1, 1, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 1, 1, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 1, 7, 7, 1, 7, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 1, 1, 1, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 1, 7, 7, 1, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 1, 7, 7, 1, 7, 7, 1, 1, 7, 7, 1, 1, 7, 1, 1, 7, 1, 7, 1, 7, 1, 7, 7, 1, 7, 1, 7, 7, 1, 1, 7, 7, 7, 7, 1, 1, 1, 1, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 7, 1, 1, 7, 1, 7, 1, 1, 7, 1, 1, 7, 7, 1, 7, 1, 7, 7, 1, 7, 7, 1, 7, 7, 1, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 1, 7, 7, 1, 7, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 1, 1, 7, 1, 1, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 1, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 1, 7, 5, 1, 7, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 1, 7, 1, 7, 1, 7, 7, 7, 7, 7, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 1, 7, 1, 7, 1, 7, 1, 1, 1, 7, 7, 1, 1, 7, 7, 7, 1, 7, 1, 7, 1, 1, 7, 1, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 1, 7, 7, 7, 7, 1, 1, 1, 1, 1, 7, 1, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 1, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 1, 7, 7, 7, 7, 1, 7, 1, 1, 7, 1, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 1, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 1, 7, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 1, 7, 1, 7, 1, 7, 7, 1, 7, 1, 1, 7, 7, 7, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 1, 7, 1, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 1, 1, 7, 7, 1, 1, 7, 7, 7, 7, 1, 5, 1, 1, 1, 7, 1, 1, 7, 1, 7, 7, 1, 7, 1, 7, 1, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 1, 1, 1, 7, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 1, 7, 1, 7, 7, 1, 1, 1, 7, 1, 7, 1, 1, 1, 7, 7, 1, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 7, 7, 1, 1, 5, 7, 1, 7, 1, 7, 1, 7, 7, 1, 7, 7, 1, 7, 7, 1, 7, 7, 7, 1, 7, 1, 1, 7, 7, 7, 1, 7, 1, 7, 1, 1, 1, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 7, 1, 1, 7, 1, 7, 7, 7, 7, 1, 7, 7, 1, 1, 1, 7, 7, 7, 7, 1, 1, 7, 7, 1, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 1, 1, 1, 7, 1, 7, 1, 1, 7, 7, 7, 1, 1, 7, 1, 7, 7, 1, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 1, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 1, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 7, 7, 1, 1, 1, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 1, 7, 1, 7, 1, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 1, 7, 1, 7, 1, 7, 1, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 1, 1, 1, 7, 1, 7, 7, 7, 7, 1, 7, 7, 1, 7, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 1, 1, 7, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 1, 1, 1, 1, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 1, 1, 7, 7, 1, 1, 1, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 7, 1, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7, 7, 1, 1, 1, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 1, 7, 1, 1, 7, 7, 7, 1, 7, 7, 7, 1, 1, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 7, 7, 7, 1, 1, 7, 7, 7, 1, 1, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 1, 7, 1, 1, 7, 7, 1, 7, 7, 7, 1, 1, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 1, 1, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 7, 1, 1, 7, 7, 1, 1, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 1, 1, 1, 7, 7, 1, 1, 7, 1, 1, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 1, 7, 1, 1, 7, 7, 7, 1, 7, 7, 1, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 1, 7, 7, 7, 7, 1, 1, 1, 7, 7, 1, 1, 7, 1, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 1, 7, 1, 1, 7, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 1, 7, 1, 1, 7, 7, 1, 1, 7, 7, 1, 1, 1, 1, 1, 7, 1, 7, 7, 1, 1, 7, 7, 7, 7, 1, 7, 1, 7, 7, 1, 7, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 7, 7, 1, 7, 7, 7, 1, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 7, 1, 1, 7, 7, 7, 7, 1, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 1, 7, 1, 1, 1, 7, 7, 1, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 1, 1, 7, 1, 7, 7, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 1, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 1, 7, 1, 1, 1, 7, 1, 7, 7, 7, 1, 7, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7, 1, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 1, 1, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 1, 1, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 7, 7, 1, 1, 7, 1, 7, 1, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 1, 7, 1, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 1, 1, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 1, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 1, 1, 1, 1, 7, 7, 7, 1, 7, 1, 1, 1, 7, 7, 1, 1, 1, 7, 7, 7, 1, 1, 7, 1, 1, 1, 7, 7, 7, 1, 7, 7, 1, 7, 1, 7, 1, 7, 7, 1, 1, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 1, 1, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 1, 7, 7, 1, 7, 1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 5, 7, 7, 7, 1, 7, 7, 1, 7, 7, 1, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 1, 1, 7, 1, 7, 1, 1, 7, 1, 1, 1, 1, 1, 7, 1, 1, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 1, 1, 1, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 7, 1, 7, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 1, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 7, 7, 7, 7, 7, 1, 1, 1, 7, 1, 7, 7, 1, 7, 1, 7, 1, 1, 1, 7, 7, 7, 1, 7, 7, 7, 1, 7, 1, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 1, 7, 7, 1, 1, 1, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 7, 7, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 7, 1, 1, 7, 7, 7, 1, 7, 1, 7, 1, 7, 1, 1, 7, 1, 1, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 1, 7, 7, 1, 1, 1, 7, 0, 4, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 5, 0, 0, 7, 0, 0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 7, 4, 5, 4, 0, 0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 4, 0, 7, 4, 0, 4, 0, 2, 4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 4, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 7, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 8, 0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 4, 0, 5, 4, 7, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 7, 4, 0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 4, 5, 8, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 7, 0, 0, 0, 4, 5, 0, 4, 5, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 5, 0, 0, 0, 4, 0, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 5, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 5, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 5, 5, 0, 5, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 4, 7, 0, 8, 4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 5, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 4, 7, 5, 0, 0, 0, 0, 0, 0, 4, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 4, 0, 0, 4, 4, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 8, 0, 0, 5, 0, 5, 8, 5, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 4, 0, 4, 0, 0, 5, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 5, 2, 5, 0, 0, 0, 5, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 4, 0, 4, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 5, 0, 0, 4, 0, 7, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 4, 0, 0, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 4, 0, 4, 5, 0, 4, 0, 0, 0, 0, 6, 5, 4, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 4, 5, 0, 0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 8, 0, 4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 6, 0, 0, 4, 0, 4, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 5, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 5, 0, 0, 4, 0, 4, 4, 0, 0, 0, 0, 0, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 2, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4, 4, 4, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 4, 5, 0, 0, 0, 4, 0, 0, 0, 0, 4, 5, 0, 5, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 4, 4, 4, 7, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 5, 4, 0, 0, 5, 4, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 7, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 4, 5, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 7, 5, 0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 0, 4, 0, 0, 0, 0, 5, 0, 4, 0, 5, 0, 7, 0, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 4, 7, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 7, 0, 0, 0, 5, 0, 5, 4, 5, 0, 0, 0, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 4, 0, 0, 5, 0, 4, 4, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 0, 4, 4, 5, 0, 0, 0, 0, 4, 7, 0, 5, 0, 0, 5, 0, 0, 0, 0, 2, 0, 4, 4, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 5, 7, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 5, 5, 0, 4, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 7, 0, 5, 7, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 0, 4, 5, 4, 0, 0, 7, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 5, 5, 0, 0, 0, 0, 5, 0, 0, 0, 4, 0, 5, 0, 4, 0, 4, 0, 0, 0, 5, 0, 0, 4, 5, 0, 5, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 4, 4, 4, 0, 5, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 4, 0, 7, 0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 7, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 4, 5, 0, 0, 5, 0, 8, 0, 4, 4, 8, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 4, 0, 7, 4, 0, 5, 2, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 5, 0, 5, 0, 0, 0, 0, 0, 4, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 5, 5, 0, 7, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 5, 0, 0, 4, 5, 0, 0, 0, 4, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 7, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 8, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 5, 4, 4, 0, 0, 0, 0, 0, 7, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 2, 0, 0, 0, 1, 0, 0, 0, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 7, 7, 0, 0, 7, 0, 0, 1, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 7, 2, 7, 1, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 2, 0, 0, 0, 7, 7, 7, 0, 4, 0, 7, 7, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 7, 4, 0, 0, 0, 4, 0, 0, 1, 7, 7, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 7, 0, 0, 0, 4, 0, 0, 1, 0, 0, 0, 7, 0, 4, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 7, 0, 0, 4, 0, 2, 0, 0, 4, 1, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 4, 0, 0, 7, 7, 0, 7, 0, 0, 0, 0, 7, 0, 0, 7, 2, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 7, 0, 7, 0, 0, 0, 7, 0, 0, 0, 1, 0, 7, 1, 0, 0, 7, 4, 7, 0, 0, 1, 0, 0, 0, 0, 7, 0, 2, 0, 7, 0, 0, 7, 0, 7, 0, 7, 7, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 7, 0, 2, 0, 7, 0, 0, 1, 7, 0, 0, 0, 0, 7, 0, 7, 7, 0, 7, 0, 0, 1, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 4, 7, 0, 0, 7, 4, 0, 0, 0, 0, 0, 7, 0, 0, 7, 4, 1, 0, 4, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 8, 0, 7, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 4, 0, 0, 0, 0, 0, 0, 7, 2, 0, 0, 0, 0, 7, 2, 5, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 0, 5, 0, 7, 0, 0, 0, 0, 7, 0, 7, 0, 4, 0, 0, 2, 0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 0, 0, 7, 0, 0, 0, 7, 0, 7, 7, 0, 0, 0, 2, 7, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 7, 0, 5, 0, 7, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 7, 2, 7, 2, 7, 0, 0, 7, 0, 7, 7, 0, 4, 0, 0, 0, 1, 7, 0, 0, 0, 7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 7, 0, 0, 2, 0, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 2, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 4, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 7, 0, 0, 1, 0, 0, 7, 0, 0, 4, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 7, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 7, 0, 0, 0, 0, 1, 7, 0, 0, 0, 0, 7, 0, 7, 0, 0, 7, 0, 0, 0, 2, 0, 0, 7, 0, 0, 5, 0, 7, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 4, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 7, 0, 2, 0, 1, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 5, 0, 7, 0, 0, 7, 0, 0, 7, 4, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 1, 7, 0, 0, 0, 5, 0, 7, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 2, 0, 0, 7, 0, 0, 0, 7, 5, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 4, 1, 0, 0, 7, 0, 1, 7, 0, 7, 0, 0, 7, 0, 0, 0, 7, 0, 4, 0, 0, 7, 7, 0, 0, 0, 0, 7, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 4, 0, 5, 1, 2, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 4, 0, 4, 7, 0, 0, 2, 0, 0, 0, 0, 7, 7, 0, 0, 0, 7, 2, 0, 5, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 7, 0, 0, 0, 7, 0, 0, 5, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 7, 1, 0, 4, 7, 7, 0, 0, 7, 0, 0, 7, 0, 7, 0, 7, 2, 0, 0, 0, 0, 7, 0, 5, 0, 7, 0, 0, 4, 0, 0, 4, 0, 4, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 2, 0, 0, 2, 0, 2, 0, 2, 4, 0, 0, 2, 0, 4, 0, 0, 0, 2, 0, 0, 4, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 4, 2, 4, 0, 4, 4, 2, 0, 4, 2, 4, 0, 0, 2, 4, 0, 0, 0, 4, 4, 2, 0, 4, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 4, 0, 4, 2, 0, 0, 7, 0, 4, 0, 4, 4, 0, 0, 4, 0, 0, 0, 5, 0, 0, 4, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 7, 4, 0, 0, 0, 4, 2, 0, 0, 0, 2, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 5, 4, 4, 0, 4, 4, 4, 0, 4, 0, 0, 0, 4, 2, 4, 0, 0, 4, 4, 0, 4, 4, 4, 0, 4, 0, 0, 0, 4, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 5, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 2, 0, 4, 4, 2, 0, 4, 0, 2, 0, 4, 0, 2, 4, 0, 0, 0, 0, 0, 2, 0, 6, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 4, 0, 0, 0, 1, 4, 0, 4, 0, 0, 0, 0, 4, 0, 4, 4, 4, 0, 0, 4, 0, 0, 2, 2, 4, 0, 0, 0, 4, 4, 4, 4, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 0, 4, 2, 0, 4, 2, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0, 4, 0, 4, 0, 2, 0, 7, 2, 0, 4, 2, 0, 4, 0, 4, 0, 6, 0, 4, 4, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 2, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 4, 0, 4, 0, 2, 0, 0, 4, 0, 0, 0, 0, 4, 2, 0, 0, 0, 0, 4, 0, 2, 0, 4, 0, 0, 0, 0, 4, 2, 0, 7, 2, 2, 0, 0, 0, 0, 4, 0, 0, 4, 0, 4, 0, 2, 0, 4, 0, 0, 2, 4, 0, 0, 0, 0, 4, 2, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 4, 4, 4, 2, 0, 4, 5, 0, 2, 4, 0, 0, 4, 4, 0, 0, 4, 2, 0, 0, 0, 0, 4, 0, 2, 0, 4, 0, 0, 4, 4, 4, 0, 4, 0, 0, 0, 0, 2, 4, 0, 4, 0, 4, 4, 4, 0, 0, 4, 0, 0, 4, 4, 4, 4, 0, 4, 0, 2, 0, 0, 0, 4, 0, 4, 0, 0, 2, 4, 2, 2, 0, 0, 0, 7, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 4, 0, 2, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 2, 0, 4, 1, 2, 4, 4, 4, 0, 4, 2, 4, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 2, 0, 4, 0, 0, 4, 4, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 4, 2, 0, 4, 0, 4, 4, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 2, 4, 2, 0, 4, 0, 0, 0, 2, 4, 0, 4, 4, 4, 2, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4, 2, 4, 4, 1, 2, 0, 2, 0, 0, 0, 4, 2, 0, 2, 0, 4, 0, 0, 4, 0, 2, 2, 4, 0, 2, 4, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 2, 0, 4, 0, 4, 4, 0, 4, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 4, 4, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 4, 0, 4, 4, 2, 4, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 2, 0, 0, 7, 4, 4, 4, 2, 0, 0, 0, 2, 0, 2, 4, 4, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 4, 0, 0, 2, 4, 4, 0, 0, 4, 4, 4, 0, 4, 0, 4, 2, 2, 2, 0, 4, 0, 0, 2, 4, 4, 4, 0, 4, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 2, 0, 4, 0, 4, 0, 0, 0, 0, 2, 0, 4, 4, 0, 4, 0, 0, 0, 6, 0, 4, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 4, 0, 4, 2, 0, 0, 0, 4, 6, 2, 0, 0, 0, 2, 0, 0, 2, 2, 4, 0, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 2, 0, 4, 2, 0, 4, 0, 0, 0, 2, 2, 4, 0, 0, 4, 4, 0, 0, 0, 0, 2, 2, 0, 0, 4, 4, 0, 4, 4, 0, 2, 4, 0, 0, 4, 2, 4, 0, 0, 4, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 4, 4, 4, 4, 4, 0, 0, 5, 0, 2, 0, 6, 0, 4, 0, 4, 2, 0, 8, 4, 0, 0, 0, 0, 2, 0, 0, 4, 2, 0, 0, 2, 2, 0, 0, 4, 4, 0, 0, 2, 4, 0, 0, 4, 4, 4, 0, 0, 0, 4, 2, 0, 0, 2, 0, 4, 0, 0, 4, 4, 4, 0, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 2, 0, 0, 0, 4, 0, 4, 4, 0, 0, 4, 4, 0, 2, 1, 0, 0, 0, 0, 2, 0, 4, 2, 0, 4, 0, 4, 0, 0, 4, 2, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 2, 4, 4, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 0, 4, 4, 4, 4, 6, 0, 4, 0, 4, 0, 4, 0, 4, 4, 0, 4, 7, 0, 4, 0, 4, 0, 0, 0, 0, 4, 0, 6, 0, 4, 0, 0, 0, 0, 4, 4, 4, 2, 0, 4, 4, 0, 0, 2, 4, 0, 0, 0, 4, 4, 0, 4, 0, 4, 0, 0, 4, 6, 0, 0, 0, 4, 0, 4, 4, 2, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 4, 4, 0, 4, 0, 0, 2, 0, 0, 4, 2, 2, 0, 0, 0, 0, 0, 2, 0, 4, 4, 0, 0, 4, 0, 4, 2, 4, 4, 4, 0, 0, 7, 0, 0, 0, 0, 0, 2, 4, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 0, 0, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 2, 0, 4, 0, 2, 0, 7, 7, 4, 0, 0, 0, 4, 4, 0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 2, 2, 0, 4, 4, 4, 4, 0, 0, 2, 4, 0, 4, 5, 4, 0, 4, 2, 0, 7, 2, 2, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 2, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 4, 4, 2, 4, 0, 4, 0, 1, 0, 0, 4, 0, 4, 0, 4, 0, 0, 0, 2, 4, 0, 4, 0, 4, 2, 4, 0, 0, 4, 0, 0, 2, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 4, 0, 2, 0, 4, 0, 2, 0, 0, 4, 2, 2, 4, 4, 4, 4, 0, 4, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 2, 4, 0, 4, 2, 0, 4, 0, 0, 4, 2, 5, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 4, 0, 0, 4, 2, 2, 0, 0, 0, 0, 4, 2, 1, 7, 1, 7, 2, 2, 7, 2, 7, 7, 2, 7, 7, 7, 5, 7, 1, 7, 1, 2, 2, 6, 7, 2, 7, 7, 2, 6, 2, 2, 7, 2, 7, 2, 2, 2, 7, 7, 7, 2, 7, 7, 4, 4, 4, 7, 4, 0, 4, 4, 5, 4, 0, 4, 7, 5, 5, 7, 0, 7, 4, 4, 4, 7, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 0, 4, 4, 4, 7, 5, 5, 4, 7, 7, 7, 4, 8, 4, 7, 5, 4, 4, 4, 5, 0, 0, 4, 4, 0, 7, 4, 4, 7, 5, 4, 4, 4, 4, 8, 7, 0, 4, 4, 0, 0, 4, 2, 4, 4, 4, 4, 4, 5, 7, 4, 7, 4, 4, 0, 0, 4, 4, 5, 4, 7, 4, 4, 0, 0, 7, 8, 7, 4, 4, 4, 4, 4, 0, 7, 4, 8, 4, 4, 4, 0, 0, 4, 4, 0, 0, 4, 0, 4, 4, 4, 4, 8, 0, 4, 4, 4, 0, 5, 4, 4, 0, 4, 8, 8, 0, 4, 5, 0, 4, 4, 4, 4, 4, 7, 0, 4, 4, 4, 7, 7, 4, 0, 4, 4, 4, 5, 7, 4, 7, 7, 5, 7, 4, 5, 4, 4, 4, 4, 4, 0, 4, 4, 0, 4, 7, 5, 0, 5, 4, 4, 7, 4, 5, 4, 4, 2, 4, 4, 4, 4, 4, 8, 5, 4, 4, 4, 7, 7, 7, 4, 5, 0, 4, 5, 0, 0, 4, 4, 4, 0, 4, 0, 4, 4, 7, 4, 4, 4, 4, 5, 7, 4, 4, 0, 7, 4, 0, 0, 7, 4, 4, 4, 4, 4, 8, 4, 5, 4, 4, 5, 4, 4, 4, 7, 4, 4, 4, 7, 4, 4, 0, 4, 7, 4, 4, 8, 4, 7, 4, 4, 4, 4, 2, 4, 4, 7, 4, 4, 4, 4, 0, 0, 5, 4, 4, 4, 7, 5, 0, 4, 7, 5, 4, 7, 4, 4, 0, 4, 8, 0, 7, 4, 8, 8, 4, 4, 0, 4, 4, 4, 0, 0, 0, 4, 7, 4, 0, 0, 4, 4, 4, 4, 1, 4, 0, 4, 4, 5, 0, 4, 8, 8, 8, 5, 7, 8, 4, 4, 4, 5, 0, 0, 0, 5, 7, 4, 4, 4, 4, 4, 4, 4, 8, 0, 4, 4, 7, 7, 4, 4, 4, 4, 4, 0, 4, 5, 4, 4, 4, 4, 4, 4, 4, 0, 4, 7, 1, 4, 5, 8, 4, 0, 7, 4, 4, 7, 8, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 0, 0, 0, 4, 4, 7, 5, 5, 4, 8, 4, 4, 4, 4, 4, 7, 4, 5, 4, 5, 4, 0, 4, 0, 4, 5, 4, 7, 0, 5, 4, 7, 4, 4, 4, 4, 7, 4, 4, 7, 7, 4, 4, 7, 0, 0, 2, 5, 7, 4, 8, 4, 8, 4, 4, 0, 7, 7, 4, 4, 4, 4, 0, 4, 4, 7, 4, 4, 4, 7, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 0, 0, 7, 0, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 8, 4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 5, 4, 4, 5, 4, 4, 4, 4, 4, 1, 0, 4, 0, 4, 7, 4, 4, 4, 4, 4, 4, 7, 5, 4, 4, 7, 4, 0, 7, 7, 4, 4, 7, 4, 7, 4, 0, 7, 4, 7, 4, 2, 2, 7, 7, 4, 8, 4, 4, 5, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 0, 0, 0, 7, 7, 4, 4, 0, 4, 4, 0, 4, 4, 7, 0, 4, 7, 4, 0, 4, 7, 4, 0, 7, 5, 4, 4, 4, 7, 7, 7, 4, 8, 7, 2, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 7, 7, 4, 0, 4, 8, 0, 7, 7, 4, 7, 7, 0, 7, 4, 4, 7, 4, 4, 5, 4, 4, 4, 4, 4, 0, 5, 4, 4, 0, 0, 0, 7, 7, 4, 7, 8, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 7, 4, 4, 5, 7, 7, 4, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 8, 4, 4, 4, 4, 5, 7, 4, 0, 4, 4, 5, 0, 4, 4, 4, 7, 4, 0, 7, 4, 4, 5, 4, 4, 4, 8, 0, 4, 8, 4, 4, 4, 4, 0, 5, 8, 4, 0, 0, 4, 4, 4, 4, 4, 0, 0, 8, 7, 7, 0, 4, 4, 7, 4, 4, 7, 5, 4, 4, 8, 2, 8, 8, 7, 2, 8, 7, 8, 7, 1, 7, 8, 7, 7, 2, 2, 7, 7, 7, 1, 5, 5, 1, 2, 7, 2, 7, 2, 8, 7, 7, 7, 7, 8, 7, 8, 1, 2, 8, 7, 1, 7, 1, 8, 8, 2, 1, 1, 8, 2, 1, 8, 2, 8, 1, 7, 5, 7, 1, 8, 5, 8, 8, 2, 8, 7, 7, 8, 2, 7, 8, 2, 5, 8, 8, 2, 7, 2, 7, 8, 2, 8, 2, 3, 8, 8, 8, 1, 7, 7, 7, 8, 7, 7, 7, 8, 1, 1, 1, 7, 1, 8, 7, 2, 2, 7, 1, 2, 2, 7, 7, 8, 2, 1, 8, 1, 7, 7, 1, 7, 2, 7, 7, 2, 2, 1, 8, 7, 8, 5, 7, 7, 7, 2, 7, 2, 8, 2, 7, 8, 2, 1, 1, 7, 7, 7, 5, 2, 7, 1, 1, 7, 8, 7, 2, 7, 8, 8, 7, 2, 8, 7, 7, 8, 8, 1, 7, 1, 2, 7, 7, 1, 5, 1, 2, 8, 7, 8, 8, 2, 1, 7, 5, 2, 1, 1, 2, 2, 7, 8, 8, 1, 8, 8, 8, 8, 2, 8, 8, 1, 2, 8, 5, 2, 7, 1, 1, 2, 8, 8, 2, 8, 7, 2, 7, 7, 2, 8, 2, 2, 8, 1, 5, 7, 7, 8, 1, 8, 8, 8, 8, 8, 2, 7, 1, 8, 7, 1, 1, 2, 2, 2, 2, 7, 8, 7, 5, 2, 8, 8, 8, 7, 1, 5, 1, 1, 2, 8, 8, 7, 1, 1, 8, 1, 1, 7, 8, 8, 8, 1, 7, 2, 7, 8, 8, 2, 8, 8, 8, 8, 8, 1, 7, 8, 7, 5, 8, 8, 7, 8, 1, 5, 2, 5, 8, 2, 2, 7, 8, 8, 7, 1, 8, 2, 7, 1, 8, 8, 2, 8, 7, 1, 8, 7, 8, 2, 8, 2, 8, 8, 8, 7, 2, 8, 1, 8, 7, 1, 8, 2, 5, 7, 8, 8, 7, 8, 7, 7, 8, 8, 1, 7, 1, 8, 7, 7, 8, 1, 8, 8, 2, 2, 7, 8, 8, 1, 2, 8, 8, 2, 2, 8, 8, 7, 2, 7, 0, 1, 7, 8, 8, 8, 8, 5, 2, 2, 1, 1, 2, 8, 1, 1, 7, 7, 2, 1, 8, 7, 8, 1, 7, 2, 1, 1, 7, 7, 1, 1, 8, 7, 7, 7, 2, 2, 8, 7, 7, 8, 7, 8, 8, 2, 8, 8, 7, 8, 7, 7, 1, 2, 8, 1, 8, 1, 7, 8, 5, 7, 2, 8, 5, 1, 2, 2, 2, 8, 1, 7, 2, 5, 8, 8, 7, 8, 8, 7, 8, 1, 7, 8, 8, 8, 2, 2, 2, 7, 7, 2, 2, 1, 2, 2, 2, 1, 2, 1, 8, 1, 1, 2, 8, 7, 8, 7]
with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/Net_Vgg16.csv"), 'w', newline='') as f2:
    # for i, s in enumerate(cls.labels_):
    for i, s in enumerate(data):
        f2.write(str(s))
        f2.write('\n')