# TASK 1
"""
A classification task with two predictor variables given and entropy is used as a criterion for making the binary splits. 
There are 20 samples (11 ones are in class 1, 9 ones are in class 0). 
The best split by the first predictor gives ((5, 8), (6, 1)), by the second one - ((2, 7), (9, 2)) relatively. 
What split should be chosen?
"""

import numpy as np

source_entropy = -(9/20)*np.log(9/20) - (11/20)*np.log(11/20)
print('source_entropy', source_entropy)
left_entropy1 = -(5/13)*np.log(5/13) - (8/13)*np.log(8/13)
right_entropy1 = -(6/7)*np.log(6/7) - (1/7)*np.log(1/7)
weighted_entropy1 = (13/20)*left_entropy1 + (7/20)*right_entropy1
print('weighted entropy 1', weighted_entropy1)
left_entropy2 = -(2/9)*np.log(2/9) - (7/9)*np.log(7/9)
right_entropy2 = -(9/11)*np.log(9/11) - (2/11)*np.log(2/11)
weighted_entropy2 = (9/20)*left_entropy2 + (11/20)*right_entropy2
print('weighted entropy 2', weighted_entropy2)
info_gain = source_entropy - weighted_entropy2
print('Information gain', info_gain, round(info_gain, 3))

# TASK 2
"""
What is the value of the information gain for the best split from previous question?
Round according to the mathematical rules of rounding to 3 decimal places. 
Make sure you take natural logarithm (aka np.log()) and NOT logarithm base of two!
"""

left_gini = 1 - (2/9)**2 - (7/9)**2
right_gini = 1 - (9/11)**2 - (2/11)**2
weighted_gini = (9/20)*left_gini + (11/20)*right_gini
print('weighted gini', weighted_gini)
source_gini = 1 - (11/20)**2 - (9/20)**2
gini_gain = source_gini - weighted_gini
print('gini_gain', gini_gain, round(gini_gain, 3))