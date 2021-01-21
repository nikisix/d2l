
"""
1. We conducted  𝑚=500  groups of experiments where each group draws  𝑛=10  samples. Vary  𝑚  and  𝑛 . Observe and analyze the experimental results.

2. Given two events with probability  𝑃(A)  and  𝑃(B) , compute upper and lower bounds on  𝑃(A∪ B)  and  𝑃(A∩ B) . (Hint: display the situation using a Venn Diagram.)

3. Assume that we have a sequence of random variables, say  𝐴 ,  𝐵 , and  𝐶 , where  𝐵  only depends on  𝐴 , and  𝐶  only depends on  𝐵 , can you simplify the joint probability  𝑃(𝐴,𝐵,𝐶) ? (Hint: this is a Markov Chain.)

4. In Section 2.6.2.6, the first test is more accurate. Why not run the first test twice rather than run both the first and second tests?
"""


import numpy as np

# Q1. 𝑚=500  groups of experiments where each group draws  𝑛=10  samples
exps = np.random.multinomial(10, [1/6]*6, size=500)
exps.sum()/len(exps)
exps.sum(axis=0)/len(exps)
# Out[20]: array([1.688, 1.6  , 1.63 , 1.58 , 1.724, 1.778])
# about the 1/6 chance for throwing a dice


# Q2. 𝑃(A∪ B)  and  𝑃(A∩ B) upper and lower bounds
"""
𝑃(A∪ B) upper bound
    P(A) + P(B), when A and B indep.

𝑃(A∪ B) lower bound
    The lesser of P(A) or P(B), when A and B completely dep.

𝑃(A∩ B) upper bound
    The lesser of The lesser of P(A) or P(B), when dep.

𝑃(A∩ B) upper bound
    0 when indep.
"""

# Q3.
# P(C|B) * P(B|A) * P(A)
