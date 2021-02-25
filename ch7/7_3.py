""" 7.3.5. Exercises
1. Tune the hyperparameters to improve the classification accuracy.
2. Why are there two 1×1 convolutional layers in the NiN block? Remove one of them, and then observe and analyze the experimental phenomena.

3. Calculate the resource usage for NiN.
    What is the number of parameters?
    What is the amount of computation?
    What is the amount of memory needed during training?
    What is the amount of memory needed during prediction?

4. What are possible problems with reducing the 384×5×5 representation to a 10×5×5 representation in one step?
"""

"""
2. Why are there two 1×1 convolutional layers in the NiN block? Remove one of them, and then observe and analyze the
experimental phenomena.

Two layers are needed to capture non-linear relationships between the feature representations.
"""

"""
3. Calculate the resource usage for NiN.
    What is the number of parameters?
    What is the amount of computation?
    What is the amount of memory needed during training?
    What is the amount of memory needed during prediction?

Unrelatedly, here are my calculations for the resource usage of the NiN in the textbook chapter:

    INPUT: 224 x 224 x 1    ACTIVS: 224 * 224        PARAMS: 0
CONV(1,96,11,4,0)        ACTIVS: 96 * 54 * 54    PARAMS: (11*11*1)*96
CONV(96,96,1,1,0)        ACTIVS: 96 * 54 * 54    PARAMS: (1*1*96)*96
CONV(96,96,1,1,0)        ACTIVS: 96 * 54 * 54    PARAMS: (1*1*96)*96
MaxPool(3,2)            ACTIVS: 96 * 26 * 26      PARAMS: 0
NiNBlock(96,256,5,1,2)    ACTIVS: 3*(256 * 26 * 26)   PARAMS: 256 * (256+256+(5*5*96))                                                                                     
MaxPool(3,2)            ACTIVS: 256 * 12 * 12    PARAMS: 0
NiNBlock(256,384,3,1,1)    ACTIVS: 3*(384 * 12 * 12)    PARAMS: 384 * (384+384+(3*3*256))                                                                                                                        
MaxPool(3,2)            ACTIVS:    384 * 5 * 5        PARAMS: 0
Dropout                ACTIVS: 384 * 5 * 5        PARAMS: 0
NiNBlock(384,10,3,1,1)    ACTIVS: 3*(10 * 5 * 5)        PARAMS: 10 * (10+10+(3*3*384))                                                                                                                        
AdaptiveMaxPool        ACTIVS: 10                PARAMS: 0
Flatten                ACTIVS: 10                PARAMS: 0

When training: we need 2 * the ACTIVS sum (values + gradients), and 3 * the PARAMS sum (values, gradients, and a cache
        for momentum/Adam)

When testing: we just need the sum of activs + params. If we’re being clever we can erase previous ACTIVS as we go, so
we only need the sum of the largest two consecutive ACTIVS. We also don’t need the Dropout ACTIVS.
"""

""" 4. What are possible problems with reducing the 384×5×5 representation to a 10×5×5 representation in one step?

Many important feature representations may be improperly summarized """
