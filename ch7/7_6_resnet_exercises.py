""" 
7.6.6. Exercises
    1. What are the major differences between the Inception block in Fig. 7.4.1 and the residual block?
        After removing some paths in the Inception block, how are they related to each other?

    2. Refer to Table 1 in the ResNet paper [He et al., 2016a] to implement different variants.

    3. For deeper networks, ResNet introduces a “bottleneck” architecture to reduce model complexity. Try to implement it.

    4. In subsequent versions of ResNet, the authors changed the “convolution, batch normalization, and activation”
        structure to the “batch normalization, activation, and convolution” structure.
        Make this improvement yourself.
        See Figure 1 in [He et al., 2016b] for details.

    5. Why can’t we just increase the complexity of functions without bound, even if the function classes are nested?


1. What are the major differences between the Inception block in Fig. 7.4.1 and the residual block?
    After removing some paths in the Inception block, how are they related to each other?

Differences
    Inception concatenates the psuedo-skip connection with the output of the rest of the inception block, while
    resnet directly sums the input-vector with the block-output.
    Inception does not make use of batch normalization.

Similarities
    Both contain a 1x1 Convolution skip connection at times (about every third time in the case of resnet).
    Both begin with a 7x7 Convolution into a 3x3 MaxPool layer.

2. Refer to Table 1 in the ResNet paper [He et al., 2016a] to implement different variants.
    Can implement resnet 34 by adjusting the block-counts.

3. For deeper networks, ResNet introduces a “bottleneck” architecture to reduce model complexity. Try to implement it.
    See resnet-56 on Table 1: https://arxiv.org/pdf/1512.03385.pdf or ./resnet-architectures.png
    See ./7_6_resnet50.py

5. Why can’t we just increase the complexity of functions without bound, even if the function classes are nested?
    Overfitting. Training feasibility. """
