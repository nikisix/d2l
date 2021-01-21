"""3.6.9. Exercises
1. In this section, we directly implemented the softmax function based on the mathematical definition of the softmax operation. What problems might this cause? Hint: try to calculate the size of  exp(50) .

    Numerical Over/Under Flow


2. The function cross_entropy in this section was implemented according to the definition of the cross-entropy loss function. What could be the problem with this implementation? Hint: consider the domain of the logarithm.

    def cross_entropy(y_hat, y):
        return - np.log(y_hat[range(len(y_hat)), y])

    If y_hat contains any negative values.


3. What solutions can you think of to fix the two problems above?

    Input screening:
    1.
        def softmax(X):
            X_exp = np.exp(X)
            partition = X_exp.sum(1, keepdims=True)
            return X_exp / partition  # The broadcasting mechanism is applied here

        Maybe we could log(X) before exp'ing it?
        "First subtract  max(𝑜𝑘)  from all  𝑜𝑘  before proceeding with the softmax calculation" - From section 3.7

    2. break if a negative value is sent it. Or y_hat = abs(y_hat)?


4. Is it always a good idea to return the most likely label? For example, would you do this for medical diagnosis?

    No, sometimes the false-negative is much worse than the false-positive. So we would skew our predictions to accomodate.


5. Assume that we want to use softmax regression to predict the next word based on some features. What are some problems that might arise from a large vocabulary?

    Input and Weight matrix would have to be of dimension of the number of words in the vocabulary.
    Word embeddings would offer a nice approach.
"""
