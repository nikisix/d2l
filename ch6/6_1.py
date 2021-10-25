from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class LayerReduction(nn.Block) :
    def __init__(self, t_input, weight_scale=10, **kwargs) :
        super().__init__(**kwargs)
        self._dim_i ,self._dim_j, self._dim_k = t_input

        #Weight initialization
        self._weight = self.params.get('weight', shape=t_input)
        self._weight.initialize(init = init.Uniform(scale=weight_scale))

        #Reduced weights initialization
        self._reducedWeight = self.params.get('weightReduced', shape=self._dim_k)
        self._reducedWeight.initialize(init = init.Uniform(scale=0))

        # Can also apply
        # self._reducedWeight.data()[:] = 0.


    def forward(self, X) :
        '''
        Calculates tensor reduction with following steps :
            * Step 1 : A_ik = Sum_j W_ijk . X_j
            * Step 2 : Sum_i A_ik . X_i
            INPUT : 
                *  X : (i,j)
            OUTPUT : array of dimension k
        '''
        Xi = X[:self._dim_i]
        Xj = X[self._dim_i:self._dim_i+self._dim_j]
        Wki = np.zeros((dim_k,dim_i))
        for k in range(dim_k) :
            Wk_ij = self._weight.data()[:,:,k] #(i,j)
            Wki[k,:] = np.dot(Wk_ij,Xj)#(k,i)
        Wki.shape, Xj.shape
        self._reducedWeight.data()[:] = np.dot(Wki,Xi)
        return self._reducedWeight.data()


t_input = (2,3,4)
dim_i, dim_j, dim_k = t_input
net = LayerReduction(t_input)
X = np.random.normal(0.,1.,(dim_i+dim_j))
print(net(X))
