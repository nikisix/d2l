import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def display2D_netOutput(net, list_layer, X_test, y_test, nb_sample , is_prediction = False, figsize=(5,5)) :
    '''Display a grid of pictures resulting in the processing of images with a trained convolutional network.
    Image is feeded into convolutional layer and result is displayed for each layer.
    Each row from grid of display matches with a layer.
    
    INPUT
        * net           : convolutional network containing
        * list_layer    : list of layers after which images are displayed
        * X_test        : image to be displayed.
        * y_test        : labels of images 
        * nb_sample     : number of images to be displayed
        * is_prediction : when True, this flag allows to display images issued from each layer.
        * figsize       : diaplay panel shape

     OUTPUT : none
    '''
    
    nb_layer = len(list_layer)
    #-----------------------------------------------------------------------
    # Define grid of bounding box where pictures take place
    #-----------------------------------------------------------------------
    f, axarr = plt.subplots(nb_layer, nb_sample) 
    _=plt.figure(figsize=figsize)
    
    #-----------------------------------------------------------------------
    # Get labels matching with X_test samples 
    #-----------------------------------------------------------------------
    list_label = d2l.get_fashion_mnist_labels(y_test)

    for i_layer in range(nb_layer) :
        #-------------------------------------------------------------------------------
        # Get all layers from net until limit i_layer, start counting from 1st layer
        #-------------------------------------------------------------------------------
        netconv = net[:i_layer]

        #-------------------------------------------------------------------------------
        # Input data : 
        # Get shape of input data and aggregate the samples as input data in a list.
        # Each sample from batch is reshaped as (1,nb_channels,t_shape_inputData)
        # This reshape allows to feed netconv.
        # Each feature-map is agregated into a list.
        #
        # For each input data element from batch (sized as batch_size) : 
        #    * Get feature-maps using netconv; input data is reshaped to be tensor 
        #      formated as (1, nb_channels, heigth, width)
        #-------------------------------------------------------------------------------
        t_shape_inputData =(X_test.shape[-2], X_test.shape[-1])
        list_X_hat = [ netconv(X_test[i_sample].reshape((1,1)+t_shape_inputData)) for i_sample in range(nb_sample)]


        #-------------------------------------------------------------------------------
        # Output data : 
        # Get shape of 2D feature-map to be displayed : they are last 2 dimensions 
        # of feature map.
        # The 1st feature-map from list allows to calculate feature-map shape.
        #-------------------------------------------------------------------------------
        t_shape_outputData =(list_X_hat[0].shape[-2], list_X_hat[0].shape[-1])
        X_hat_reshaped = list_X_hat[0][0][0].reshape(t_shape_outputData)
        


        for i_sample in range(nb_sample) :
            #-------------------------------------------------------------------------------
            # Aggregate feature-maps with bitwise sum over all of them
            #-------------------------------------------------------------------------------
            X_sum = np.zeros(t_shape_outputData)

            X_feature_map = list_X_hat[i_sample]
            nb_feature = X_feature_map[0].shape[0]
            
            for i_feature in range(nb_feature) :
                X_sum += X_feature_map[0][i_feature]
                
            axarr[i_layer,i_sample].set(label=list_label[i_sample])    

            axImage = axarr[i_layer,i_sample].imshow(X_sum.asnumpy().reshape(t_shape_outputData))
            if 0 == i_layer :
                axarr[i_layer,i_sample].text(0, -2, list_label[i_sample], bbox={'facecolor': 'white', 'pad': 2})
    if is_prediction :
        y_pred = d2l.get_fashion_mnist_labels(d2l.argmax(net(X_test[:nb_sample]), axis=1))
        for i_sample in range(nb_sample) :
            axarr[nb_layer-1,i_sample].text(0, 7, list_label[i_sample], bbox={'facecolor': 'white', 'pad': 2})
    

    plt.show()



#-------------------------------------------------------------------------------
#Select number of sampled input-data to be processed as features-map
#-------------------------------------------------------------------------------
nb_sample=4

#-------------------------------------------------------------------------------
#Select deepness of conv. net, leading to netconv, layers of convolutional process
#-------------------------------------------------------------------------------
list_layer = [0,1,2,3]
#--------------------------------------------------------------------------------
#Then call to the display function
#--------------------------------------------------------------------------------
display2D_netOutput(net, list_layer, X_test, y_test, nb_sample , is_prediction=True)
