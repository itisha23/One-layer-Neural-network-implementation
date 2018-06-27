from numpy import array,exp,random,dot

class NeuralNetwork():
    def __init__ (self):

        random.seed(1)
        self.synaptic_weights=2* random.random((3,1))-1
    
    def sigmoid(self,x):
        return 1/(1+exp(-x))

    def predict(self,training_set_inputs):
        return self.sigmoid(dot(training_set_inputs,self.synaptic_weights)) 
    
    def sigmoid_derivative(self,x):
        return x*(1-x)

       
    def train(self,training_set_inputs,training_set_outputs,number_of_iterations):

        for iteration in xrange(number_of_iterations):

            output=self.predict(training_set_inputs)
            error=training_set_outputs-output
            adjustments=dot(training_set_inputs.T,error*self.sigmoid_derivative(output))
            self.synaptic_weights+=adjustments



        



if __name__ =='__main__':
    neural_network=NeuralNetwork()

    print " Random Starting Synaptic Weights"
    print neural_network.synaptic_weights

    training_set_inputs=array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs=array([[0,1,1,0]]).T

    
    #train the neural 1000 times making small changes each time
    neural_network.train(training_set_inputs,training_set_outputs,100000)
    
    print "New synaptic weights after training"
    print neural_network.synaptic_weights

    print " Considering new input"
    print neural_network.predict(array([1,0,0]))


