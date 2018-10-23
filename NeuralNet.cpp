//NeuralNet.cpp
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

class Neuron;

struct Connection {
    double weight;
    double deltaweight;

};


typedef vector<Neuron> Layer;

class Neuron{
    public:
        Neuron(unsigned numOutputs, unsigned index);
        void feedForward(Layer &prevLayer);
        void setOutputVal(double val){ m_outputVal = val;}
        double getOutputVal(void){return m_outputVal;}
        void calcOutputGradients(double targetVal);
        void calcHiddenGradient(Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);
    private:
        static double randomWeight(void) {return rand()/double(RAND_MAX);}
        double m_outputVal;
        unsigned m_index;
        double m_gradient;
        vector<Connection> m_outputWeights;
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        static double alpha;
        static double eta;
};

double Neuron::alpha = .5;
double Neuron::eta = .15;

Neuron::Neuron(unsigned numOutputs, unsigned index){
    for (unsigned c = 0; c < numOutputs; c++){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
        m_outputWeights.back().deltaweight = 0.0;
        //m_outputWeights.push_back(randomWeight());
        //cout << randomWeight() << endl;
        //m_outputDeltaWeights.push_back(0.0);

    }
    m_index = index;
    //cout << "Neuron: "<< m_index  << endl;
}
void Neuron::feedForward(Layer &prevLayer){
    double sum = 0.0;
    //sum all of the prev layer's neurons
    for (int i = 0; i < prevLayer.size(); i++){
        cout << prevLayer[i].getOutputVal() << ", " << prevLayer[i].m_outputWeights[m_index].weight << endl;
        sum += prevLayer[i].getOutputVal() * 
            prevLayer[i].m_outputWeights[m_index].weight;
    }

    //transfer function
    m_outputVal = Neuron::transferFunction(sum);

    cout << "Forward prop Neuron:" << m_index << ", sum: " << m_outputVal  << endl;
}
double Neuron::transferFunction(double x){
    //tanh [-1,1]
    return tanh(x);
}
double Neuron::transferFunctionDerivative(double x){
    return 1.0 - x*x;
}
void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}
void Neuron::calcHiddenGradient(Layer &nextLayer){
    
    double sum = 0.0;
    for (unsigned i = 0; i < nextLayer.size()-1; i++){
        sum += m_outputWeights[i].weight * nextLayer[i].m_gradient;
    }
    m_gradient = sum * Neuron::transferFunctionDerivative(m_outputVal);
}
void Neuron::updateInputWeights(Layer &prevLayer){
    //weights are updated in connect cointainer
    for(unsigned i = 0; i < prevLayer.size(); i++){
        Neuron &neuron = prevLayer[i];
        double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaweight;

        double newDeltaWeight = 
            //Individual input, magnified by gradient and train rate
            eta 
            * neuron.getOutputVal()
            * m_gradient
            + (alpha
            * oldDeltaWeight);

        cout << "   START" << endl;
        cout << "   " << i << ": Delta = " << neuron.m_outputWeights[m_index].deltaweight << endl;
        cout << "   " << i << ": Weight = " << neuron.m_outputWeights[m_index].weight << endl;
        cout << "   END" << endl;
        
        neuron.m_outputWeights[m_index].deltaweight = newDeltaWeight;
        neuron.m_outputWeights[m_index].weight += newDeltaWeight;
        
    }
}
class Net{
    public:
        Net(vector<unsigned> &netSize);
        void feedForward(vector<double> &inputVals);
        void backProp(vector<double> &targetVals);
        void getResults(void);
    private:
        vector<Layer> m_layers;
        double m_error;
};
Net::Net(vector<unsigned> &netSize){
    unsigned numLayers = netSize.size();
    for (unsigned i = 0; i < numLayers; i++){
        m_layers.push_back(Layer());
        unsigned numOutputs = i == numLayers-1 ? 0:netSize[i+1];
        // <= is for bias
        for (unsigned j = 0; j <= netSize[i]; j++){

            m_layers[i].push_back(Neuron(numOutputs,j));
            //cout << "created new neuron: " << j << " on layer: " << i <<  endl;
            if (j == netSize[i]){
                m_layers[i][j].setOutputVal(1.0);
            }
        }
        
    }
    
}
void Net::feedForward(vector<double> &inputVals){
    //assign first variables
    //cout << inputVals.size() << m_layers[0].size() << endl;
    if (inputVals.size() != m_layers[0].size()-1){
        cout << "input size does not equals size of first layer" << endl;
    }
    for (unsigned i = 0; i < inputVals.size(); i++){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    //forward prop
    for (unsigned i = 1; i < m_layers.size(); i++){
        Layer &prevLayer =  m_layers[i-1];
        for (unsigned j = 0; j < m_layers[i].size() -1 ; j++){
            cout << "Forward prop layer:" << i  << endl;
            m_layers[i][j].feedForward(prevLayer);
        }
    }

}
void Net::backProp(vector<double> &targetVals){
    m_error = 0.0;

    //Calculate overall net error
    for (unsigned i = 0; i < m_layers.back().size()-1; i++){
        double delta = targetVals[i] - m_layers.back()[i].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= targetVals.size()-1;
    m_error = sqrt(m_error);

    //implement a recent avg measurement
    //no from me

    //calculate output layer gradient
    for (unsigned i = 0; i < m_layers.back().size() -1; i++){
        m_layers.back()[i].calcOutputGradients(targetVals[i]);
    }
    //calc gradients on hidden layer
    for (unsigned i = m_layers.size() -2; i > 0; i--){
        Layer &hiddenLayer = m_layers[i];
        Layer &nextLayer = m_layers[i+1];
        for (unsigned j = 0; j < hiddenLayer.size(); j++){
            hiddenLayer[j].calcHiddenGradient(nextLayer);
        }
    }
    //for all layers from output to first hidden layer
    //update
    
    for (unsigned i = m_layers.size()-1; i > 0; i--){
        Layer &layer = m_layers[i];
        Layer &prevLayer = m_layers[i-1];
        cout << "update layer: " << i << endl;
        for (unsigned j = 0; j < layer.size()-1; j++){
            cout << " update neuron :" << j << endl;
            layer[j].updateInputWeights(prevLayer);
        }
    }
    
}
void Net::getResults(){
    Layer &last = m_layers.back();
    
     cout << "------------------------" << endl;
     cout << "------RESULTS-----------" << endl;
     cout << "------------------------" << endl;
    for (unsigned i = 0; i < last.size()-1; i++){
        cout << last[i].getOutputVal();
    }
    cout << endl;
    cout << "------------------------" << endl;
    cout << "------------------------" << endl;

}


int main(){
    vector<unsigned> netSize;
    netSize.push_back(2);
    netSize.push_back(4);
    netSize.push_back(1);
    Net myNet(netSize);

    vector<double> inputVals;
    inputVals.push_back(1);
    inputVals.push_back(0);

    vector<double> targetVals;
    targetVals.push_back(1);


    for (int i = 0; i < 100; i++){
        myNet.feedForward(inputVals);
        myNet.backProp(targetVals);
    }
    myNet.feedForward(inputVals);
    myNet.getResults();

    cout << "HI" << endl;


    return 1;
}