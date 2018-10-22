//NeuralNet.cpp
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

class Neuron;


typedef vector<Neuron> Layer;

class Neuron{
    public:
        Neuron(unsigned numOutputs, unsigned index);
        void feedForward(Layer &prevLayer);
        void setOutputVal(double val){ m_outputVal = val;}
        double getOutputVal(void){return m_outputVal;}
    private:
        static double randomWeight(void) {return rand()/double(RAND_MAX);}
        double m_outputVal;
        unsigned m_index;
        vector<double> m_outputWeights;
        vector<double> m_outputDeltaWeights;
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
};
Neuron::Neuron(unsigned numOutputs, unsigned index){
    for (unsigned c = 0; c < numOutputs; c++){
        m_outputWeights.push_back(randomWeight());
        //cout << randomWeight() << endl;
        m_outputDeltaWeights.push_back(0.0);

    }
    m_index = index;
    cout << "Neuron: "<< m_index  << endl;
}

void Neuron::feedForward(Layer &prevLayer){
    double sum = 0.0;
    //sum all of the prev layer's neurons
    for (int i = 0; i < prevLayer.size(); i++){
        cout << prevLayer[i].getOutputVal() << ", " << prevLayer[i].m_outputWeights[m_index] << endl;
        sum += prevLayer[i].getOutputVal() * 
            prevLayer[i].m_outputWeights[m_index];
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


class Net{
    public:
        Net(vector<unsigned> &netSize);
        void feedForward(vector<double> &inputVals);
        void backProp(vector<double> &targetVals){};
        void getResults(vector<double> &resultVals){};
    private:
        vector<Layer> m_layers;
};

Net::Net(vector<unsigned> &netSize){
    unsigned numLayers = netSize.size();
    for (unsigned i = 0; i < numLayers; i++){
        m_layers.push_back(Layer());
        unsigned numOutputs = i == numLayers-1 ? 0:netSize[i+1];
        // <= is for bias
        for (unsigned j = 0; j <= netSize[i]; j++){

            m_layers[i].push_back(Neuron(numOutputs,j));
            cout << "created new neuron: " << j << " on layer: " << i <<  endl;
        }
    }
}

void Net::feedForward(vector<double> &inputVals){
    //assign first variables
    cout << inputVals.size() << m_layers[0].size() << endl;
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


int main(){
    vector<unsigned> netSize;
    netSize.push_back(3);
    netSize.push_back(3);
    netSize.push_back(3);
    Net myNet(netSize);

    vector<double> inputVals;
    inputVals.push_back(1);
    inputVals.push_back(2);
    inputVals.push_back(3);
    myNet.feedForward(inputVals);




    return 1;
}