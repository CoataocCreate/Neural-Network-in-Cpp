#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize) 
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
        srand(time(0));

        // Initialize weights with random values
        inputHiddenWeights.resize(inputSize * hiddenSize);
        hiddenOutputWeights.resize(hiddenSize * outputSize);
        initializeWeights(inputHiddenWeights);
        initializeWeights(hiddenOutputWeights);
    }

    // Train the network
    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& expectedOutputs, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                vector<double> hiddenLayer(hiddenSize);
                vector<double> outputLayer(outputSize);

                // Forward propagation
                forward(inputs[i], hiddenLayer, outputLayer);

                // Compute errors
                vector<double> outputError(outputSize);
                vector<double> hiddenError(hiddenSize);

                for (int j = 0; j < outputSize; ++j) {
                    outputError[j] = expectedOutputs[i][j] - outputLayer[j];
                }

                for (int j = 0; j < hiddenSize; ++j) {
                    hiddenError[j] = 0;
                    for (int k = 0; k < outputSize; ++k) {
                        hiddenError[j] += outputError[k] * hiddenOutputWeights[k * hiddenSize + j];
                    }
                    hiddenError[j] *= sigmoidDerivative(hiddenLayer[j]);
                }

                // Update weights
                updateWeights(inputs[i], hiddenLayer, outputError, hiddenError, learningRate);
            }
        }
    }

    // Test the network
    void test(const vector<vector<double>>& inputs, const vector<vector<double>>& expectedOutputs) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            vector<double> hiddenLayer(hiddenSize);
            vector<double> outputLayer(outputSize);

            forward(inputs[i], hiddenLayer, outputLayer);

            cout << "Input: ";
            for (double x : inputs[i]) cout << x << " ";
            cout << "Output: ";
            for (double y : outputLayer) cout << y << " ";
            cout << endl;
        }
    }

private:
    int inputSize, hiddenSize, outputSize;
    vector<double> inputHiddenWeights;
    vector<double> hiddenOutputWeights;

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    void initializeWeights(vector<double>& weights) {
        for (double& weight : weights) {
            weight = static_cast<double>(rand()) / RAND_MAX * 0.2 - 0.1;
        }
    }

    void forward(const vector<double>& input, vector<double>& hiddenLayer, vector<double>& outputLayer) {
        for (int j = 0; j < hiddenSize; ++j) {
            hiddenLayer[j] = 0;
            for (int k = 0; k < inputSize; ++k) {
                hiddenLayer[j] += input[k] * inputHiddenWeights[j * inputSize + k];
            }
            hiddenLayer[j] = sigmoid(hiddenLayer[j]);
        }

        for (int j = 0; j < outputSize; ++j) {
            outputLayer[j] = 0;
            for (int k = 0; k < hiddenSize; ++k) {
                outputLayer[j] += hiddenLayer[k] * hiddenOutputWeights[j * hiddenSize + k];
            }
            outputLayer[j] = sigmoid(outputLayer[j]);
        }
    }

    void updateWeights(const vector<double>& input, const vector<double>& hiddenLayer, const vector<double>& outputError, const vector<double>& hiddenError, double learningRate) {
        // Update hidden-output weights
        for (int j = 0; j < outputSize; ++j) {
            for (int k = 0; k < hiddenSize; ++k) {
                hiddenOutputWeights[j * hiddenSize + k] += learningRate * outputError[j] * hiddenLayer[k];
            }
        }

        // Update input-hidden weights
        for (int j = 0; j < hiddenSize; ++j) {
            for (int k = 0; k < inputSize; ++k) {
                inputHiddenWeights[j * inputSize + k] += learningRate * hiddenError[j] * input[k];
            }
        }
    }
};

int main() {
    // XOR problem
    vector<vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    vector<vector<double>> expectedOutputs = {
        {0},
        {1},
        {1},
        {0}
    };

    NeuralNetwork nn(2, 4, 1); // 2 input neurons, 4 hidden neurons, 1 output neuron
    nn.train(inputs, expectedOutputs, 10000, 0.1);
    nn.test(inputs, expectedOutputs);

    return 0;
}
