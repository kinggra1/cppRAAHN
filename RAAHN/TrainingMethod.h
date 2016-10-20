#pragma once
#include "ConnectionGroup.h"
#include "NeuronGroup.h"
#include <vector>

using std::vector;

class NeuralNetwork;


class TrainingMethod
{
public:
	static const double BIAS_INPUT;
	static const double ERROR_POWER;
	static const double NO_ERROR;

	//Autoencoder training with tied weights.
	static double AutoencoderTrain(int modIndex, double learningRate, NeuralNetwork *ann, NeuronGroup *inGroup,
		NeuronGroup *outGroup, vector<Connection*> connections, vector<double> biasWeights) {return 0.0;};

	//Sparse autoencoder training with tied weights.
	static double SparseAutoencoderTrain(int modIndex, double learningRate, NeuralNetwork *ann, NeuronGroup *inGroup,
		NeuronGroup *outGroup, vector<Connection*> connections, vector<double> biasWeights) {return 0.0;};

	//Hebbian learning.
	static double HebbianTrain(int modIndex, double learningRate, NeuralNetwork *ann, NeuronGroup *inGroup,
		NeuronGroup *outGroup, vector<Connection*> connections, vector<double> biasWeights) {return 0.0;};

private:
	const double HEBBIAN_SCALE = 2.0;
	//Since sigmoid returns values between 0.1 half
	//the scale will be the distance in both directions.
	const double HEBBIAN_OFFSET = HEBBIAN_SCALE / 2.0;
	const double SPARSITY_PARAMETER = 1.0;
	const double SPARSITY_WEIGHT = 0.1;
};
