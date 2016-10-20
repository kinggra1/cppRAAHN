#include "stdafx.h"
#include "TrainingMethod.h"
#include "NeuralNetwork.h"
#include "ConnectionGroup.h"
#include "NeuronGroup.h"

#include <vector>

using std::vector;


// typesafe sign helper
template <typename T>
int sign(T val) {
	return (T(0) < val) - (val < T(0));
}


const double TrainingMethod::BIAS_INPUT = 1.0;
const double TrainingMethod::ERROR_POWER = 2.0;
const double TrainingMethod::NO_ERROR = 0.0;


const double TrainingMethod::HEBBIAN_SCALE = 2.0;

const double TrainingMethod::HEBBIAN_OFFSET = HEBBIAN_SCALE / 2.0;
const double TrainingMethod::SPARSITY_PARAMETER = 1.0;
const double TrainingMethod::SPARSITY_WEIGHT = 0.1;


//Autoencoder training with tied weights.
double TrainingMethod::AutoencoderTrain(int modIndex, double learningRate, NeuralNetwork *ann, NeuronGroup *inGroup,
	NeuronGroup *outGroup, vector<Connection*> connections, vector<double> biasWeights)
{
	double weightCap = ann->GetWeightCap();

	int reconstructionCount = inGroup->neurons.size();
	int featureCount = outGroup->neurons.size();

	//Plus one for the bias neuron.
	if (biasWeights.size() != 0)
		reconstructionCount++;

	double *reconstructions = new double[reconstructionCount];
	double *errors = new double[reconstructionCount];
	double *deltas = new double[reconstructionCount];
	//No need to save backprop errors. Regular error is needed for reporting.
	double *backPropDeltas = new double[featureCount];

	//If there is a bias neuron, it's reconstruction and error will be the last value in each.
	int biasRecIndex = reconstructionCount - 1;

	//First sum the weighted values into the reconstructions to store them.
	for (unsigned i = 0; i < connections.size(); i++)
		reconstructions[(int)connections[i]->input] += outGroup->neurons[(int)connections[i]->output]
		* connections[i]->weight;

	if (biasWeights.size() != 0)
	{
		for (unsigned i = 0; i < biasWeights.size(); i++)
			reconstructions[biasRecIndex] += biasWeights[i];
	}

	//Apply the activation function after the weighted values are summed.
	//Also calculate the error of the reconstruction.
	//Do the bias weights separately.
	for (unsigned i = 0; i < inGroup->neurons.size(); i++)
	{
		reconstructions[i] = ann->activation(reconstructions[i]);
		errors[i] = inGroup->neurons[i] - reconstructions[i];
		deltas[i] = errors[i] * ann->activationDerivative(reconstructions[i]);
	}

	if (biasWeights.size() != 0)
	{
		reconstructions[biasRecIndex] = ann->activation(reconstructions[biasRecIndex]);
		errors[biasRecIndex] = BIAS_INPUT - reconstructions[biasRecIndex];
		deltas[biasRecIndex] = errors[biasRecIndex] * ann->activationDerivative(reconstructions[biasRecIndex]);
	}

	//Now that all the errors are calculated, the backpropagated error can be calculated.
	//First compute the dot products of each outgoing weight vector and its respective delta.
	//Go through each connection and add its contribution to its respective dot product.
	for (unsigned i = 0; i < connections.size(); i++)
	{
		int backPropErrorIndex = (int)connections[i]->output;
		int errorIndex = (int)connections[i]->input;

		backPropDeltas[backPropErrorIndex] += deltas[errorIndex] * connections[i]->weight;
	}

	if (biasWeights.size() != 0)
	{
		for (unsigned i = 0; i < biasWeights.size(); i++)
			backPropDeltas[i] += deltas[biasRecIndex] * biasWeights[i];
	}

	//Now multiply the delta by the derivative of the activation function at the feature neurons.
	for (int i = 0; i < featureCount; i++)
		backPropDeltas[i] *= ann->activationDerivative(outGroup->neurons[i]);

	//Update the weights with stochastic gradient descent.
	for (unsigned i = 0; i < connections.size(); i++)
	{
		int inputIndex = (int)connections[i]->input;
		int outputIndex = (int)connections[i]->output;

		double errorWeightDelta = learningRate * deltas[inputIndex] * outGroup->neurons[outputIndex];

		double backPropWeightDelta = learningRate * backPropDeltas[outputIndex] * inGroup->neurons[inputIndex];

		double weightDelta = errorWeightDelta + backPropWeightDelta;

		if (abs(connections[i]->weight + weightDelta) < weightCap)
			connections[i]->weight += weightDelta;
	}

	if (biasWeights.size() != 0)
	{
		for (unsigned i = 0; i < biasWeights.size(); i++)
		{
			double errorWeightDelta = learningRate * deltas[biasRecIndex] * outGroup->neurons[i];

			double backPropWeightDelta = learningRate * backPropDeltas[i];

			double weightDelta = errorWeightDelta + backPropWeightDelta;

			if (abs(biasWeights[i] + weightDelta) < weightCap)
				biasWeights[i] += weightDelta;
		}
	}

	double sumOfSquaredError = 0.0;

	for (int i = 0; i < reconstructionCount; i++)
		sumOfSquaredError += pow(errors[i], ERROR_POWER);

	return (sumOfSquaredError / ERROR_POWER);
}

//Sparse autoencoder training with tied weights.
double TrainingMethod::SparseAutoencoderTrain(int modIndex, double learningRate, NeuralNetwork *ann, NeuronGroup *inGroup,
	NeuronGroup *outGroup, vector<Connection*> connections, vector<double> biasWeights)
{
	double weightCap = ann->GetWeightCap();

	int reconstructionCount = inGroup->neurons.size();
	int featureCount = outGroup->neurons.size();

	//Plus one for the bias neuron.
	if (biasWeights.size() != 0)
		reconstructionCount++;

	double *reconstructions = new double[reconstructionCount];
	double *errors = new double[reconstructionCount];
	double *deltas = new double[reconstructionCount];
	//No need to save backprop errors. Regular error is needed for reporting.
	double *backPropDeltas = new double[featureCount];

	//If there is a bias neuron, it's reconstruction and error will be the last value in each.
	int biasRecIndex = reconstructionCount - 1;

	//First sum the weighted values into the reconstructions to store them.
	for (unsigned i = 0; i < connections.size(); i++)
		reconstructions[(int)connections[i]->input] += outGroup->neurons[(int)connections[i]->output] * connections[i]->weight;

	if (biasWeights.size() != 0)
	{
		for (unsigned i = 0; i < biasWeights.size(); i++)
			reconstructions[biasRecIndex] += biasWeights[i];
	}

	//Apply the activation function after the weighted values are summed.
	//Also calculate the error of the reconstruction.
	//Do the bias weights separately.
	for (unsigned i = 0; i < inGroup->neurons.size(); i++)
	{
		reconstructions[i] = ann->activation(reconstructions[i]);
		errors[i] = inGroup->neurons[i] - reconstructions[i];
		deltas[i] = errors[i] * ann->activationDerivative(reconstructions[i]);
	}

	if (biasWeights.size() != 0)
	{
		reconstructions[biasRecIndex] = ann->activation(reconstructions[biasRecIndex]);
		errors[biasRecIndex] = BIAS_INPUT - reconstructions[biasRecIndex];
		deltas[biasRecIndex] = errors[biasRecIndex] * ann->activationDerivative(reconstructions[biasRecIndex]);
	}

	//Now that all the errors are calculated, the backpropagated error can be calculated.
	//First compute the dot products of each outgoing weight vector and its respective delta.
	//Go through each connection and add its contribution to its respective dot product.
	for (unsigned i = 0; i < connections.size(); i++)
	{
		int backPropErrorIndex = (int)connections[i]->output;
		int errorIndex = (int)connections[i]->input;

		backPropDeltas[backPropErrorIndex] += deltas[errorIndex] * connections[i]->weight;
	}

	if (biasWeights.size() != 0)
	{
		for (unsigned i = 0; i < biasWeights.size(); i++)
			backPropDeltas[i] += deltas[biasRecIndex] * biasWeights[i];
	}

	//Add the spasity term. Also multiply the delta by the 
	//derivative of the activation function at the feature neurons.
	for (int i = 0; i < featureCount; i++)
	{
		//Sparsity parameter over average reconstruction.
		double paramOverReconst = SPARSITY_PARAMETER / outGroup->averages[i];
		double oneMinusParamOverOneMinusReconst = (1.0 - SPARSITY_PARAMETER) / (1.0 - outGroup->averages[i]);
		double sparsityTerm = SPARSITY_WEIGHT * (oneMinusParamOverOneMinusReconst - paramOverReconst);

		backPropDeltas[i] *= ann->activationDerivative(outGroup->neurons[i]);
		backPropDeltas[i] -= sparsityTerm;
	}

	//Update the weights with stochastic gradient descent.
	for (unsigned i = 0; i < connections.size(); i++)
	{
		int inputIndex = (int)connections[i]->input;
		int outputIndex = (int)connections[i]->output;

		double errorWeightDelta = learningRate * deltas[inputIndex] * outGroup->neurons[outputIndex];

		double backPropWeightDelta = learningRate * backPropDeltas[outputIndex] * inGroup->neurons[inputIndex];

		double weightDelta = errorWeightDelta + backPropWeightDelta;

		if (abs(connections[i]->weight + weightDelta) < weightCap)
			connections[i]->weight += weightDelta;
		else
			connections[i]->weight = abs(connections[i]->weight) * weightCap;
	}

	if (biasWeights.size() != 0)
	{
		for (unsigned i = 0; i < biasWeights.size(); i++)
		{
			double errorWeightDelta = learningRate * deltas[biasRecIndex] * outGroup->neurons[i];

			double backPropWeightDelta = learningRate * backPropDeltas[i];

			double weightDelta = errorWeightDelta + backPropWeightDelta;

			if (abs(biasWeights[i] + weightDelta) < weightCap)
				biasWeights[i] += weightDelta;
			else
				biasWeights[i] = abs(biasWeights[i]) * weightCap;
		}
	}

	double sumOfSquaredError = 0.0;

	for (int i = 0; i < reconstructionCount; i++)
		sumOfSquaredError += pow(errors[i], ERROR_POWER);

	return (sumOfSquaredError / ERROR_POWER);
}

//Hebbian learning.
double TrainingMethod::HebbianTrain(int modIndex, double learningRate, NeuralNetwork *ann, NeuronGroup *inGroup,
	NeuronGroup *outGroup, vector<Connection*> connections, vector<double> biasWeights)
{
	double modSig = ModulationSignal::GetSignal(modIndex);

	//If the modulation signal is zero there is no weight change.
	if (modSig == ModulationSignal::NO_MODULATION)
		return NO_ERROR;

	double weightCap = ann->GetWeightCap();

	for (unsigned i = 0; i < connections.size(); i++)
	{
		//Normalize to [-1, 1] to allow for positive and negative deltas without modulation.
		double normalizedInput = inGroup->neurons[(int)connections[i]->input];
		double normalizedOutput = outGroup->neurons[(int)connections[i]->output] * HEBBIAN_SCALE - HEBBIAN_OFFSET;
		double noise = (ann->NextDouble() * ann->getWeightNoiseRange()) - ann->getWeightNoiseMagnitude();

		double weightDelta = (modSig * learningRate * normalizedInput * normalizedOutput) + noise;

		if (abs(connections[i]->weight + weightDelta) < weightCap)
			connections[i]->weight += weightDelta;
		else
			connections[i]->weight = sign(connections[i]->weight) * weightCap;
	}

	if (biasWeights.size() != 0)
	{
		//The length of biasWeights should always be equal to the length of outGroup->neurons.
		for (unsigned i = 0; i < biasWeights.size(); i++)
		{
			double normalizedOutput = outGroup->neurons[i] * HEBBIAN_SCALE - HEBBIAN_OFFSET;
			double noise = (ann->NextDouble() * ann->getWeightNoiseRange()) - ann->getWeightNoiseMagnitude();

			double weightDelta = (modSig * learningRate * normalizedOutput) + noise;

			if (abs(biasWeights[i] + weightDelta) < weightCap)
				biasWeights[i] += weightDelta;
			else
				biasWeights[i] = sign(biasWeights[i]) * weightCap;
		}
	}

	return NO_ERROR;
}
