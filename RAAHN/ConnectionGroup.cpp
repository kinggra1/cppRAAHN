#include "stdafx.h"

#include "NeuronGroup.h"
#include "ConnectionGroup.h"
#include "TrainingMethod.h"
#include "NeuralNetwork.h"

#include <string>
#include <math.h>

using std::string;



	Connection::Connection()
	{
		input = 0;
		output = 0;
		weight = 0.0;
	}

	Connection::Connection(unsigned i, unsigned o, double w)
	{
		input = i;
		output = o;
		weight = w;
	}

	const double DEFAULT_LEARNING_RATE = 0.1;
	unsigned sampleUsageCount;


	ConnectionGroup::ConnectionGroup(NeuralNetwork *network, NeuronGroup *inGroup, NeuronGroup *outGroup, bool useBias)
	{
		learningRate = DEFAULT_LEARNING_RATE;

		connections = vector<Connection*>();

		modSigIndex = ModulationSignal::INVALID_INDEX;

		ann = network;

		inputGroup = inGroup;
		outputGroup = outGroup;

		//Default to autoencoder training.
		trainingMethod = TrainingMethod::AutoencoderTrain; // function pointer assignment looks like it worked? Nice.



		usingBias = useBias;
		biasWeights = vector<double>();
		/*
		if (useBias)
			biasWeights = vector<double>();
		else
			biasWeights = null;
		*/
	}

	void ConnectionGroup::AddConnection(unsigned inputIndex, unsigned outputIndex, double weight)
	{
		connections.push_back(new Connection(inputIndex, outputIndex, weight));
	}

	void ConnectionGroup::AddBiasWeights(unsigned outputCount)
	{
		if (!usingBias)
			return;

		double neuronInOutCount = (double)(inputGroup->neurons.size() + outputGroup->neurons.size() + 1);

		for (unsigned i = 0; i < outputCount; i++)
		{
			double range = sqrt(ann->WEIGHT_RANGE_SCALE / neuronInOutCount);
			//Keep in the range of [-range, range]

			biasWeights.push_back((ann->NextDouble() * range * ann->DOUBLE_WEIGHT_RANGE) - range);
			//biasWeights.push_back((rand.NextDouble() * range * DOUBLE_WEIGHT_RANGE) - range);
		}
	}

	void ConnectionGroup::PropagateSignal()
	{
		//Make sure the input group is computed.
		if (!inputGroup->computed)
			inputGroup->ComputeSignal();

		//Use the output neurons to temporarily hold the sum of
		//the weighted inputs. Then apply the activation function.
		for (unsigned i = 0; i < connections.size(); i++)
			outputGroup->neurons[(int)connections[i]->output] += inputGroup->neurons[(int)connections[i]->input]
			* connections[i]->weight;

		if (usingBias)
		{
			for (unsigned i = 0; i < biasWeights.size(); i++)
				outputGroup->neurons[i] += biasWeights[i];
		}
	}

	void ConnectionGroup::UpdateAverages()
	{
		if (trainingMethod == TrainingMethod::SparseAutoencoderTrain)
			outputGroup->UpdateAverages();
	}

	void ConnectionGroup::SetModulationIndex(int index)
	{
		modSigIndex = index;
	}

	void ConnectionGroup::SetLearningRate(double lRate)
	{
		learningRate = lRate;
	}

	/*
	void ConnectionGroup::SetTrainingMethod(TrainFunctionType method)
	{
		trainingMethod = method;
	}
	*/

	void ConnectionGroup::ResetWeights()
	{
		double neuronInOutCount = (double)(inputGroup->neurons.size() + outputGroup->neurons.size());

		if (usingBias)
			neuronInOutCount++;

		for (unsigned i = 0; i < connections.size(); i++)
		{
			double range = sqrt(ann->WEIGHT_RANGE_SCALE / neuronInOutCount);
			//Keep in the range of [-range, range]
			connections[i]->weight = (ann->NextDouble() * range * ann->DOUBLE_WEIGHT_RANGE) - range;
		}

		if (usingBias)
		{
			for (unsigned i = 0; i < biasWeights.size(); i++)
			{
				double range = sqrt(ann->WEIGHT_RANGE_SCALE / neuronInOutCount);
				//Keep in the range of [-range, range]
				biasWeights[i] = (ann->NextDouble() * range * ann->DOUBLE_WEIGHT_RANGE) - range;
			}
		}
	}

	int ConnectionGroup::GetInputGroupIndex()
	{
		return inputGroup->index;
	}

	int ConnectionGroup::GetOutputGroupIndex()
	{
		return outputGroup->index;
	}

	double ConnectionGroup::Train()
	{
		return trainingMethod(modSigIndex, learningRate, ann, inputGroup, outputGroup, connections, biasWeights);
	}

	double ConnectionGroup::GetReconstructionError()
	{
		//If a autoencoder training was not used there is no reconstruction error.
		if (trainingMethod == TrainingMethod::HebbianTrain)
			return 0.0;

		int reconstructionCount = inputGroup->neurons.size();

		//Plus one for the bias neuron.
		if (usingBias)
			reconstructionCount++;

		double* reconstructions = new double[reconstructionCount];
		double* errors = new double[reconstructionCount];

		//If there is a bias neuron, it's reconstruction and error will be the last value in each.
		int biasRecIndex = reconstructionCount - 1; //reconstructions.Length - 1;

		//First sum the weighted values into the reconstructions to store them.
		for (unsigned i = 0; i < connections.size(); i++)
			reconstructions[(int)connections[i]->input] += outputGroup->neurons[(int)connections[i]->output]
			* connections[i]->weight;

		if (usingBias)
		{
			for (unsigned i = 0; i < biasWeights.size(); i++)
				reconstructions[biasRecIndex] += biasWeights[i];
		}

		//Apply the activation function after the weighted values are summed.
		//Also calculate the error of the reconstruction.
		//Do the bias weights separately.
		for (unsigned i = 0; i < inputGroup->neurons.size(); i++)
		{
			reconstructions[i] = ann->activation(reconstructions[i]);
			errors[i] = inputGroup->neurons[i] - reconstructions[i];
		}

		if (usingBias)
		{
			reconstructions[biasRecIndex] = ann->activation(reconstructions[biasRecIndex]);
			errors[biasRecIndex] = TrainingMethod::BIAS_INPUT - reconstructions[biasRecIndex];
		}

		double sumOfSquaredError = 0.0;

		for (int i = 0; i < reconstructionCount; i++)
			sumOfSquaredError += pow(errors[i], TrainingMethod::ERROR_POWER);

		return (sumOfSquaredError / TrainingMethod::ERROR_POWER);
	}

	/*
	ConnectionGroup::TrainFunctionType ConnectionGroup::GetTrainingMethod()
	{
		return nullptr;
	}
	*/

	bool ConnectionGroup::UsesBiasWeights()
	{
		if (usingBias)
			return true;

		return false;
	}

	//Returns true if the connection was able to be removed, false otherwise.
	bool ConnectionGroup::RemoveConnection(unsigned index)
	{
		if (index < connections.size())
		{
			connections.erase(connections.begin() + (int)index);
			return true;
		}
		else
			return false;
	}

	bool ConnectionGroup::IsConnectedTo(NeuronGroup::Identifier toGroup)
	{
		if (outputGroup->type == toGroup.type && outputGroup->index == toGroup.index)
			return true;

		return false;
	}

	NeuronGroup::Type ConnectionGroup::GetInputGroupType()
	{
		return inputGroup->type;
	}

	NeuronGroup::Type ConnectionGroup::GetOutputGroupType()
	{
		return outputGroup->type;
	}

	//Returns a copy of the weights.
	vector<double> ConnectionGroup::GetWeights()
	{
		vector<double> weights = vector<double>();

		for (unsigned i = 0; i < connections.size(); i++)
			weights.push_back(connections[i]->weight);

		if (usingBias)
		{
			for (unsigned i = 0; i < biasWeights.size(); i++)
				weights.push_back(biasWeights[i]);
		}

		return weights;
	}

