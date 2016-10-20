#include "stdafx.h"
#include "NeuralNetwork.h"
#include "NeuronGroup.h"
#include "ConnectionGroup.h"
#include "TrainingMethod.h"

#include <vector>
#include <algorithm>

using std::vector;
using std::find;


NeuronGroup::NeuronGroup(NeuralNetwork *network, Type t)
{
	Construct(network, t);
}

NeuronGroup::NeuronGroup(unsigned count, NeuralNetwork *network, Type t)
{
	Construct(network, t);

	AddNeurons(count);
}

void NeuronGroup::Construct(NeuralNetwork *network, Type t)
{
	ann = network;

	averages = vector<double>();

	computed = true;

	useNoise = false;

	neurons = vector<double>();

	incomingGroups = vector<ConnectionGroup*>();
	outgoingGroups = vector<ConnectionGroup*>();
	outTrainRecent = vector<ConnectionGroup*>();
	outTrainSeveral = vector<ConnectionGroup*>();
}

void NeuronGroup::AddNeurons(unsigned count)
{
	for (unsigned i = 0; i < count; i++)
		neurons.push_back(DEFAULT_NEURON_VALUE);
}

void NeuronGroup::AddIncomingGroup(ConnectionGroup *incomingGroup)
{
	incomingGroups.push_back(incomingGroup);

	ConnectionGroup::TrainFunctionType method = incomingGroup->GetTrainingMethod();

	//Noise must be used for Hebbian trained connection groups.
	//Noise is added after the activation function, so it has to
	//be addded if there is at least one Hebbain trained connection group.
	if (method == TrainingMethod::HebbianTrain)
		useNoise = true;
	else if (method == TrainingMethod::SparseAutoencoderTrain)
	{
		//if (averages == null)
		if (averages.size() == 0)
		{
			int featureCount = neurons.size();

			averages = vector<double>(featureCount);

			for (int i = 0; i < featureCount; i++)
				averages.push_back(DEFAULT_NEURON_VALUE);
		}
	}
}

//mostRecent refers to whether the group should train off of only the most recent experience.
void NeuronGroup::AddOutgoingGroup(ConnectionGroup *outgoingGroup, bool mostRecent)
{
	outgoingGroups.push_back(outgoingGroup);

	//Should the group use the most recent example or several randomly selected ones.
	if (mostRecent)
		outTrainRecent.push_back(outgoingGroup);
	else
		outTrainSeveral.push_back(outgoingGroup);
}

void NeuronGroup::UpdateAverages()
{
	double decay = 0.0;
	double exponent = 1.0 / (double)ann->getHistoryBufferSize();

	decay = pow(DECAY_BASE, exponent);

	for (unsigned i = 0; i < averages.size(); i++)
		averages[i] = (decay * averages[i]) + ((1.0 - decay) * neurons[i]);
}

void NeuronGroup::Reset()
{
	for (unsigned i = 0; i < neurons.size(); i++)
		neurons[i] = 0.0;
}

void NeuronGroup::ResetOutgoingGroups()
{
	//Weights randomized between 0.0 and 1.0.
	for (unsigned i = 0; i < outgoingGroups.size(); i++)
		outgoingGroups[i]->ResetWeights();
}

void NeuronGroup::ComputeSignal()
{
	for (unsigned i = 0; i < incomingGroups.size(); i++)
		incomingGroups[i]->PropagateSignal();

	//Finish computing the signal by applying the activation function.
	//Add noise if Hebbian trained connections are present.
	if (useNoise)
	{
		for (unsigned i = 0; i < neurons.size(); i++)
		{
			double noise = ann->NextDouble() * ann->getOutputNoiseRange() - ann->getOutputNoiseMagnitude();
			neurons[i] = ann->activation(neurons[i]) + noise;
		}
	}
	else
	{
		for (unsigned i = 0; i < neurons.size(); i++)
			neurons[i] = ann->activation(neurons[i]);
	}

	computed = true;
}

//Train groups which use the most recent experience.
double NeuronGroup::TrainRecent()
{
	if (outTrainRecent.size() < 1)
		return TrainingMethod::NO_ERROR;

	double error = 0.0;

	for (unsigned i = 0; i < outTrainRecent.size(); i++)
	{
		if (outTrainRecent[i]->GetTrainingMethod() == TrainingMethod::SparseAutoencoderTrain)
			UpdateAverages();

		error += outTrainRecent[i]->Train();
	}

	return error;
}

//Train groups which use several randomly selected experiences.
double NeuronGroup::TrainSeveral()
{
	if (outTrainSeveral.size() < 1)
		return TrainingMethod::NO_ERROR;

	double error = 0.0;

	unsigned historyBufferCount = 0;

	if (ann->useNovelty)
		historyBufferCount = ann->getNoveltyBufferSize();
	else
		historyBufferCount = ann->getHistoryBufferSize();

	vector<vector<double>> samples = vector<vector<double>>();

	for (unsigned i = 0; i < outTrainSeveral.size(); i++)
	{
		if (ann->useNovelty)
		{
			for (NeuralNetwork::NoveltyBufferOccupant *occupant : ann->getNoveltyBuffer())
				samples.push_back(occupant->experience);
		}
		else
		{
			for (vector<double> sample : ann->getHistoryBuffer())
				samples.push_back(sample);
		}

		unsigned sampleCount = outTrainSeveral[i]->sampleUsageCount;

		if (sampleCount > historyBufferCount)
			sampleCount = historyBufferCount;

		for (unsigned y = 0; y < sampleCount; y++)
		{
			//Select a random sample.
			//vector<double> sample = samples.ElementAt(NeuralNetwork.rand.Next(samples.Count)); // TODO: MABE RANDOM
			//samples.erase(sample);
			unsigned index = rand() % samples.size(); // TODO: MABE RANDOM
			vector<double> sample = samples[index];
			samples.erase(samples.begin() + index);

			ann->SetExperience(sample);
			ann->PropagateSignal();

			outTrainSeveral[i]->UpdateAverages();

			error += outTrainSeveral[i]->Train();
		}

		//Divide by the number of samples used.
		error /= sampleCount;

		samples.clear();
	}

	return error;
}

unsigned NeuronGroup::GetNeuronCount()
{
	unsigned count = (unsigned)neurons.size();

	for (unsigned i = 0; i < outgoingGroups.size(); i++)
	{
		if (outgoingGroups[i]->UsesBiasWeights())
			return count + 1;
	}

	return count;
}

vector<double> NeuronGroup::GetWeights(NeuronGroup::Identifier toGroup)
{
	for (unsigned i = 0; i < outgoingGroups.size(); i++)
	{
		if (outgoingGroups[i]->IsConnectedTo(toGroup))
			return outgoingGroups[i]->GetWeights();
	}

	return vector<double>();
}

vector<NeuronGroup::Identifier> NeuronGroup::GetGroupsConnected()
{
	vector<NeuronGroup::Identifier> groupsConnected = vector<NeuronGroup::Identifier>(outgoingGroups.size());

	for (unsigned i = 0; i < outgoingGroups.size(); i++)
	{
		NeuronGroup::Identifier ident;
		ident.type = outgoingGroups[i]->GetOutputGroupType();
		ident.index = outgoingGroups[i]->GetOutputGroupIndex();

		// does not contain ident
		if (find(groupsConnected.begin(), groupsConnected.end(), ident) == groupsConnected.end())
			groupsConnected.push_back(ident);
	}

	return groupsConnected;
}

//Returns true if the neuron was able to be removed, false otherwise.
bool NeuronGroup::RemoveNeuron(unsigned index)
{
	if (index < neurons.size())
	{
		neurons.erase(neurons.begin() + index);
		return true;
	}
	else
		return false;
}

double NeuronGroup::GetReconstructionError()
{
	double error = 0.0;

	for (unsigned i = 0; i < outgoingGroups.size(); i++)
		error += outgoingGroups[i]->GetReconstructionError();

	return error;
}				