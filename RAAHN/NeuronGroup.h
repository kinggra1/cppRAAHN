#pragma once
#include <vector>

using std::vector;

class ConnectionGroup;
class NeuralNetwork;


class NeuronGroup
{
public:
	enum Type
	{
		NONE = -1,
		INPUT = 0,
		HIDDEN = 1,
		OUTPUT = 2
	};

	struct Identifier
	{
		int index;
		Type type;


		// TODO: veryify that this is not breaking things
		bool operator==(const Identifier &other) {
			return (index == other.index && type == other.type);
		}
	};

	static const int INVALID_NEURON_INDEX = -1;

	int index;
	bool computed;
	NeuronGroup::Type type;
	vector<double> neurons;
	vector<double> averages;

	NeuronGroup(NeuralNetwork *network, Type t);

	NeuronGroup(unsigned count, NeuralNetwork *network, Type t);

	void Construct(NeuralNetwork *network, Type t);

	void AddNeurons(unsigned count);

	void AddIncomingGroup(ConnectionGroup *incomingGroup);

	//mostRecent refers to whether the group should train off of only the most recent experience.
	void AddOutgoingGroup(ConnectionGroup *outgoingGroup, bool mostRecent);

	void UpdateAverages();

	void Reset();

	void ResetOutgoingGroups();

	void ComputeSignal();

	//Train groups which use the most recent experience.
	double TrainRecent();

	//Train groups which use several randomly selected experiences.
	double TrainSeveral();

	unsigned GetNeuronCount();

	vector<double> GetWeights(NeuronGroup::Identifier *toGroup);

	vector<NeuronGroup::Identifier> GetGroupsConnected();

	//Returns true if the neuron was able to be removed, false otherwise.
	bool RemoveNeuron(unsigned index);

	double GetReconstructionError();


private:

	const double DEFAULT_NEURON_VALUE = 0.0;
	const double DECAY_BASE = 0.01;

	bool useNoise;
	vector<ConnectionGroup*> incomingGroups;
	//All outgoing groups.
	vector<ConnectionGroup*> outgoingGroups;
	//Outgoing groups that train only off the most recent experience.
	vector<ConnectionGroup*> outTrainRecent;
	//Outgoing groups that train off of several randomly selected experiences.
	vector<ConnectionGroup*> outTrainSeveral;
	NeuralNetwork *ann;
};
