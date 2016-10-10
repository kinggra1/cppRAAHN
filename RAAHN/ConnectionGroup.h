#pragma once
#include "ModulationSignal.h"
#include "ConnectionGroup.h"
#include "NeuronGroup.h"
#include <string>
#include <math.h>

using std::string;

class NeuralNetwork;
class TrainingMethod;

namespace Raahn
{
	class Connection
	{
	public:
		unsigned input;
		unsigned output;
		double weight;

		Connection();
		Connection(unsigned i, unsigned o, double w);
	};

	class ConnectionGroup
	{
	public:
		const double DEFAULT_LEARNING_RATE = 0.1;
		unsigned sampleUsageCount;

		struct TrainFunctionType{
			double operator()(int modIndex, double learningRate, NeuralNetwork *ann,
				NeuronGroup *inGroup, NeuronGroup *outGroup,
				vector<Connection*> connections, vector<double> biasWeights);
		};

		ConnectionGroup(NeuralNetwork *network, NeuronGroup *inGroup, NeuronGroup *outGroup, bool useBias);

		void AddConnection(unsigned inputIndex, unsigned outputIndex, double weight);

		void AddBiasWeights(unsigned outputCount);

		void PropagateSignal();

		void UpdateAverages();

		void SetModulationIndex(int index);

		void SetLearningRate(double lRate);

		void SetTrainingMethod(TrainFunctionType method);

		void ResetWeights();

		int GetInputGroupIndex();

		int GetOutputGroupIndex();

		double Train();

		double GetReconstructionError();

		TrainFunctionType GetTrainingMethod();

		bool UsesBiasWeights();

		//Returns true if the connection was able to be removed, false otherwise.
		bool RemoveConnection(unsigned index);

		bool IsConnectedTo(NeuronGroup::Identifier toGroup);

		NeuronGroup::Type GetInputGroupType();

		NeuronGroup::Type GetOutputGroupType();

		//Returns a copy of the weights.
		vector<double> GetWeights();

		private:
			const string WEIGHTS_FILE = "weights.txt";

			int modSigIndex;
			//Learning rate for all connections within the group.
			double learningRate;
			vector<double> biasWeights;
			vector<Connection*> connections;
			NeuralNetwork *ann;
			NeuronGroup *inputGroup;
			NeuronGroup *outputGroup;
			double (*trainingMethod)(int modIndex, double learningRate, NeuralNetwork *ann,
				NeuronGroup *inGroup, NeuronGroup *outGroup,
				vector<Connection*> connections, vector<double> biasWeights);
	};
}
