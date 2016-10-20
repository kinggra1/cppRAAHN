#pragma once
#include "ConnectionGroup.h"
#include "NeuronGroup.h"
#include "TrainingMethod.h"
#include "Activation.h"
#include <vector>
#include <queue>
#include <deque>
#include <random>

using std::deque;
using std::queue;
using std::vector;
using std::uniform_real_distribution; using std::default_random_engine;


class NeuralNetwork
{
public:

		

	// FROM C# for reference: delegate double ActivationFunctionType(double x);
	typedef double(*ActivationFunctionType)(double);


	const unsigned DEFAULT_HISTORY_BUFFER_SIZE = 1;

	//How many distances to keep for each experience.
	const unsigned N_KEEP = N_NEAREST + 20;
	const double DEFAULT_NOISE_MAGNITUDE = 1.0;
	const double DOUBLE_MAGNITUDE = 2.0;
	const double WEIGHT_RANGE_SCALE = 6.0;
	const double DOUBLE_WEIGHT_RANGE = 2.0;


	// replacement of Random.NextDouble()
	// default initialization gives us values in the range [0, 1) which is desired
	uniform_real_distribution<double> rand;
	default_random_engine eng;


	bool useNovelty;
	//Default to using the logistic function.
	ActivationFunctionType activation = Activation::Logistic;
	ActivationFunctionType activationDerivative = Activation::LogisticDerivative;


	NeuralNetwork() {};

	NeuralNetwork(unsigned historySize, bool useNoveltyBuffer);

	NeuralNetwork(double outputNoiseMag, double weightNoiseMag);

	NeuralNetwork(unsigned historySize, double outputNoiseMag, double weightNoiseMag, bool useNoveltyBuffer);

	//Adds a training sample to the history buffer.
	void AddExperience(vector<double> newExperience);

	//Public access for setting an experience.
	void SetExperience(unsigned index);

	//Propagates the inputs completely and gets output.
	void PropagateSignal() {};

	//Returns autoencoder error.
	void Train();

	//Resets the weights and every neuron of the neural network.
	void Reset();

	//Sets the maximum weight value. Ignores sign.
	void SetWeightCap(double cap);

	void SetOutputNoiseMagnitude(double outputNoiseMag);

	void SetWeightNoiseMagnitude(double weightNoiseMag);

	//Returns whether the output was able to be set.
	bool SetOutput(unsigned groupIndex, unsigned index, double value);

	//Returns false if one or both of the groups do not exist.
	//Returns true if the groups could be connected.
	//Sample count refers to how many training samples should be used each time Train() is called.
	bool ConnectGroups(NeuronGroup::Identifier input, NeuronGroup::Identifier output,
		ConnectionGroup::TrainFunctionType trainMethod, int modulationIndex,
		unsigned sampleCount, double learningRate, bool useBias);

	//Gets the number of neurons in a group. Returns 0 if the group is invalid.
	unsigned GetGroupNeuronCount(NeuronGroup::Identifier ident);

	//Returns the index of the neuron group.
	int AddNeuronGroup(unsigned neuronCount, NeuronGroup::Type type);

	double GetWeightCap() { return weightCap; }

	//Returns double.Nan if the neuron or neuron group does not exist.
	double GetNeuronValue(NeuronGroup::Identifier ident, unsigned neuronIndex);

	//Returns double.Nan if the neuron or neuron group does not exist.
	double GetOutputValue(unsigned groupIndex, unsigned index);

	//Returns the sum of the squared reconstruction error for the current sample.
	double CalculateBatchError();

	//Returns the average reconstruction error for the entire history buffer.
	double GetBatchError();

	//Returns the average reconstruction error over the past [historyBufferSize] ticks.
	double GetOnlineError();

	//Get neuron values of a neuron group.
	vector<double> GetNeuronValues(NeuronGroup::Identifier nGroup);

	//Get the strength of connections in a connection group.
	vector<double> GetWeights(NeuronGroup::Identifier fromGroup, NeuronGroup::Identifier toGroup);

	//Returns the Ids of all groups connected by outgoing connections to the specifed group.
	vector<NeuronGroup::Identifier> GetGroupsConnected(NeuronGroup::Identifier connectedTo);






		class NoveltyBufferOccupant;

		class DistanceDescription {
		public:
			DistanceDescription() {};
				
			bool operator>(const DistanceDescription &other) {
				return (distance > other.distance);
			}

			bool operator<(const DistanceDescription &other) {
				return (distance < other.distance);
			}

			bool operator==(const DistanceDescription &other) {
				return (distance == other.distance);
			}


			double distance = 0.0;
			NoveltyBufferOccupant *distanceOwner = nullptr;
		};

		/*
		//Description of a distance between two novelty buffer occupants.
		class DistanceDescription : IComparable<DistanceDescription>
		{
			double distance;
			NoveltyBufferOccupant *distanceOwner;

			DistanceDescription()
			{
				distance = 0.0;
				distanceOwner = null;
			}

			//Sort DistanceDescription by distance.
			int CompareTo(DistanceDescription cmp)
			{
				if (cmp == null)
					return 1;
				if (distance > cmp.distance)
					return 1;
				if (distance < cmp.distance)
					return -1;

				return 0;
			}
		};

		*/



	class NoveltyBufferOccupant {
	public:
		NoveltyBufferOccupant() {};

		bool operator>(const NoveltyBufferOccupant &other) {
			return (noveltyScore > other.noveltyScore);
		}

		bool operator<(const NoveltyBufferOccupant &other) {
			return (noveltyScore < other.noveltyScore);
		}

		bool operator==(const NoveltyBufferOccupant &other) {
			return (noveltyScore == other.noveltyScore);
		}


		double noveltyScore = 0.0;
		vector<double> experience;
		vector<DistanceDescription*> distanceDescriptions;
	};

	/*
	class NoveltyBufferOccupant : IComparable<NoveltyBufferOccupant>
	{
	public:
		double noveltyScore;
		vector<double> experience;
		//Ordered from nearest to farthest.
		Linkedvector<DistanceDescription> distanceDescriptions;

		NoveltyBufferOccupant()
		{
			noveltyScore = 0.0;

			experience = null;
			distanceDescriptions = new Linkedvector<DistanceDescription>();
		}

		//Sort NoveltyBufferOccupant by noveltyScore.
		int CompareTo(NoveltyBufferOccupant cmp)
		{
			if (cmp == null)
				return 1;
			if (noveltyScore > cmp.noveltyScore)
				return 1;
			if (noveltyScore < cmp.noveltyScore)
				return -1;

			return 0;
		}
	};
	*/


	//Constructs NeuralNetwork.
	void Construct(unsigned historySize, double outputNoiseMag, double weightNoiseMag, bool useNoveltyBuffer);

	//Set the inputs of the neural network to a given experience.
	void SetExperience(vector<double> sample) {};

	void UpdateOnlineError(double currentError);

	//Add a new experience to the novelty buffer.
	void AddNoveltyOccupant(NoveltyBufferOccupant newOccupant, vector<DistanceDescription> distDescriptions);

	//Insert the distance if it is closer than the current farthest distance.
	void TryInsertDistance(NoveltyBufferOccupant occupant, DistanceDescription distDesc);

	//Remove an experience from the novelty buffer.
	void RemoveNoveltyOccupant(NoveltyBufferOccupant oldOccupant);

	void UpdateNoveltyScores();

	//Expensive computation of distances for an experience already in buffer.
	void ComputeDistances(NoveltyBufferOccupant occupant);

	//Makes sure a type is INPUT, HIDDEN, or OUTPUT.
	bool VerifyType(NeuronGroup::Type type);

	//Makes sure an identifier specifies a neuron group within allvectorGroups.
	bool VerifyIdentifier(NeuronGroup::Identifier ident);

	double ExpDistance(vector<double> exp, vector<double> compare);

	//Expensive computation of novelty score for a new experiences.
	vector<DistanceDescription*> ComputeNewDistances(NoveltyBufferOccupant *occupant);







	// ADDED HELPERS
	double NextDouble() {
		return rand(eng);
	}

	double getOutputNoiseRange() {
		return outputNoiseRange;
	}

	double getOutputNoiseMagnitude() {
		return outputNoiseMagnitude;
	}

	double getWeightNoiseRange() {
		return weightNoiseRange;
	}

	double getWeightNoiseMagnitude() {
		return weightNoiseMagnitude;
	}

	unsigned getNoveltyBufferSize() {
		return noveltyBuffer.size();
	}

	unsigned getHistoryBufferSize() {
		return historyBuffer.size();
	}

	vector<NoveltyBufferOccupant*> getNoveltyBuffer() {
		return noveltyBuffer;
	}

	deque<vector<double>> getHistoryBuffer() {
		return historyBuffer;
	}







private:


	unsigned historyBufferSize;
	double weightCap;
	double outputNoiseMagnitude;
	double weightNoiseMagnitude;
	//Difference between max and min noise values.
	double outputNoiseRange;
	double weightNoiseRange;
	double averageError;
	vector<vector<NeuronGroup*>> allvectorGroups;
	vector<NeuronGroup*> inputGroups;
	vector<NeuronGroup*> hiddenGroups;
	vector<NeuronGroup*> outputGroups;
	//Ordered from least novel to most novel.
	vector<NoveltyBufferOccupant*> noveltyBuffer;
	queue<double> errorBuffer;
	deque<vector<double>> historyBuffer;


	//Whether or not to use a novelty buffer.
	const bool DEFAULT_NOVELTY_USE = false;
	//Number of nearest neighbors to use for novelty score calculations.
	const unsigned N_NEAREST = 20;
};
