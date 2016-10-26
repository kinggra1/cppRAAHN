// RAAHN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "NeuralNetwork.h"

#include <iostream>

using std::cin; using std::cout; using std::endl;


int main()
{
	NeuralNetwork brain = NeuralNetwork();

	int input_idx = brain.AddNeuronGroup(5, NeuronGroup::Type::INPUT);
	int hidden_idx = brain.AddNeuronGroup(10, NeuronGroup::Type::HIDDEN);
	int output_idx = brain.AddNeuronGroup(1, NeuronGroup::Type::OUTPUT);

	NeuronGroup::Identifier input;
	input.index = input_idx;
	input.type = NeuronGroup::Type::INPUT;
	
	NeuronGroup::Identifier hidden;
	hidden.index = hidden_idx;
	hidden.type = NeuronGroup::Type::HIDDEN;

	NeuronGroup::Identifier output;
	output.index = output_idx;
	output.type = NeuronGroup::Type::OUTPUT;


	ConnectionGroup::TrainFunctionType trainMethod = TrainingMethod::HebbianTrain;

	unsigned modSig = ModulationSignal::AddSignal();


	cout << brain.ConnectGroups(&input, &hidden, trainMethod, (int)modSig, 10, 0.1, true);
	cout << brain.ConnectGroups(&hidden, &output, trainMethod, (int)modSig, 10, 0.1, true);

	brain.SetWeightNoiseMagnitude(0.1);
	brain.SetOutputNoiseMagnitude(0.1);



	for (double i = 0; i < 50; i++) {
		cout << brain.GetOutputValue(0, 0) << endl;
		brain.AddExperience({ i, i, i, i, i });
		brain.PropagateSignal();
		brain.Train();
	}
}

