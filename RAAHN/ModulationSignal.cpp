#include "stdafx.h"
#include "ModulationSignal.h"

#include <vector>

using std::vector;



const double ModulationSignal::NO_MODULATION = 0.0;
const int ModulationSignal::INVALID_INDEX = -1;
vector<double> ModulationSignal::modulations;

unsigned ModulationSignal::AddSignal()
{
	modulations.push_back(NO_MODULATION);
	return (unsigned)(modulations.size() - 1);
}
unsigned ModulationSignal::AddSignal(double defaultValue)
{
	modulations.push_back(defaultValue);
	return (unsigned)(modulations.size() - 1);
}

double ModulationSignal::GetSignal(int index)
{
	if (index < 0 || (unsigned)index >= modulations.size())
		return NO_MODULATION;
	else
		return modulations[index];
}

unsigned ModulationSignal::GetSignalCount()
{
	return (unsigned)modulations.size();
}

void ModulationSignal::SetSignal(unsigned index, double value)
{
	if (index >= modulations.size())
		return;

	modulations[(int)index] = value;
}
