#pragma once
#include <vector>

using std::vector;

class ModulationSignal
{
public:
	static const double NO_MODULATION;
	//-1 to obtain passive modulation from ModulationSignal.GetSignal
	static const int INVALID_INDEX;

	//Returns the index of the signal.
	static unsigned AddSignal();

	//Returns the index of the signal.
	static unsigned AddSignal(double defaultValue);

	//If the modulation does not exist, the default modulation is returnned.
	static double GetSignal(int index);

	static unsigned GetSignalCount();

	static void SetSignal(unsigned index, double value);

private:
	static vector<double> modulations;
};
