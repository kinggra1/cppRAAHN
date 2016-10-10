#include "stdafx.h"
#include "Activation.h"

#include <math.h>

namespace Raahn
{
	double Activation::Logistic(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}

	double Activation::LogisticDerivative(double x)
	{
		return x * (1.0 - x);
	}
}