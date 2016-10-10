#pragma once

namespace Raahn
{
	class Activation
	{
		// activation function of 1/(1+e^-x)
		static double Logistic(double x);

		//Takes the already computed value of sigmoid.
		static double LogisticDerivative(double x);
	};
}