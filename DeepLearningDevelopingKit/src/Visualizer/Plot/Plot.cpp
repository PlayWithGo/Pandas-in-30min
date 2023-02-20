#include "Plot.hpp"

class PerlinNoise {
public:
	double operator()(float x, float y) {
		double total = 0;
		double p = persistence;
		int n = Number_Of_Octaves;
		for (int i = 0; i<n; i++)
		{
			double frequency = pow(2, i);
			double amplitude = pow(p, i);
			total = total + InterpolatedNoise(x * frequency, y * frequency) * amplitude;
		}
		return total;
	}
private:
	float persistence = 0.50;
	int Number_Of_Octaves = 4;

	double Noise(int x, int y)
	{
		int n = x + y * 57;
		n = (n << 13) ^ n;
		return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0);
	}

	double SmoothedNoise(int x, int y)
	{
		double corners = (Noise(x - 1, y - 1) + Noise(x + 1, y - 1) + Noise(x - 1, y + 1) + Noise(x + 1, y + 1)) / 16;
		double sides = (Noise(x - 1, y) + Noise(x + 1, y) + Noise(x, y - 1) + Noise(x, y + 1)) / 8;
		double center = Noise(x, y) / 4;
		return corners + sides + center;
	}

	double Cosine_I