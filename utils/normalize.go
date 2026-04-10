package utils

import "math"

func NormalizeFeatures(X [][]float64) ([][]float64, []float64, []float64) {
	// Handle empty input case
	if len(X) == 0 || len(X[0]) == 0 {
		return X, []float64{}, []float64{}
	}

	numSamples := len(X)
	numFeatures := len(X[0])

	// Initialize slices to store the mean and standart deviation for each features
	means := make([]float64, numFeatures)
	stdDevs := make([]float64, numFeatures)
	normalizedX := make([][]float64, numSamples)

	for i := range normalizedX {
		normalizedX[i] = make([]float64, numFeatures)
	}

	// Calculate means and stdDevs in a single pass per feature
	for j := range numFeatures {
		// Calculate mean
		var sum float64
		for i := range numSamples {
			sum += X[i][j]
		}

		means[j] = sum / float64(numSamples)

		// Calculate stdDev
		var varianceSum float64
		for i := range numSamples {
			diff := X[i][j] - means[j]
			varianceSum += diff * diff
		}

		// Prevent division by zero with small epsilon
		epsilon := 1e-10
		stdDevs[j] = math.Max(math.Sqrt(varianceSum/float64(numSamples)), epsilon)

		// Normalize values for this feature
		for i := range numSamples {
			normalizedX[i][j] = (X[i][j] - means[j]) / stdDevs[j]
		}

	}

	return normalizedX, means, stdDevs
}
