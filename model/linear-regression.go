package model

import (
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/ecetinerdem/go-regress/utils"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type LinearRegression struct {
	Coefficients    []float64 `json:"coefficients,omitempty"`
	Intercept       float64   `json:"intercept,omitempty"`
	Features        []string  `json:"features,omitempty"`
	Target          string    `json:"target,omitempty"`
	RSquared        float64   `json:"rsquared,omitempty"`
	FeatureMeans    []float64 `json:"feature_means,omitempty"`
	FeaturesStdDevs []float64 `json:"feature_std_devs,omitempty"`
	IsNormalized    bool      `json:"is_normalized,omitempty"`
	SavedAt         time.Time `json:"-"`
	Description     string    `json:"description,omitempty"`
	NumSamples      int       `json:"num_samples,omitempty"`
	Version         string    `json:"version,omitempty"`
}

func (lr *LinearRegression) Predict(featureValues [][]float64) []float64 {
	predictions := make([]float64, len(featureValues))

	for dataPointIndex, featureRow := range featureValues {
		// Start with intercept
		predictedValue := lr.Intercept

		normalizedFeatures := make([]float64, len(featureRow))
		copy(normalizedFeatures, featureRow)

		if lr.IsNormalized && len(lr.FeatureMeans) == len(featureRow) {
			for i := range normalizedFeatures {
				if lr.FeaturesStdDevs[i] > 0 {
					normalizedFeatures[i] = (featureRow[i] - lr.FeatureMeans[i]) / lr.FeaturesStdDevs[i]
				} else {
					normalizedFeatures[i] = 0
				}
			}
		}
		// Add contribution of each feature
		for featureIndex, coefficient := range lr.Coefficients {
			if lr.IsNormalized {
				predictedValue += coefficient * normalizedFeatures[featureIndex]
			} else {
				predictedValue += coefficient * featureRow[featureIndex]
			}
		}
		predictions[dataPointIndex] = predictedValue
	}
	return predictions
}

func TrainLinearRegression(dataFrame dataframe.DataFrame, featureNames []string, target string, normalize bool) (*LinearRegression, error) {
	// Check if all feature columns and target column exist
	columnNames := dataFrame.Names()

	for _, name := range featureNames {
		if !slices.Contains(columnNames, name) {
			return nil, fmt.Errorf("feature column '%s' not found in dataset", name)
		}
	}

	if !slices.Contains(columnNames, target) {
		return nil, fmt.Errorf("target column '%s' not found in dataset", target)
	}

	// Get feature columns as float slices
	featureColumns := make([]series.Series, len(featureNames))
	for i, name := range featureNames {
		featureColumns[i] = dataFrame.Col(name)
	}

	// Get target column as float slice
	targetColumn := dataFrame.Col(target)

	numSamples := dataFrame.Nrow()
	featureMatrix := make([][]float64, numSamples)
	targetValues := make([]float64, numSamples)

	//Fill feature matrix (X) and target vector (Y) with values from the dataframe
	for rowIndex := range numSamples {
		featureMatrix[rowIndex] = make([]float64, len(featureNames))
		for colIndex, column := range featureColumns {
			featureMatrix[rowIndex][colIndex] = column.Elem(rowIndex).Float()
		}
		targetValues[rowIndex] = targetColumn.Elem(rowIndex).Float()
	}

	// Variables to store normalization variables
	var normalizedFeatures [][]float64
	var featureMeans []float64
	var featureStdDevs []float64

	if normalize {
		normalizedFeatures, featureMeans, featureStdDevs = utils.NormalizeFeatures(featureMatrix)
		featureMatrix = normalizedFeatures
	}

	// Create a design matrix
	numFeatures := len(featureNames)
	designMatrix := mat.NewDense(numSamples, numFeatures+1, nil)
	targetVector := mat.NewVecDense(numSamples, nil)

	for rowIndex := range numSamples {
		designMatrix.Set(rowIndex, 0, 1.0)
		for featureIndex := range numFeatures {
			designMatrix.Set(rowIndex, featureIndex+1, featureMatrix[rowIndex][featureIndex])
		}
		targetVector.SetVec(rowIndex, targetValues[rowIndex])
	}

	// Step 1: Calculate X^T x (transpose of X multiplied by X)
	var transposeTimeDesign mat.Dense
	transposeTimeDesign.Mul(designMatrix.T(), designMatrix)

	// Step 2: Calculate (X^T)^(-1)
	var inverseMatrix mat.Dense
	if err := inverseMatrix.Inverse(&transposeTimeDesign); err != nil {
		return nil, fmt.Errorf("failed to compute inverse: %v - matrix may be singular; try adding more data or remove highly co-related features", err)
	}

	// Step 3: Calculate X^T y
	var transposeTimesTarget mat.Dense
	transposeTimesTarget.Mul(designMatrix.T(), targetVector)

	// Step 4: Calculate optimal co-efficients
	var coefficientMatrix mat.Dense
	coefficientMatrix.Mul(&inverseMatrix, &transposeTimesTarget)

	// Extract co-efficients
	interceptAndCoefficients := make([]float64, numFeatures+1)
	for i := range numFeatures + 1 {
		interceptAndCoefficients[i] = coefficientMatrix.At(i, 0)
	}

	// Calculate predictions using the trained model
	predictedValues := make([]float64, numSamples)
	for i := range numSamples {
		// Start with intercept
		predictedValues[i] = interceptAndCoefficients[0]

		// Add contribution of each feature
		for j := range numFeatures {
			predictedValues[i] += interceptAndCoefficients[j+1] * featureMatrix[i][j]
		}
	}

	// Calculate R-squared
	targetMean := stat.Mean(targetValues, nil)

	var totalSumofSquares, sumOfSquareResidiuals float64

	for i := range numSamples {
		totalSumofSquares += math.Pow(
			targetValues[i]-targetMean,
			2,
		)
		sumOfSquareResidiuals += math.Pow(targetValues[i]-predictedValues[i], 2)
	}

	rSquared := 1 - (sumOfSquareResidiuals / totalSumofSquares)

	return &LinearRegression{
		Intercept:       interceptAndCoefficients[0],
		Coefficients:    interceptAndCoefficients[1:],
		Features:        featureNames,
		Target:          target,
		RSquared:        rSquared,
		FeatureMeans:    featureMeans,
		FeaturesStdDevs: featureStdDevs,
		IsNormalized:    normalize,
	}, nil

}

func (lr *LinearRegression) PrintModelSummary() {
	fmt.Println("\n==== Model Summary ====\n")
	fmt.Printf("Regression Equation: %s = %.4f", lr.Target, lr.Intercept)

	for i, feature := range lr.Features {
		if lr.Coefficients[i] >= 0 {
			fmt.Printf(" + %.4f x %s", lr.Coefficients[i], feature)
		} else {
			fmt.Printf(" - %.4f x %s", -lr.Coefficients[i], feature)
		}
	}
	fmt.Println()

	// Display model fit statistics
	fmt.Printf("\nModel Performance:\n")
	fmt.Printf(" - R-squared: %.4f\n", lr.RSquared)
	fmt.Printf(" - Interpretation: %.2f%% of variance in %s is explained by this model\n", lr.RSquared*100, lr.Target)

	fmt.Printf("\nCoefficient Interpretation:\n")
	fmt.Printf("- Intercept (%.4f): The base %s when all features are zero\n", lr.Intercept, lr.Target)

	for i, feature := range lr.Features {
		fmt.Printf(" - %s Coefficient (%.4f): For each additional unit of %s, %s changes by %.4f units\n", feature, lr.Coefficients[i], feature, lr.Target, lr.Coefficients[i])
	}

	if lr.IsNormalized {
		fmt.Printf("\nNote: This model was trained on normalized data. Predictions on new data will automatically be normalized.\n")
	}
}
