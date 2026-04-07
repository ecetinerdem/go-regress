package utils

import (
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/go-gota/gota/dataframe"
)

func ValidateData(
	df dataframe.DataFrame,
	features []string,
	target string,
	lowerBoundMultiplier float64,
	upperBoundMultiplier float64,
) (dataframe.DataFrame, error) {
	// Get all columns and required columns
	allColumns := df.Names()
	requiredColumns := slices.Clone(features)

	requiredColumns = append(requiredColumns, target)

	for _, col := range requiredColumns {
		if !slices.Contains(allColumns, col) {
			return df, fmt.Errorf("column %s not found in the dataset", col)
		}
	}

	// Validate all columns for missing and negative data

	for _, colName := range requiredColumns {
		col := df.Col(colName)
		for i := range col.Len() {
			value := col.Elem(i).Float()
			if math.IsNaN(value) {
				return df, fmt.Errorf("missing value found in column %s at row %d", colName, i+1)
			}
			if value < 0 {
				return df, fmt.Errorf("negative value found in column %s at row %d", colName, i+1)
			}
		}
	}

	// Track valid rows (non-outliers)
	validRows := make([]bool, df.Nrow())
	for i := range validRows {
		validRows[i] = true
	}

	// Find outliers in columns at once
	outlierCount := 0

	for _, colName := range requiredColumns {
		values := df.Col(colName).Float()
		if len(values) < 4 {
			continue
		}

		// Lets calculate quartiles and IQR
		sortedValues := make([]float64, len(values))
		copy(sortedValues, values)
		sort.Float64s(sortedValues)

		n := len(sortedValues)

		q1, q3 := sortedValues[n/4], sortedValues[(3*n)/4]

		IQR := q3 - q1
		// Define my bounds
		lowerBound := q1 - lowerBoundMultiplier*IQR
		upperBound := q3 + upperBoundMultiplier*IQR

		// Identify outliers
		for i, value := range values {
			if value < lowerBound || value > upperBound {
				validRows[i] = false
				outlierCount++
				// Only log a few to avoid console flooding
				if outlierCount <= 3 {
					fmt.Println("   - Removing outlier in '%s' at row '%d': %.2f (outside range %.2f - %.2f)\n", colName, i+1, value, lowerBound, upperBound)
				}

			}
		}
	}

	// Build a list of row indecies to keep
	rowsToKeep := make([]int, df.Nrow())
	for i, isValid := range validRows {
		if isValid {
			rowsToKeep = append(rowsToKeep, i)
		}
	}

	// Print information about rows if any
	if outlierCount > 0 {
		fmt.Printf("Removed %d outlier record (%.1%% of data)\n", outlierCount, 100*float64(outlierCount)/float64(df.Nrow()))
	}

	// Return filtered data fram if there are rows to drop
	if len(rowsToKeep) < df.Nrow() {
		return df.Subset(rowsToKeep), nil
	}

	return df, nil
}

func CheckDatasetSize(numSamples int, numFeatures int) error {
	minRecommendedSamples := numFeatures * 20

	if numSamples < numFeatures+2 {
		return fmt.Errorf("Dataset has  too few samples (%d) for the number of features (%d) - model will be overfitted", numSamples, numFeatures)
	} else if numSamples < minRecommendedSamples {
		return fmt.Errorf("datasetsize (%d sample) is smaller than the recommended %d samples for %d features. Results maybe unreliable",
			numSamples, minRecommendedSamples, numFeatures,
		)
	}
	return nil
}
