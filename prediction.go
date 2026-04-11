package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ecetinerdem/go-regress/model"
)

func handlePrediction(config Config, dataModel *model.LinearRegression) {
	// Sanity check
	if len(config.DataToPredict) == 0 {
		return
	}

	// PArse the key value string into a map
	kvMap := make(map[string]string)
	pairs := strings.SplitSeq(config.DataToPredict, ",")

	for pair := range pairs {
		kv := strings.Split(pair, "=")

		if len(kv) == 2 {
			kvMap[kv[0]] = kv[1]
		}
	}

	// Make sure requested features are the same in kind and number,
	// As the features in our trained model.

	if len(kvMap) != len(dataModel.Features) {
		fmt.Println("Cannot do prediction of new data, wrong number of features specified")
		return
	} else {
		// Now make sure our requested features for prediction match the features field
		for _, feature := range dataModel.Features {
			if kvMap[feature] == "" {
				fmt.Println("Cannot do prediction of newdata; incorrect features requested. Use", strings.Join(dataModel.Features, ","))
				return
			}
		}
	}
	// Create an array of floats in the same order as features in the model
	// This ensures the values match up with the correct coefficients
	values := make([]float64, len(dataModel.Features))

	for i, feature := range dataModel.Features {
		if val, ok := kvMap[feature]; ok {
			if f, err := strconv.ParseFloat(val, 64); err == nil {
				values[i] = f
			}
		}
	}

	// Prepare input for prediction
	var newData [][]float64
	newData = append(newData, values)

	// Use this to make predictions
	predictions := dataModel.Predict(newData)

	// Display the results
	fmt.Println("\nPredictions:")

	displayPredictionTable(newData, predictions, dataModel)

}

func displayPredictionTable(newData [][]float64, predictions []float64, dataModel *model.LinearRegression) {
}
