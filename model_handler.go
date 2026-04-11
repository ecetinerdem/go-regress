package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/ecetinerdem/go-regress/model"
)

func getOrTrainModel(config Config, logger *log.Logger) (*model.LinearRegression, *DataContext, error) {
	var dataModel *model.LinearRegression
	var err error
	var dataContext DataContext

	// Aplication either can load a model or train a new one
	if config.LoadModelPath != "" {
		// Load the existing model from path as JSON
		dataModel, err := model.LoadModelFromJSON(config.LoadModelPath)
		if err != nil {
			return nil, nil, fmt.Errorf("error loading model: %v", err)
		}

		return dataModel, &dataContext, nil
	}

	// Train a new model from a csv file
	if config.CSVFilePath == "" {
		flag.Usage()
		return nil, nil, fmt.Errorf("Please provide a path to the csv file the --file flag")
	}

	// Load and prepare training data
	dataContext, err = loadAndPrepareDate(config, logger)
	if err != nil {
		return nil, nil, err
	}

	// Train linear regression model using data
	logger.Printf("Training model with features: %v\n", config.FeatureVars)
	dataModel, err = model.TrainLinearRegression(dataContext.Data, config.FeatureVars, config.TargetVariable, config.Normalize)
	if err != nil {
		return nil, nil, fmt.Errorf("error training model: %v", err)
	}

	dataModel.PrintModelSummary()

	return dataModel, &dataContext, nil
}
