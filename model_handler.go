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
		// TODO: Load the model from path
	}

	// Train a new model from a csv file
	if config.CSVFilePath == "" {
		flag.Usage()
		return nil, nil, fmt.Errorf("Please provide a path to the csv file the --file flag")
	}

	dataContext, err = loadAndPrepareDate(config, logger)

	if err != nil {
		return nil, nil, err
	}

	return dataModel, &dataContext, nil
}
