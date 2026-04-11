package main

import (
	"log"
	"os"

	"github.com/ecetinerdem/go-regress/model"
)

func main() {
	// Parse command line arguments
	config := parseCommandLineArgs()
	// Set up a logger
	logger := log.New(os.Stdout, "", log.LstdFlags)

	logger.Println("Parsed command line flags", config.FeatureVars)

	// Either load or train a model
	dataModel, dataContext, err := getOrTrainModel(config, logger)

	if err != nil {
		logger.Fatalf("model error: %v", err)
	}

	// Save model if requested
	if config.SaveModelPath != "" {
		if err := model.SaveModelToJSON(dataModel, config.SaveModelPath, config.ModelDescription, dataContext.Data.Nrow()); err != nil {
			log.Fatalf("Error saving model: %v", err)
		}
	}

}
