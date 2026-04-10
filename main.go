package main

import (
	"log"
	"os"
)

func main() {
	// Parse command line arguments
	config := parseCommandLineArgs()
	// Set up a logger
	logger := log.New(os.Stdout, "", log.LstdFlags)

	logger.Println("Parsed command line flags", config.FeatureVars)

	// Either load or train a model
	_, _, err := getOrTrainModel(config, logger)

	if err != nil {
		logger.Fatalf("model error: %v", err)
	}

}
