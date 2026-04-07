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
}
