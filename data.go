package main

import (
	"fmt"
	"log"
	"os"

	"github.com/ecetinerdem/go-regress/utils"
	"github.com/go-gota/gota/dataframe"
)

type DataContext struct {
	Data         dataframe.DataFrame
	FeatureData  map[string][]float64
	TargetValues []float64
}

func loadAndPrepareDate(config Config, logger *log.Logger) (DataContext, error) {
	var dataContext DataContext

	// Read data from csv
	datafile, err := os.Open(config.CSVFilePath)
	if err != nil {
		return dataContext, fmt.Errorf("Could not open file: %v", err)
	}

	// Parse csv into dataframe
	dataContext.Data = dataframe.ReadCSV(datafile)

	// Display a summary of the data for the user
	printDataSummary(dataContext.Data, logger, "before outlier removal")

	// Validate and remove outliers
	dataContext.Data, err = utils.ValidateData(dataContext.Data, config.FeatureVars, config.TargetVariable, config.OutlierLowerBound, config.OutlierUpperBound)
	if err != nil {
		return dataContext, fmt.Errorf("Data validation error: %v,", err)
	}

	if dataContext.Data.Nrow() > 0 {
		printDataSummary(dataContext.Data, logger, "after outlier removal")
	}

	if err := utils.CheckDatasetSize(dataContext.Data.Nrow(), len(config.FeatureVars)); err != nil {
		logger.Printf("Warning: %v\n", err)
	}

	dataContext.FeatureData = make(map[string][]float64)
	for _, feature := range config.FeatureVars {
		featureCol := dataContext.Data.Col(feature)
		featureValues := make([]float64, featureCol.Len())

		for i := range featureCol.Len() {
			featureValues[i] = featureCol.Elem(i).Float()
		}

		dataContext.FeatureData[feature] = featureValues
	}

	targetCol := dataContext.Data.Col(config.TargetVariable)
	dataContext.TargetValues = make([]float64, targetCol.Len())

	for i := range targetCol.Len() {
		dataContext.TargetValues[i] = targetCol.Elem(i).Float()
	}

	return dataContext, nil

}

func printDataSummary(df dataframe.DataFrame, logger *log.Logger, stage string) {
	logger.Printf("Data Preview (%s):\n", stage)
	logger.Println(df.Describe())
	logger.Printf("Columns in dataset: %v \n", df.Names())
	logger.Printf("Row Count: %d\n", df.Nrow())

	// Show sample rows
	if df.Nrow() > 0 {
		// Get the minimum of 3 or the number of columns available
		numCols := min(df.Ncol(), 3)

		// Get the minimum of 5 or rows available
		numRows := min(df.Nrow(), 5)

		columnsToShow := make([]int, numCols)

		for i := range numCols {
			columnsToShow[i] = i
		}

		logger.Println(df.Select(columnsToShow).Subset(numRows))
	}
}
