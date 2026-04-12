package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/ecetinerdem/go-regress/model"
)

func handlePlot(config Config, dataModel *model.LinearRegression, dataContext *DataContext) {
	// Skip if plot flag is not set
	if !config.Plot {
		return
	}

	type RegressionRequest struct {
		X      [][]float64       `json:"X"`
		Y      []float64         `json:"y"`
		Plot   string            `json:"plot,omitempty"`
		Labels map[string]string `json:"labels,omitempty"`
		Layout map[string]any    `json:"layout,omitempty"`
	}

	type RegressionResponse struct {
		HTML string `json:"html"`
	}

	// Check how many features we are using
	numFeatures := len(dataModel.Features)

	//Prepare a request to send to plotting service

	req := RegressionRequest{
		Labels: map[string]string{
			"title":   fmt.Sprintf("%s Regression model", dataModel.Target),
			"x_label": dataModel.Features[0],
			"y_label": dataModel.Target,
		},
		Layout: map[string]any{},
	}

	// Set plot type and prepare data based on the number of features
	if dataContext == nil || len(dataContext.FeatureData) == 0 || len(dataContext.TargetValues) == 0 {
		fmt.Println("Cannot generate plot: no data available")
		return
	}

	// Ensure we have enough data points
	numDataPoints := len(dataContext.TargetValues)
	req.Y = dataContext.TargetValues

	if numFeatures == 1 {
		// 2D plot
		req.Plot = "2d"

		// Extract the feature values
		featureValues := dataContext.FeatureData[dataModel.Features[0]]

		// Prepare X data as 2D slice where each inner slice has one value
		req.X = make([][]float64, numDataPoints)
		for i := range numDataPoints {
			req.X[i] = []float64{featureValues[i]}
		}

	} else {
		// 3D plot
		req.Plot = "3d"

		// Sort our features by absolute coefficient value to find most important ones
		type featureCoefficient struct {
			feature string
			coef    float64
			index   int
		}

		featureImportance := make([]featureCoefficient, numFeatures)

		for i, f := range dataModel.Features {
			featureImportance[i] = featureCoefficient{
				feature: f,
				coef:    math.Abs(dataModel.Coefficients[i]),
				index:   i,
			}
		}

		// Sort by coefficeint magnitude (descending)
		sort.Slice(featureImportance, func(i, j int) bool {
			return featureImportance[i].coef > featureImportance[j].coef
		})

		// Use the two most important features for visualization
		feature1 := featureImportance[0].feature
		feature2 := featureImportance[1].feature

		// Update labels to use most important features
		req.Labels["z_label"] = feature2

		// Get feature values
		feature1Values := dataContext.FeatureData[feature1]
		feature2Values := dataContext.FeatureData[feature2]

		// Prepare X data as a 2D slice with two value per inner slice
		req.X = make([][]float64, numDataPoints)
		for i := range numDataPoints {
			req.X[i] = []float64{feature1Values[i], feature2Values[i]}
		}
	}
}
