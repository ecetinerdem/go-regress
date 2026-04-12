package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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

	// Add predicted values as additional data series
	predictedValues := dataModel.Predict(req.X)
	req.Layout["showPredictions"] = true
	req.Layout["predictions"] = predictedValues

	// Connect to plotting service using the URL from config
	client := &http.Client{}
	jsonData, err := json.Marshal(req)
	if err != nil {
		fmt.Printf("error preparing plot request: %v\n", err)
		return
	}

	// Send request to plotting service
	resp, err := client.Post(
		config.PlotURI,
		"application/json",
		bytes.NewBuffer(jsonData),
	)

	if err != nil {
		fmt.Printf("err connecting to plotting service: %v\n", err)
		return
	}
	defer resp.Body.Close()

	// Process the response
	var plotResp RegressionResponse
	if err := json.NewDecoder(resp.Body).Decode(&plotResp); err != nil {
		fmt.Printf("error decoding plot response: %v\n", err)
		return
	}

	plotFile := filepath.Join(".", "visualizations", "plot.html")
	err = os.WriteFile(plotFile, []byte(plotResp.HTML), 0644)
	if err != nil {
		fmt.Printf("error saving plot html: %v\n", err)
		return
	}

	// Display information about the plot
	fmt.Println("Plot generated successfully!")
	fmt.Printf("View your plot at %s\n", plotFile)

	// Provide some additional information about the plot
	if numFeatures == 1 {
		fmt.Printf("\nThe plot shows the relationship between %s and %s.\n", dataModel.Features[0], dataModel.Target)
		fmt.Printf("The regression line represents the model: %s = %.4f +%.4f x %s", dataModel.Target, dataModel.Intercept, dataModel.Coefficients[0], dataModel.Features[0])
	} else {
		fmt.Printf("\nThe 3D plot shows the relationship between the two most important features (%s and %s) and %s.\n", req.Labels["x"], req.Labels["z"], dataModel.Target)
		fmt.Printf("The regression plane represents the multiple linear regression model with R-squared: %.4f\n", dataModel.RSquared)
	}
	openBrowser(plotFile)
}

func openBrowser(htmlFile string) {
	// Get the absolute path with proper URL
	absPath, err := getAbsolutePath(htmlFile)
	if err != nil {
		fmt.Printf("error getting absolute path: %v\n", err)
		return
	}

	// Convert filepath to URL format
	fileURL := "file://" + absPath

	var cmd *exec.Cmd

	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", fileURL)
	case "darwin":
		cmd = exec.Command("open", fileURL)
	default:
		cmd = exec.Command("xdg-open", fileURL)
	}
	err = cmd.Start()
	if err != nil {
		fmt.Printf("error opening browser: %v\n", err)
	}

}

func getAbsolutePath(filePath string) (string, error) {
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return "", err
	}

	urlPath := filepath.ToSlash(absPath)
	return urlPath, nil
}
