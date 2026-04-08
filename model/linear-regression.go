package model

import "time"

type LinearRegression struct {
	Coefficients    []float64 `json:"coefficients,omitempty"`
	Intercept       float64   `json:"intercept,omitempty"`
	Features        string    `json:"features,omitempty"`
	Target          string    `json:"target,omitempty"`
	RSquared        float64   `json:"rsquared,omitempty"`
	FeatureMeans    []float64 `json:"feature_means,omitempty"`
	FeaturesStdDevs []float64 `json:"feature_std_devs,omitempty"`
	IsNormalized    bool      `json:"is_normalized,omitempty"`
	SavedAt         time.Time `json:"-"`
	Description     string    `json:"description,omitempty"`
	NumSamples      int       `json:"num_samples,omitempty"`
	Version         string    `json:"version,omitempty"`
}
