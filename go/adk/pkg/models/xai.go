package models

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/kagent-dev/kagent/go/core/pkg/env"
	"github.com/kagent-dev/kagent/go/pkg/logging"
)

// DefaultXAIBaseURL is the default endpoint for the xAI API.
// The XAI_API_BASE environment variable and XAIConfig.BaseUrl both override it.
const DefaultXAIBaseURL = "https://api.x.ai/v1"

// XAIConfig holds xAI configuration. xAI speaks the OpenAI wire protocol, so
// the runtime reuses the OpenAI SDK client and honors the same generation
// parameters (temperature, top_p, max_tokens, timeout).
type XAIConfig struct {
	TransportConfig
	Model       string
	BaseUrl     string
	MaxTokens   *int64
	Temperature *float64
	TopP        *float64
}

// XAIModel implements model.LLM (see xai_adk.go) for xAI.
// It wraps an OpenAIModel because xAI exposes an OpenAI-compatible API
// (POST {base_url}/chat/completions with a Bearer token).
type XAIModel struct {
	Config *XAIConfig
	inner  *OpenAIModel
	Logger *slog.Logger
}

// NewXAIModel creates a new xAI model instance.
// It reads XAI_API_KEY unless APIKeyPassthrough is enabled; base URL falls
// back to XAI_API_BASE then DefaultXAIBaseURL.
func NewXAIModel(ctx context.Context, config *XAIConfig) (*XAIModel, error) {
	logger := logging.FromContext(ctx)
	apiKey := "passthrough" // placeholder; real auth set per-request by transport
	if !config.APIKeyPassthrough {
		apiKey = env.XAIAPIKey.Get()
		if apiKey == "" {
			return nil, fmt.Errorf("XAI_API_KEY environment variable is not set")
		}
	}

	baseURL := config.BaseUrl
	if baseURL == "" {
		baseURL = env.XAIAPIBase.Get()
	}
	if baseURL == "" {
		baseURL = DefaultXAIBaseURL
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	httpClient, err := BuildHTTPClient(config.TransportConfig)
	if err != nil {
		return nil, err
	}
	if len(config.Headers) > 0 {
		logger.InfoContext(ctx, "setting default headers for xAI client", "headers_count", len(config.Headers))
	}
	opts = append(opts, option.WithHTTPClient(httpClient))

	client := openai.NewClient(opts...)
	logger.InfoContext(ctx, "initialized xAI model", "model", config.Model, "base_url", baseURL)

	inner := &OpenAIModel{
		Config: &OpenAIConfig{
			TransportConfig: config.TransportConfig,
			Model:           config.Model,
			BaseUrl:         baseURL,
			MaxTokens:       config.MaxTokens,
			Temperature:     config.Temperature,
			TopP:            config.TopP,
		},
		Client:  client,
		IsAzure: false,
		Logger:  logger,
	}
	return &XAIModel{
		Config: config,
		inner:  inner,
		Logger: logger,
	}, nil
}
