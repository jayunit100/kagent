package models

import (
	"testing"
)

func TestNewXAIModel(t *testing.T) {
	tests := []struct {
		name        string
		envAPIKey   string
		envAPIBase  string
		config      *XAIConfig
		wantErr     bool
		wantBaseURL string
	}{
		{
			name:      "missing API key without passthrough returns error",
			envAPIKey: "",
			config:    &XAIConfig{Model: "grok-4.7"},
			wantErr:   true,
		},
		{
			name:        "default base URL when nothing set",
			envAPIKey:   "test-key",
			config:      &XAIConfig{Model: "grok-4.7"},
			wantBaseURL: DefaultXAIBaseURL,
		},
		{
			name:        "XAI_API_BASE env overrides default",
			envAPIKey:   "test-key",
			envAPIBase:  "https://gateway.example.com/xai/v1",
			config:      &XAIConfig{Model: "grok-4.7"},
			wantBaseURL: "https://gateway.example.com/xai/v1",
		},
		{
			name:       "config BaseUrl wins over env",
			envAPIKey:  "test-key",
			envAPIBase: "https://from-env.example.com/v1",
			config: &XAIConfig{
				Model:   "grok-4.7",
				BaseUrl: "https://from-config.example.com/v1",
			},
			wantBaseURL: "https://from-config.example.com/v1",
		},
		{
			name:      "passthrough skips XAI_API_KEY check",
			envAPIKey: "",
			config: &XAIConfig{
				TransportConfig: TransportConfig{APIKeyPassthrough: true},
				Model:           "grok-4.3",
			},
			wantBaseURL: DefaultXAIBaseURL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("XAI_API_KEY", tt.envAPIKey)
			t.Setenv("XAI_API_BASE", tt.envAPIBase)

			m, err := NewXAIModel(t.Context(), tt.config)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if m == nil {
				t.Fatal("expected non-nil XAIModel")
			}
			if m.Name() != tt.config.Model {
				t.Errorf("Name() = %q, want %q", m.Name(), tt.config.Model)
			}
			if m.inner == nil {
				t.Fatal("expected inner OpenAIModel to be initialized")
			}
			if m.inner.Config.BaseUrl != tt.wantBaseURL {
				t.Errorf("inner BaseUrl = %q, want %q", m.inner.Config.BaseUrl, tt.wantBaseURL)
			}
			if m.inner.Config.Model != tt.config.Model {
				t.Errorf("inner Model = %q, want %q", m.inner.Config.Model, tt.config.Model)
			}
		})
	}
}

func TestXAIModel_NamePropagatesModelNotProvider(t *testing.T) {
	m := &XAIModel{
		Config: &XAIConfig{Model: "grok-4.7"},
	}
	if got := m.Name(); got != "grok-4.7" {
		t.Errorf("Name() = %q, want %q", got, "grok-4.7")
	}
	if m.Name() == "xai" {
		t.Error("Name() returns provider name 'xai' instead of model name — this causes 404 from the xAI API")
	}
}
