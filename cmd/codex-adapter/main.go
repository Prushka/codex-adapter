package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	codexadapter "codex-adapter"
	"go.uber.org/zap"
)

type cliConfig struct {
	listenAddr      string
	providerURL     string
	model           string
	reasoningEffort string
	apiKey          string
	apiKeyEnv       string
	debug           bool
	debugDir        string
	timeout         time.Duration
}

func main() {
	logger, err := newLogger()
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = logger.Sync() }()

	cfg := parseFlags()
	if err := cfg.validate(); err != nil {
		exitWithError(logger, err.Error())
	}

	upstreamAPIKey, err := cfg.resolveAPIKey()
	if err != nil {
		exitWithError(logger, err.Error(), zap.String("env", cfg.apiKeyEnv))
	}

	var recorder *codexadapter.DebugRecorder
	if cfg.debug {
		recorder, err = codexadapter.NewDebugRecorder(cfg.debugDir)
		if err != nil {
			exitWithError(logger, "failed to create debug recorder", zap.Error(err))
		}
	}

	handler, err := codexadapter.NewAdapter(codexadapter.AdapterConfig{
		ProviderURL:     cfg.providerURL,
		Model:           cfg.model,
		ReasoningEffort: cfg.reasoningEffort,
		APIKey:          upstreamAPIKey,
		Debug:           recorder,
		HTTPClient:      &http.Client{Timeout: cfg.timeout},
		Logger:          logger,
	})
	if err != nil {
		exitWithError(logger, "failed to create adapter", zap.Error(err))
	}

	logger.Info("codex-adapter listening",
		zap.String("listen", cfg.listenAddr),
		zap.String("upstream_chat_completions_url", handler.ChatCompletionsURL()),
	)

	server := &http.Server{
		Addr:              cfg.listenAddr,
		Handler:           handler,
		ReadHeaderTimeout: 30 * time.Second,
	}
	if err := server.ListenAndServe(); err != nil {
		exitWithError(logger, "server stopped", zap.Error(err))
	}
}

func parseFlags() cliConfig {
	var cfg cliConfig
	flag.StringVar(&cfg.listenAddr, "listen", "127.0.0.1:8080", "local listening address for Responses API requests")
	flag.StringVar(&cfg.providerURL, "provider-url", "", "OpenAI-compatible upstream provider base URL or /v1 URL")
	flag.StringVar(&cfg.model, "model", "", "upstream chat_completions model to force into every request")
	flag.StringVar(&cfg.reasoningEffort, "reasoning-effort", "medium", "reasoning_effort value to force into every upstream request")
	flag.StringVar(&cfg.apiKey, "api-key", "", "upstream provider API key; overrides any Authorization header sent by Codex")
	flag.StringVar(&cfg.apiKeyEnv, "api-key-env", "", "environment variable containing the upstream provider API key")
	flag.BoolVar(&cfg.debug, "debug", false, "save all translated requests and responses as ordered JSON files")
	flag.StringVar(&cfg.debugDir, "debug-dir", "debug", "directory for debug JSON files")
	flag.DurationVar(&cfg.timeout, "timeout", 10*time.Minute, "upstream request timeout")
	flag.Parse()
	return cfg
}

func (c cliConfig) validate() error {
	switch {
	case c.providerURL == "":
		return fmt.Errorf("missing required flag: -provider-url")
	case c.model == "":
		return fmt.Errorf("missing required flag: -model")
	case c.reasoningEffort == "":
		return fmt.Errorf("missing required flag: -reasoning-effort")
	case c.apiKey != "" && c.apiKeyEnv != "":
		return fmt.Errorf("only one API key source may be set: -api-key or -api-key-env")
	default:
		return nil
	}
}

func (c cliConfig) resolveAPIKey() (string, error) {
	if c.apiKeyEnv == "" {
		return c.apiKey, nil
	}
	value := strings.TrimSpace(os.Getenv(c.apiKeyEnv))
	if value == "" {
		return "", fmt.Errorf("API key environment variable is unset or empty")
	}
	return value, nil
}

func newLogger() (*zap.Logger, error) {
	cfg := zap.NewProductionConfig()
	cfg.DisableStacktrace = true
	return cfg.Build()
}

func exitWithError(logger *zap.Logger, msg string, fields ...zap.Field) {
	logger.Error(msg, fields...)
	_ = logger.Sync()
	os.Exit(1)
}
