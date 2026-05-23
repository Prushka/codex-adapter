package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"time"

	"go.uber.org/zap"
)

func main() {
	logger, err := newLogger()
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = logger.Sync() }()

	listenAddr := flag.String("listen", "127.0.0.1:8080", "local listening address for Responses API requests")
	providerURL := flag.String("provider-url", "", "OpenAI-compatible upstream provider base URL or /v1 URL")
	model := flag.String("model", "", "upstream chat_completions model to force into every request")
	reasoningEffort := flag.String("reasoning-effort", "medium", "reasoning_effort value to force into every upstream request")
	debug := flag.Bool("debug", false, "save all translated requests and responses as ordered JSON files")
	debugDir := flag.String("debug-dir", "debug", "directory for debug JSON files")
	timeout := flag.Duration("timeout", 10*time.Minute, "upstream request timeout")
	flag.Parse()

	if *providerURL == "" {
		exitWithError(logger, "missing required flag", zap.String("flag", "-provider-url"))
	}
	if *model == "" {
		exitWithError(logger, "missing required flag", zap.String("flag", "-model"))
	}
	if *reasoningEffort == "" {
		exitWithError(logger, "missing required flag", zap.String("flag", "-reasoning-effort"))
	}

	var recorder *DebugRecorder
	if *debug {
		var err error
		recorder, err = NewDebugRecorder(*debugDir)
		if err != nil {
			exitWithError(logger, "failed to create debug recorder", zap.Error(err))
		}
	}

	adapter, err := NewAdapter(AdapterConfig{
		ProviderURL:     *providerURL,
		Model:           *model,
		ReasoningEffort: *reasoningEffort,
		Debug:           recorder,
		HTTPClient:      &http.Client{Timeout: *timeout},
		Logger:          logger,
	})
	if err != nil {
		exitWithError(logger, "failed to create adapter", zap.Error(err))
	}

	logger.Info("codex-adapter listening",
		zap.String("listen", *listenAddr),
		zap.String("upstream_chat_completions_url", adapter.ChatCompletionsURL()),
	)
	server := &http.Server{
		Addr:              *listenAddr,
		Handler:           adapter,
		ReadHeaderTimeout: 30 * time.Second,
	}
	if err := server.ListenAndServe(); err != nil {
		exitWithError(logger, "server stopped", zap.Error(err))
	}
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
