package main

import (
	"flag"
	"log"
	"net/http"
	"time"
)

func main() {
	listenAddr := flag.String("listen", "127.0.0.1:8080", "local listening address for Responses API requests")
	providerURL := flag.String("provider-url", "", "OpenAI-compatible upstream provider base URL or /v1 URL")
	model := flag.String("model", "", "upstream chat_completions model to force into every request")
	reasoningEffort := flag.String("reasoning-effort", "medium", "reasoning_effort value to force into every upstream request")
	debug := flag.Bool("debug", false, "save all translated requests and responses as ordered JSON files")
	debugDir := flag.String("debug-dir", "debug", "directory for debug JSON files")
	timeout := flag.Duration("timeout", 10*time.Minute, "upstream request timeout")
	flag.Parse()

	if *providerURL == "" {
		log.Fatal("-provider-url is required")
	}
	if *model == "" {
		log.Fatal("-model is required")
	}
	if *reasoningEffort == "" {
		log.Fatal("-reasoning-effort is required")
	}

	var recorder *DebugRecorder
	if *debug {
		var err error
		recorder, err = NewDebugRecorder(*debugDir)
		if err != nil {
			log.Fatalf("failed to create debug recorder: %v", err)
		}
	}

	adapter, err := NewAdapter(AdapterConfig{
		ProviderURL:     *providerURL,
		Model:           *model,
		ReasoningEffort: *reasoningEffort,
		Debug:           recorder,
		HTTPClient:      &http.Client{Timeout: *timeout},
	})
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("codex-adapter listening on %s and forwarding to %s", *listenAddr, adapter.ChatCompletionsURL())
	server := &http.Server{
		Addr:              *listenAddr,
		Handler:           adapter,
		ReadHeaderTimeout: 30 * time.Second,
	}
	log.Fatal(server.ListenAndServe())
}
