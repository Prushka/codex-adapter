package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

const (
	maxRequestBodyBytes = 128 << 20
	defaultAttempts     = 5
	defaultAttemptDelay = time.Minute
	defaultTimeout      = 30 * time.Minute
)

type cliConfig struct {
	listenAddr string
	apiKey     string
	providers  providerList
	timeout    time.Duration
	attempts   int
	delay      time.Duration
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

	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DisableCompression = true
	client := &http.Client{
		Timeout:   cfg.timeout,
		Transport: transport,
	}

	handler, err := newProxy(proxyConfig{
		Providers: cfg.providers,
		Client:    client,
		Logger:    logger,
		APIKey:    cfg.apiKey,
		Attempts:  cfg.attempts,
		Delay:     cfg.delay,
	})
	if err != nil {
		exitWithError(logger, "failed to create load balancer", zap.Error(err))
	}

	logger.Info("OpenAI-compatible load balancer listening",
		zap.String("listen", cfg.listenAddr),
		zap.Int("providers", len(handler.pool.providers)),
		zap.Int("models", len(handler.pool.modelProviders)),
		zap.Int("attempts", handler.attempts),
		zap.Duration("delay", handler.delay),
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
	flag.StringVar(&cfg.listenAddr, "listen", "127.0.0.1:18081", "local listening address for Chat Completions and Responses requests")
	flag.StringVar(&cfg.apiKey, "api-key", "", "load balancer API key required from downstream requests")
	flag.Var(&cfg.providers, "provider", "upstream provider tuple id,url,key; may be repeated")
	flag.DurationVar(&cfg.timeout, "timeout", defaultTimeout, "upstream request timeout")
	flag.IntVar(&cfg.attempts, "attempts", defaultAttempts, "number of full provider-pool passes before returning failure")
	flag.DurationVar(&cfg.delay, "delay", defaultAttemptDelay, "delay between full provider-pool attempts; first attempt has no delay")
	flag.Parse()
	return cfg
}

func (c cliConfig) validate() error {
	switch {
	case strings.TrimSpace(c.apiKey) == "":
		return errors.New("missing required flag: -api-key")
	case len(c.providers) == 0:
		return errors.New("missing required flag: -provider")
	case c.attempts < 1:
		return errors.New("attempts must be at least 1")
	case c.delay < 0:
		return errors.New("delay must not be negative")
	default:
		seen := map[string]struct{}{}
		for _, provider := range c.providers {
			if _, ok := seen[provider.ID]; ok {
				return fmt.Errorf("duplicate provider id %q", provider.ID)
			}
			seen[provider.ID] = struct{}{}
		}
		return nil
	}
}

type providerList []providerConfig

func (p *providerList) String() string {
	if p == nil {
		return ""
	}
	parts := make([]string, 0, len(*p))
	for _, provider := range *p {
		parts = append(parts, provider.ID+","+provider.ProviderURL+",<redacted>")
	}
	return strings.Join(parts, ";")
}

func (p *providerList) Set(value string) error {
	provider, err := parseProviderConfig(value)
	if err != nil {
		return err
	}
	*p = append(*p, provider)
	return nil
}

func parseProviderConfig(value string) (providerConfig, error) {
	parts := strings.SplitN(value, ",", 3)
	if len(parts) != 3 {
		return providerConfig{}, fmt.Errorf("invalid provider %q: expected id,url,key", value)
	}
	cfg := providerConfig{
		ID:          strings.TrimSpace(parts[0]),
		ProviderURL: strings.TrimSpace(parts[1]),
		APIKey:      strings.TrimSpace(parts[2]),
	}
	switch {
	case cfg.ID == "":
		return providerConfig{}, fmt.Errorf("invalid provider %q: id is required", value)
	case strings.ContainsAny(cfg.ID, " \t\r\n,"):
		return providerConfig{}, fmt.Errorf("invalid provider id %q: id must not contain whitespace or commas", cfg.ID)
	case cfg.ProviderURL == "":
		return providerConfig{}, fmt.Errorf("invalid provider %q: url is required", value)
	case cfg.APIKey == "":
		return providerConfig{}, fmt.Errorf("invalid provider %q: key is required", value)
	default:
		return cfg, nil
	}
}

type proxyConfig struct {
	Providers []providerConfig
	Client    *http.Client
	Logger    *zap.Logger
	APIKey    string
	Attempts  int
	Delay     time.Duration
}

type proxy struct {
	pool     *providerPool
	client   *http.Client
	logger   *zap.Logger
	apiKey   string
	attempts int
	delay    time.Duration
}

func newProxy(cfg proxyConfig) (*proxy, error) {
	client := cfg.Client
	if client == nil {
		return nil, errors.New("http client is required")
	}
	apiKey := strings.TrimSpace(cfg.APIKey)
	if apiKey == "" {
		return nil, errors.New("load balancer API key is required")
	}
	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}
	attempts := cfg.Attempts
	if attempts < 1 {
		attempts = defaultAttempts
	}
	pool, err := newProviderPool(context.Background(), cfg.Providers, client, logger)
	if err != nil {
		return nil, err
	}
	return &proxy{
		pool:     pool,
		client:   client,
		logger:   logger,
		apiKey:   apiKey,
		attempts: attempts,
		delay:    cfg.Delay,
	}, nil
}

type providerConfig struct {
	ID          string
	ProviderURL string
	APIKey      string
}

type provider struct {
	id           string
	chatURL      string
	responsesURL string
	modelsURL    string
	apiKey       string
	models       map[string]struct{}
	busy         int
}

type providerPool struct {
	mu             sync.Mutex
	providers      []provider
	modelProviders map[string][]int
	candidateOrder func([]int) []int
}

type modelListResponse struct {
	Object string          `json:"object"`
	Data   []modelListItem `json:"data"`
}

type modelListItem struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func newProviderPool(ctx context.Context, cfgs []providerConfig, client *http.Client, logger *zap.Logger) (*providerPool, error) {
	if len(cfgs) == 0 {
		return nil, errors.New("at least one upstream provider is required")
	}
	if logger == nil {
		logger = zap.NewNop()
	}

	providers := make([]provider, 0, len(cfgs))
	modelProviders := map[string][]int{}
	for i, cfg := range cfgs {
		chatURL, err := normalizeChatCompletionsURL(cfg.ProviderURL)
		if err != nil {
			return nil, fmt.Errorf("invalid provider %q URL %q: %w", cfg.ID, cfg.ProviderURL, err)
		}
		responsesURL, err := normalizeResponsesURL(cfg.ProviderURL)
		if err != nil {
			return nil, fmt.Errorf("invalid provider %q responses URL %q: %w", cfg.ID, cfg.ProviderURL, err)
		}
		modelsURL, err := normalizeModelsURL(cfg.ProviderURL)
		if err != nil {
			return nil, fmt.Errorf("invalid provider %q models URL %q: %w", cfg.ID, cfg.ProviderURL, err)
		}
		models, err := fetchProviderModels(ctx, client, modelsURL, cfg.APIKey)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch models for provider %q (%s): %w", cfg.ID, modelsURL, err)
		}
		if len(models) == 0 {
			return nil, fmt.Errorf("provider %q (%s) returned no models", cfg.ID, modelsURL)
		}
		providers = append(providers, provider{
			id:           cfg.ID,
			chatURL:      chatURL,
			responsesURL: responsesURL,
			modelsURL:    modelsURL,
			apiKey:       cfg.APIKey,
			models:       models,
		})
		for model := range models {
			modelProviders[model] = append(modelProviders[model], i)
		}
		modelIDs := sortedModelIDs(models)
		logger.Info("loaded provider models",
			zap.Int("provider_index", i),
			zap.String("provider_id", cfg.ID),
			zap.String("models_url", modelsURL),
			zap.Int("models", len(models)),
			zap.Strings("model_ids", modelIDs),
		)
	}

	return &providerPool{
		providers:      providers,
		modelProviders: modelProviders,
		candidateOrder: randomCandidateOrder,
	}, nil
}

func (p *providerPool) provider(index int) provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.providers[index]
}

func (p *providerPool) candidatesForModel(model string) ([]int, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return nil, errors.New("request must include a model")
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	candidates := append([]int(nil), p.modelProviders[model]...)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("model %q is not available on any configured provider", model)
	}
	return candidates, nil
}

func (p *providerPool) allModels() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	models := make([]string, 0, len(p.modelProviders))
	for model := range p.modelProviders {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}

func (p *providerPool) providerModelMap() map[string][]string {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make(map[string][]string, len(p.providers))
	for _, provider := range p.providers {
		out[provider.id] = sortedModelIDs(provider.models)
	}
	return out
}

func sortedModelIDs(models map[string]struct{}) []string {
	modelIDs := make([]string, 0, len(models))
	for model := range models {
		modelIDs = append(modelIDs, model)
	}
	sort.Strings(modelIDs)
	return modelIDs
}

func (p *providerPool) acquire(orderedCandidates []int, tried map[int]struct{}) (int, bool, func(), error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(orderedCandidates) == 0 {
		return 0, false, nil, errors.New("no provider candidates")
	}

	for _, providerIndex := range orderedCandidates {
		if _, ok := tried[providerIndex]; ok {
			continue
		}
		if p.providers[providerIndex].busy == 0 {
			p.providers[providerIndex].busy++
			return providerIndex, false, p.releaseFunc(providerIndex), nil
		}
	}

	for _, providerIndex := range orderedCandidates {
		if _, ok := tried[providerIndex]; ok {
			continue
		}
		p.providers[providerIndex].busy++
		return providerIndex, true, p.releaseFunc(providerIndex), nil
	}

	return 0, false, nil, errors.New("no untried provider candidates")
}

func (p *providerPool) orderedCandidates(candidates []int) []int {
	if p.candidateOrder == nil {
		return append([]int(nil), candidates...)
	}
	return p.candidateOrder(candidates)
}

func randomCandidateOrder(candidates []int) []int {
	ordered := append([]int(nil), candidates...)
	if len(ordered) > 1 {
		rand.Shuffle(len(ordered), func(i, j int) {
			ordered[i], ordered[j] = ordered[j], ordered[i]
		})
	}
	return ordered
}

func (p *providerPool) releaseFunc(index int) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			p.mu.Lock()
			defer p.mu.Unlock()
			if index >= 0 && index < len(p.providers) && p.providers[index].busy > 0 {
				p.providers[index].busy--
			}
		})
	}
}

func requestedModel(body []byte) (string, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return "", fmt.Errorf("invalid JSON request body: %w", err)
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		return "", errors.New("chat completions request must include a model")
	}
	return model, nil
}

type modelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

func fetchProviderModels(ctx context.Context, client *http.Client, modelsURL, apiKey string) (map[string]struct{}, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", authorizationHeader(apiKey))

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer closeBody(zap.NewNop(), resp.Body)

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxRequestBodyBytes))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("models endpoint returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var payload modelsResponse
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	models := map[string]struct{}{}
	for _, model := range payload.Data {
		id := strings.TrimSpace(model.ID)
		if id != "" {
			models[id] = struct{}{}
		}
	}
	return models, nil
}

func (p *proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimRight(r.URL.Path, "/")
	if path == "" {
		path = "/"
	}
	switch path {
	case "/healthz":
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	case "/models", "/v1/models":
		p.handleModels(w, r)
	case "/models/map", "/v1/models/map":
		p.handleModelMap(w, r)
	case "/chat/completions", "/v1/chat/completions":
		p.handleAPIRequest(w, r, endpointChatCompletions)
	case "/responses", "/v1/responses":
		p.handleAPIRequest(w, r, endpointResponses)
	default:
		http.NotFound(w, r)
	}
}

func (p *proxy) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !p.authorizeDownstream(w, r, "models") {
		return
	}

	models := p.pool.allModels()
	data := make([]modelListItem, 0, len(models))
	now := time.Now().Unix()
	for _, model := range models {
		data = append(data, modelListItem{
			ID:      model,
			Object:  "model",
			Created: now,
			OwnedBy: "load-balancer",
		})
	}
	writeJSON(w, modelListResponse{
		Object: "list",
		Data:   data,
	})
}

func (p *proxy) handleModelMap(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !p.authorizeDownstream(w, r, "models map") {
		return
	}
	writeJSON(w, p.pool.providerModelMap())
}

func (p *proxy) authorizeDownstream(w http.ResponseWriter, r *http.Request, api string) bool {
	if validAuthorization(r.Header.Get("Authorization"), p.apiKey) {
		return true
	}
	p.logger.Warn("unauthorized downstream request",
		zap.String("api", api),
		zap.String("path", r.URL.Path),
	)
	w.Header().Set("WWW-Authenticate", "Bearer")
	writeJSONError(w, http.StatusUnauthorized, "invalid_api_key", "invalid or missing API key")
	return false
}

type upstreamEndpoint string

const (
	endpointChatCompletions upstreamEndpoint = "chat completions"
	endpointResponses       upstreamEndpoint = "responses"
)

func (e upstreamEndpoint) url(provider provider) string {
	switch e {
	case endpointResponses:
		return provider.responsesURL
	default:
		return provider.chatURL
	}
}

func (p *proxy) handleAPIRequest(w http.ResponseWriter, r *http.Request, endpoint upstreamEndpoint) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !p.authorizeDownstream(w, r, string(endpoint)) {
		return
	}
	requestStarted := time.Now()
	body, err := readRequestBody(r, maxRequestBodyBytes)
	if err != nil {
		p.logger.Warn("failed to read downstream request",
			zap.String("api", string(endpoint)),
			zap.String("path", r.URL.Path),
			zap.Error(err),
		)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	model, err := requestedModel(body)
	if err != nil {
		p.logger.Warn("invalid downstream request",
			zap.String("api", string(endpoint)),
			zap.String("path", r.URL.Path),
			zap.Int("request_bytes", len(body)),
			zap.Error(err),
		)
		writeJSONError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	candidates, err := p.pool.candidatesForModel(model)
	if err != nil {
		p.logger.Warn("requested model is not available",
			zap.String("api", string(endpoint)),
			zap.String("path", r.URL.Path),
			zap.String("model", model),
			zap.Int("request_bytes", len(body)),
			zap.Error(err),
		)
		writeJSONError(w, http.StatusBadRequest, "model_not_available", err.Error())
		return
	}
	requestFields := []zap.Field{
		zap.String("api", string(endpoint)),
		zap.String("path", r.URL.Path),
		zap.String("model", model),
		zap.Int("request_bytes", len(body)),
		zap.Int("candidate_providers", len(candidates)),
	}

	var lastHTTP *bufferedResponse
	var lastErr error
	attempt := 0
	attemptPasses := p.attempts
	if attemptPasses < 1 {
		attemptPasses = defaultAttempts
	}
	for pass := 0; pass < attemptPasses; pass++ {
		if pass > 0 && p.delay > 0 {
			if err := sleepWithContext(r.Context(), p.delay); err != nil {
				p.logger.Info("downstream canceled request during retry delay",
					appendRequestFields(requestFields,
						zap.Int("attempt_pass", pass+1),
						zap.Duration("elapsed", time.Since(requestStarted)),
						zap.Error(err),
					)...,
				)
				return
			}
		}
		orderedCandidates := p.pool.orderedCandidates(candidates)
		tried := make(map[int]struct{}, len(orderedCandidates))
		for passAttempt := 0; passAttempt < len(orderedCandidates); passAttempt++ {
			providerIndex, busyFallback, release, err := p.pool.acquire(orderedCandidates, tried)
			if err != nil {
				break
			}
			tried[providerIndex] = struct{}{}
			attempt++

			attemptStarted := time.Now()
			resp, err := p.postUpstream(r, body, providerIndex, endpoint)
			if err != nil {
				if r.Context().Err() != nil {
					p.logger.Info("downstream canceled request",
						appendRequestFields(requestFields,
							zap.Int("attempt", attempt),
							zap.Int("attempt_pass", pass+1),
							zap.Int("provider_index", providerIndex),
							zap.Bool("busy_fallback", busyFallback),
							zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
							zap.Duration("elapsed", time.Since(requestStarted)),
							zap.Error(err),
						)...,
					)
					release()
					return
				}
				provider := p.pool.provider(providerIndex)
				lastErr = err
				release()
				p.logger.Warn("upstream request failed",
					appendRequestFields(requestFields,
						zap.Int("attempt", attempt),
						zap.Int("attempt_pass", pass+1),
						zap.Int("provider_index", providerIndex),
						zap.String("provider_id", provider.id),
						zap.Bool("busy_fallback", busyFallback),
						zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
						zap.Error(err),
					)...,
				)
				continue
			}

			provider := p.pool.provider(providerIndex)
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				defer release()
				defer closeBody(p.logger, resp.Body)
				copyResponseHeaders(w.Header(), resp.Header)
				w.WriteHeader(resp.StatusCode)
				responseBytes, err := copyResponseBody(w, resp.Body)
				if err != nil {
					p.logger.Warn("failed to copy upstream response",
						appendRequestFields(requestFields,
							zap.Int("attempt", attempt),
							zap.Int("attempt_pass", pass+1),
							zap.Int("provider_index", providerIndex),
							zap.String("provider_id", provider.id),
							zap.Bool("busy_fallback", busyFallback),
							zap.Int("status", resp.StatusCode),
							zap.Int64("response_bytes", responseBytes),
							zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
							zap.Duration("elapsed", time.Since(requestStarted)),
							zap.Error(err),
						)...,
					)
					return
				}
				p.logger.Info("upstream request succeeded",
					appendRequestFields(requestFields,
						zap.Int("attempt", attempt),
						zap.Int("attempt_pass", pass+1),
						zap.Int("provider_index", providerIndex),
						zap.String("provider_id", provider.id),
						zap.Bool("busy_fallback", busyFallback),
						zap.Int("status", resp.StatusCode),
						zap.Int64("response_bytes", responseBytes),
						zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
						zap.Duration("elapsed", time.Since(requestStarted)),
					)...,
				)
				return
			}

			failure, readErr := bufferResponse(resp)
			release()
			if readErr != nil {
				lastErr = readErr
				p.logger.Warn("failed to read upstream error response",
					appendRequestFields(requestFields,
						zap.Int("attempt", attempt),
						zap.Int("attempt_pass", pass+1),
						zap.Int("provider_index", providerIndex),
						zap.String("provider_id", provider.id),
						zap.Bool("busy_fallback", busyFallback),
						zap.Int("status", resp.StatusCode),
						zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
						zap.Error(readErr),
					)...,
				)
			} else {
				lastHTTP = failure
				lastErr = nil
				p.logger.Warn("upstream returned failure",
					appendRequestFields(requestFields,
						zap.Int("attempt", attempt),
						zap.Int("attempt_pass", pass+1),
						zap.Int("provider_index", providerIndex),
						zap.String("provider_id", provider.id),
						zap.Bool("busy_fallback", busyFallback),
						zap.Int("status", resp.StatusCode),
						zap.Int("response_bytes", len(failure.body)),
						zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
					)...,
				)
			}
		}
	}

	if lastHTTP != nil {
		p.logger.Warn("returning last upstream failure",
			appendRequestFields(requestFields,
				zap.Int("attempts", attempt),
				zap.Int("status", lastHTTP.statusCode),
				zap.Int("response_bytes", len(lastHTTP.body)),
				zap.Duration("elapsed", time.Since(requestStarted)),
			)...,
		)
		copyResponseHeaders(w.Header(), lastHTTP.header)
		w.WriteHeader(lastHTTP.statusCode)
		_, _ = w.Write(lastHTTP.body)
		return
	}
	msg := "all matching upstream providers failed"
	if lastErr != nil {
		msg = lastErr.Error()
	}
	p.logger.Warn("all matching upstream providers failed",
		appendRequestFields(requestFields,
			zap.Int("attempts", attempt),
			zap.Duration("elapsed", time.Since(requestStarted)),
			zap.Error(lastErr),
		)...,
	)
	writeJSONError(w, http.StatusBadGateway, "upstream_request_error", msg)
}

func appendRequestFields(base []zap.Field, fields ...zap.Field) []zap.Field {
	out := make([]zap.Field, 0, len(base)+len(fields))
	out = append(out, base...)
	out = append(out, fields...)
	return out
}

func sleepWithContext(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *proxy) postUpstream(inbound *http.Request, body []byte, providerIndex int, endpoint upstreamEndpoint) (*http.Response, error) {
	provider := p.pool.provider(providerIndex)
	targetURL := urlWithRawQuery(endpoint.url(provider), inbound.URL.RawQuery)
	req, err := http.NewRequestWithContext(inbound.Context(), http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	copyForwardHeaders(req.Header, inbound.Header)
	req.Header.Set("Authorization", authorizationHeader(provider.apiKey))
	return p.client.Do(req)
}

type bufferedResponse struct {
	statusCode int
	header     http.Header
	body       []byte
}

func bufferResponse(resp *http.Response) (*bufferedResponse, error) {
	defer closeBody(zap.NewNop(), resp.Body)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return &bufferedResponse{
		statusCode: resp.StatusCode,
		header:     resp.Header.Clone(),
		body:       body,
	}, nil
}

func readRequestBody(r *http.Request, limit int64) ([]byte, error) {
	defer closeBody(zap.NewNop(), r.Body)
	limited := &io.LimitedReader{R: r.Body, N: limit + 1}
	body, err := io.ReadAll(limited)
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, fmt.Errorf("request body exceeds %d bytes", limit)
	}
	return body, nil
}

func writeJSONError(w http.ResponseWriter, status int, code string, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = fmt.Fprintf(w, `{"error":{"type":"server_error","code":%q,"message":%q}}`, code, message)
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func normalizeChatCompletionsURL(raw string) (string, error) {
	return normalizeProviderEndpointURL(raw, "/chat/completions")
}

func normalizeResponsesURL(raw string) (string, error) {
	return normalizeProviderEndpointURL(raw, "/responses")
}

func normalizeProviderEndpointURL(raw string, endpointPath string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("provider URL must include scheme and host: %s", raw)
	}
	path := strings.TrimRight(u.Path, "/")
	if basePath, ok := stripKnownProviderEndpoint(path); ok {
		u.Path = joinEndpointPath(basePath, endpointPath)
		return u.String(), nil
	}
	switch {
	case path == "":
		u.Path = "/v1" + endpointPath
	case strings.HasSuffix(path, "/v1"):
		u.Path = path + endpointPath
	default:
		u.Path = path + endpointPath
	}
	return u.String(), nil
}

func normalizeModelsURL(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("provider URL must include scheme and host: %s", raw)
	}
	path := strings.TrimRight(u.Path, "/")
	if basePath, ok := stripKnownProviderEndpoint(path); ok {
		u.Path = joinEndpointPath(basePath, "/models")
		u.RawQuery = ""
		return u.String(), nil
	}
	switch {
	case path == "":
		u.Path = "/v1/models"
	case strings.HasSuffix(path, "/v1"):
		u.Path = path + "/models"
	default:
		u.Path = path + "/models"
	}
	u.RawQuery = ""
	return u.String(), nil
}

func stripKnownProviderEndpoint(path string) (string, bool) {
	for _, suffix := range []string{"/chat/completions", "/responses", "/models"} {
		if strings.HasSuffix(path, suffix) {
			return strings.TrimRight(strings.TrimSuffix(path, suffix), "/"), true
		}
	}
	return "", false
}

func joinEndpointPath(basePath string, endpointPath string) string {
	if basePath == "" {
		return endpointPath
	}
	return basePath + endpointPath
}

func urlWithRawQuery(rawURL, rawQuery string) string {
	if rawQuery == "" {
		return rawURL
	}
	u, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}
	if u.RawQuery == "" {
		u.RawQuery = rawQuery
	} else {
		u.RawQuery += "&" + rawQuery
	}
	return u.String()
}

func authorizationHeader(apiKey string) string {
	apiKey = strings.TrimSpace(apiKey)
	if strings.Contains(apiKey, " ") {
		return apiKey
	}
	return "Bearer " + apiKey
}

func validAuthorization(got string, apiKey string) bool {
	want := authorizationHeader(apiKey)
	return subtle.ConstantTimeCompare([]byte(strings.TrimSpace(got)), []byte(want)) == 1
}

func copyForwardHeaders(dst, src http.Header) {
	for k, values := range src {
		lower := strings.ToLower(k)
		if isHopByHopHeader(lower) || lower == "authorization" || lower == "content-length" {
			continue
		}
		for _, v := range values {
			dst.Add(k, v)
		}
	}
}

func copyResponseHeaders(dst, src http.Header) {
	for k, values := range src {
		if isHopByHopHeader(strings.ToLower(k)) {
			continue
		}
		for _, v := range values {
			dst.Add(k, v)
		}
	}
}

func isHopByHopHeader(lower string) bool {
	switch lower {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
		"te", "trailer", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}

func copyResponseBody(w http.ResponseWriter, body io.Reader) (int64, error) {
	if flusher, ok := w.(http.Flusher); ok {
		buf := make([]byte, 32<<10)
		var copied int64
		for {
			n, readErr := body.Read(buf)
			if n > 0 {
				written, writeErr := w.Write(buf[:n])
				copied += int64(written)
				if writeErr != nil {
					return copied, writeErr
				}
				flusher.Flush()
			}
			if readErr != nil {
				if errors.Is(readErr, io.EOF) {
					return copied, nil
				}
				return copied, readErr
			}
		}
	}
	return io.Copy(w, body)
}

func closeBody(logger *zap.Logger, body io.Closer) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		logger.Warn("failed to close body", zap.Error(err))
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
