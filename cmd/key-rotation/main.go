package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

const maxRequestBodyBytes = 128 << 20

type cliConfig struct {
	listenAddr string
	providers  providerList
	timeout    time.Duration
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
	})
	if err != nil {
		exitWithError(logger, "failed to create load balancer", zap.Error(err))
	}

	logger.Info("chat completions load balancer listening",
		zap.String("listen", cfg.listenAddr),
		zap.Int("providers", len(handler.pool.providers)),
		zap.Int("models", len(handler.pool.modelProviders)),
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
	flag.StringVar(&cfg.listenAddr, "listen", "127.0.0.1:18081", "local listening address for Chat Completions requests")
	flag.Var(&cfg.providers, "provider", "upstream provider tuple id,url,key; may be repeated")
	flag.DurationVar(&cfg.timeout, "timeout", 10*time.Minute, "upstream request timeout")
	flag.Parse()
	return cfg
}

func (c cliConfig) validate() error {
	switch {
	case len(c.providers) == 0:
		return errors.New("missing required flag: -provider")
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
}

type proxy struct {
	pool   *providerPool
	client *http.Client
	logger *zap.Logger
}

func newProxy(cfg proxyConfig) (*proxy, error) {
	client := cfg.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}
	pool, err := newProviderPool(context.Background(), cfg.Providers, client, logger)
	if err != nil {
		return nil, err
	}
	return &proxy{
		pool:   pool,
		client: client,
		logger: logger,
	}, nil
}

type providerConfig struct {
	ID          string
	ProviderURL string
	APIKey      string
}

type provider struct {
	id        string
	chatURL   string
	modelsURL string
	apiKey    string
	models    map[string]struct{}
	busy      int
}

type providerPool struct {
	mu             sync.Mutex
	providers      []provider
	modelProviders map[string][]int
	current        int
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
			id:        cfg.ID,
			chatURL:   chatURL,
			modelsURL: modelsURL,
			apiKey:    cfg.APIKey,
			models:    models,
		})
		for model := range models {
			modelProviders[model] = append(modelProviders[model], i)
		}
		logger.Info("loaded provider models",
			zap.Int("provider_index", i),
			zap.String("provider_id", cfg.ID),
			zap.String("models_url", modelsURL),
			zap.Int("models", len(models)),
		)
	}

	return &providerPool{
		providers:      providers,
		modelProviders: modelProviders,
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
		return nil, errors.New("chat completions request must include a model")
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	candidates := append([]int(nil), p.modelProviders[model]...)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("model %q is not available on any configured provider", model)
	}
	return candidates, nil
}

func (p *providerPool) acquire(candidates []int, tried map[int]struct{}) (int, bool, func(), error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(candidates) == 0 {
		return 0, false, nil, errors.New("no provider candidates")
	}

	ordered := rotateCandidates(candidates, p.current)
	for _, providerIndex := range ordered {
		if _, ok := tried[providerIndex]; ok {
			continue
		}
		if p.providers[providerIndex].busy == 0 {
			p.providers[providerIndex].busy++
			p.current = nextProviderPosition(candidates, providerIndex)
			return providerIndex, false, p.releaseFunc(providerIndex), nil
		}
	}

	for _, providerIndex := range candidates {
		if _, ok := tried[providerIndex]; ok {
			continue
		}
		p.providers[providerIndex].busy++
		p.current = nextProviderPosition(candidates, providerIndex)
		return providerIndex, true, p.releaseFunc(providerIndex), nil
	}

	return 0, false, nil, errors.New("no untried provider candidates")
}

func rotateCandidates(candidates []int, current int) []int {
	if len(candidates) < 2 {
		return append([]int(nil), candidates...)
	}
	start := 0
	for i, candidate := range candidates {
		if candidate == current {
			start = i
			break
		}
	}
	ordered := make([]int, 0, len(candidates))
	ordered = append(ordered, candidates[start:]...)
	ordered = append(ordered, candidates[:start]...)
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

func nextProviderPosition(candidates []int, providerIndex int) int {
	for i, candidate := range candidates {
		if candidate == providerIndex {
			return candidates[(i+1)%len(candidates)]
		}
	}
	return providerIndex
}

func requestedModel(body []byte) string {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	return strings.TrimSpace(req.Model)
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
	case "/chat/completions", "/v1/chat/completions":
		p.handleChatCompletions(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (p *proxy) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := readRequestBody(r, maxRequestBodyBytes)
	if err != nil {
		p.logger.Warn("failed to read chat completions request", zap.String("path", r.URL.Path), zap.Error(err))
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	model := requestedModel(body)
	candidates, err := p.pool.candidatesForModel(model)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "model_not_available", err.Error())
		return
	}

	var lastHTTP *bufferedResponse
	var lastErr error
	tried := make(map[int]struct{}, len(candidates))
	for attempt := 0; attempt < len(candidates); attempt++ {
		providerIndex, busyFallback, release, err := p.pool.acquire(candidates, tried)
		if err != nil {
			break
		}
		tried[providerIndex] = struct{}{}

		resp, err := p.postChat(r, body, providerIndex)
		if err != nil {
			if r.Context().Err() != nil {
				p.logger.Info("downstream canceled chat completions request", zap.Error(err))
				release()
				return
			}
			provider := p.pool.provider(providerIndex)
			lastErr = err
			release()
			p.logger.Warn("upstream chat completions request failed",
				zap.Int("attempt", attempt+1),
				zap.Int("provider_index", providerIndex),
				zap.String("provider_id", provider.id),
				zap.Bool("busy_fallback", busyFallback),
				zap.Error(err),
			)
			continue
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			defer release()
			defer closeBody(p.logger, resp.Body)
			copyResponseHeaders(w.Header(), resp.Header)
			w.WriteHeader(resp.StatusCode)
			if err := copyResponseBody(w, resp.Body); err != nil {
				p.logger.Warn("failed to copy upstream chat completions response",
					zap.Int("status", resp.StatusCode),
					zap.Int("provider_index", providerIndex),
					zap.Error(err),
				)
			}
			return
		}

		failure, readErr := bufferResponse(resp)
		provider := p.pool.provider(providerIndex)
		release()
		if readErr != nil {
			lastErr = readErr
			p.logger.Warn("failed to read upstream chat completions error response",
				zap.Int("attempt", attempt+1),
				zap.Int("provider_index", providerIndex),
				zap.String("provider_id", provider.id),
				zap.Bool("busy_fallback", busyFallback),
				zap.Int("status", resp.StatusCode),
				zap.Error(readErr),
			)
		} else {
			lastHTTP = failure
			lastErr = nil
			p.logger.Warn("upstream chat completions returned failure",
				zap.Int("attempt", attempt+1),
				zap.Int("provider_index", providerIndex),
				zap.String("provider_id", provider.id),
				zap.Bool("busy_fallback", busyFallback),
				zap.Int("status", resp.StatusCode),
			)
		}
	}

	if lastHTTP != nil {
		copyResponseHeaders(w.Header(), lastHTTP.header)
		w.WriteHeader(lastHTTP.statusCode)
		_, _ = w.Write(lastHTTP.body)
		return
	}
	msg := "all matching upstream providers failed"
	if lastErr != nil {
		msg = lastErr.Error()
	}
	writeJSONError(w, http.StatusBadGateway, "upstream_request_error", msg)
}

func (p *proxy) postChat(inbound *http.Request, body []byte, providerIndex int) (*http.Response, error) {
	provider := p.pool.provider(providerIndex)
	targetURL := urlWithRawQuery(provider.chatURL, inbound.URL.RawQuery)
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

func normalizeChatCompletionsURL(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("provider URL must include scheme and host: %s", raw)
	}
	path := strings.TrimRight(u.Path, "/")
	switch {
	case strings.HasSuffix(path, "/chat/completions"):
		u.Path = path
	case path == "":
		u.Path = "/v1/chat/completions"
	case strings.HasSuffix(path, "/v1"):
		u.Path = path + "/chat/completions"
	default:
		u.Path = path + "/chat/completions"
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
	switch {
	case strings.HasSuffix(path, "/chat/completions"):
		path = strings.TrimRight(strings.TrimSuffix(path, "/chat/completions"), "/")
		u.Path = path + "/models"
	case strings.HasSuffix(path, "/models"):
		u.Path = path
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

func copyResponseBody(w http.ResponseWriter, body io.Reader) error {
	if flusher, ok := w.(http.Flusher); ok {
		buf := make([]byte, 32<<10)
		for {
			n, readErr := body.Read(buf)
			if n > 0 {
				if _, writeErr := w.Write(buf[:n]); writeErr != nil {
					return writeErr
				}
				flusher.Flush()
			}
			if readErr != nil {
				if errors.Is(readErr, io.EOF) {
					return nil
				}
				return readErr
			}
		}
	}
	_, err := io.Copy(w, body)
	return err
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
