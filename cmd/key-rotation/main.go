package main

import (
	"bytes"
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
	listenAddr  string
	providerURL string
	apiKeys     keyList
	apiKeysEnv  string
	timeout     time.Duration
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
	keys, err := cfg.resolveAPIKeys()
	if err != nil {
		exitWithError(logger, err.Error(), zap.String("env", cfg.apiKeysEnv))
	}

	chatURL, err := normalizeChatCompletionsURL(cfg.providerURL)
	if err != nil {
		exitWithError(logger, "invalid provider URL", zap.Error(err))
	}

	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DisableCompression = true

	handler := newProxy(proxyConfig{
		ChatURL: chatURL,
		Keys:    keys,
		Client: &http.Client{
			Timeout:   cfg.timeout,
			Transport: transport,
		},
		Logger: logger,
	})

	logger.Info("key-rotation proxy listening",
		zap.String("listen", cfg.listenAddr),
		zap.String("upstream_chat_completions_url", chatURL),
		zap.Int("keys", len(keys)),
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
	flag.StringVar(&cfg.providerURL, "provider-url", "", "OpenAI-compatible upstream provider base URL, /v1 URL, or direct /chat/completions URL")
	flag.Var(&cfg.apiKeys, "api-key", "upstream provider API key; may be repeated and overrides inbound Authorization")
	flag.StringVar(&cfg.apiKeysEnv, "api-keys-env", "", "environment variable containing upstream API keys separated by commas or newlines")
	flag.DurationVar(&cfg.timeout, "timeout", 10*time.Minute, "upstream request timeout")
	flag.Parse()
	return cfg
}

func (c cliConfig) validate() error {
	switch {
	case c.providerURL == "":
		return errors.New("missing required flag: -provider-url")
	case len(c.apiKeys) > 0 && c.apiKeysEnv != "":
		return errors.New("only one API key source may be set: repeated -api-key or -api-keys-env")
	default:
		return nil
	}
}

func (c cliConfig) resolveAPIKeys() ([]string, error) {
	var keys []string
	if c.apiKeysEnv != "" {
		value := strings.TrimSpace(os.Getenv(c.apiKeysEnv))
		if value == "" {
			return nil, errors.New("API keys environment variable is unset or empty")
		}
		keys = splitAPIKeys(value)
	} else {
		for _, raw := range c.apiKeys {
			keys = append(keys, splitAPIKeys(raw)...)
		}
	}
	keys = compactStrings(keys)
	if len(keys) == 0 {
		return nil, errors.New("at least one upstream API key is required")
	}
	return keys, nil
}

type keyList []string

func (k *keyList) String() string {
	if k == nil {
		return ""
	}
	return strings.Join(*k, ",")
}

func (k *keyList) Set(value string) error {
	*k = append(*k, value)
	return nil
}

type proxyConfig struct {
	ChatURL string
	Keys    []string
	Client  *http.Client
	Logger  *zap.Logger
}

type proxy struct {
	chatURL string
	keys    *keyRing
	client  *http.Client
	logger  *zap.Logger
}

func newProxy(cfg proxyConfig) *proxy {
	client := cfg.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}
	return &proxy{
		chatURL: cfg.ChatURL,
		keys:    newKeyRing(cfg.Keys),
		client:  client,
		logger:  logger,
	}
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

	order := p.keys.order()
	if len(order) == 0 {
		writeJSONError(w, http.StatusBadGateway, "missing_api_keys", "no upstream API keys are configured")
		return
	}

	var lastHTTP *bufferedResponse
	var lastErr error
	for attempt, keyIndex := range order {
		resp, err := p.postChat(r, body, keyIndex)
		if err != nil {
			if r.Context().Err() != nil {
				p.logger.Info("downstream canceled chat completions request", zap.Error(err))
				return
			}
			lastErr = err
			p.keys.advanceFrom(keyIndex)
			p.logger.Warn("upstream chat completions request failed",
				zap.Int("attempt", attempt+1),
				zap.Int("key_index", keyIndex),
				zap.Error(err),
			)
			continue
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			defer closeBody(p.logger, resp.Body)
			copyResponseHeaders(w.Header(), resp.Header)
			w.WriteHeader(resp.StatusCode)
			if err := copyResponseBody(w, resp.Body); err != nil {
				p.logger.Warn("failed to copy upstream chat completions response",
					zap.Int("status", resp.StatusCode),
					zap.Error(err),
				)
			}
			return
		}

		failure, readErr := bufferResponse(resp)
		if readErr != nil {
			lastErr = readErr
			p.logger.Warn("failed to read upstream chat completions error response",
				zap.Int("attempt", attempt+1),
				zap.Int("key_index", keyIndex),
				zap.Int("status", resp.StatusCode),
				zap.Error(readErr),
			)
		} else {
			lastHTTP = failure
			lastErr = nil
			p.logger.Warn("upstream chat completions returned failure",
				zap.Int("attempt", attempt+1),
				zap.Int("key_index", keyIndex),
				zap.Int("status", resp.StatusCode),
			)
		}
		p.keys.advanceFrom(keyIndex)
	}

	if lastHTTP != nil {
		copyResponseHeaders(w.Header(), lastHTTP.header)
		w.WriteHeader(lastHTTP.statusCode)
		_, _ = w.Write(lastHTTP.body)
		return
	}
	msg := "all upstream API keys failed"
	if lastErr != nil {
		msg = lastErr.Error()
	}
	writeJSONError(w, http.StatusBadGateway, "upstream_request_error", msg)
}

func (p *proxy) postChat(inbound *http.Request, body []byte, keyIndex int) (*http.Response, error) {
	targetURL := urlWithRawQuery(p.chatURL, inbound.URL.RawQuery)
	req, err := http.NewRequestWithContext(inbound.Context(), http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	copyForwardHeaders(req.Header, inbound.Header)
	req.Header.Set("Authorization", authorizationHeader(p.keys.key(keyIndex)))
	return p.client.Do(req)
}

type keyRing struct {
	mu      sync.Mutex
	keys    []string
	current int
}

func newKeyRing(keys []string) *keyRing {
	return &keyRing{keys: append([]string(nil), keys...)}
}

func (r *keyRing) key(index int) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.keys) == 0 || index < 0 || index >= len(r.keys) {
		return ""
	}
	return r.keys[index]
}

func (r *keyRing) order() []int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.keys) == 0 {
		return nil
	}
	order := make([]int, 0, len(r.keys))
	for i := range r.keys {
		order = append(order, (r.current+i)%len(r.keys))
	}
	return order
}

func (r *keyRing) advanceFrom(index int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.keys) == 0 {
		return
	}
	if r.current == index {
		r.current = (r.current + 1) % len(r.keys)
	}
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

func splitAPIKeys(raw string) []string {
	return strings.FieldsFunc(raw, func(r rune) bool {
		return r == ',' || r == '\n' || r == '\r' || r == '\t'
	})
}

func compactStrings(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
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
