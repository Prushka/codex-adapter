package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestProxySkipsBusyProviderAndFallsBack(t *testing.T) {
	const requestBody = `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}`

	firstRelease := make(chan struct{})
	secondRelease := make(chan struct{})
	first := newMockProvider(t, []string{"m"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		<-firstRelease
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"one"}`))
	})
	defer first.Close()
	second := newMockProvider(t, []string{"m"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		<-secondRelease
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"two"}`))
	})
	defer second.Close()

	proxy := newTestProxy(t, first.providerConfig(), second.providerConfig())

	firstDone := make(chan int, 1)
	go func() {
		firstDone <- serveChat(proxy, requestBody).Code
	}()
	first.waitForChats(t, 1)

	secondDone := make(chan int, 1)
	go func() {
		secondDone <- serveChat(proxy, requestBody).Code
	}()
	second.waitForChats(t, 1)

	close(secondRelease)
	if got := waitStatus(t, secondDone); got != http.StatusOK {
		t.Fatalf("second status = %d", got)
	}
	close(firstRelease)
	if got := waitStatus(t, firstDone); got != http.StatusOK {
		t.Fatalf("first status = %d", got)
	}

	if got := first.joinAuthorizations(); got != "Bearer key-1" {
		t.Fatalf("first authorizations = %s", got)
	}
	if got := second.joinAuthorizations(); got != "Bearer key-2" {
		t.Fatalf("second authorizations = %s", got)
	}
	for _, body := range append(first.bodies(), second.bodies()...) {
		if body != requestBody {
			t.Fatalf("forwarded body = %s", body)
		}
	}
}

func TestProxyWrapsToFirstProviderWhenAllAreBusy(t *testing.T) {
	const requestBody = `{"model":"m","stream":false}`

	firstRelease := make(chan struct{})
	secondRelease := make(chan struct{})
	first := newMockProvider(t, []string{"m"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		<-firstRelease
		_, _ = w.Write([]byte(`{"id":"one"}`))
	})
	defer first.Close()
	second := newMockProvider(t, []string{"m"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		<-secondRelease
		_, _ = w.Write([]byte(`{"id":"two"}`))
	})
	defer second.Close()

	proxy := newTestProxy(t, first.providerConfig(), second.providerConfig())

	firstDone := make(chan int, 1)
	go func() { firstDone <- serveChat(proxy, requestBody).Code }()
	first.waitForChats(t, 1)

	secondDone := make(chan int, 1)
	go func() { secondDone <- serveChat(proxy, requestBody).Code }()
	second.waitForChats(t, 1)

	thirdDone := make(chan int, 1)
	go func() {
		thirdDone <- serveChatWithHeader(proxy, requestBody, "X-Req", "third").Code
	}()
	first.waitForChats(t, 2)

	close(secondRelease)
	close(firstRelease)
	if got := waitStatus(t, thirdDone); got != http.StatusOK {
		t.Fatalf("third status = %d", got)
	}
	if got := waitStatus(t, secondDone); got != http.StatusOK {
		t.Fatalf("second status = %d", got)
	}
	if got := waitStatus(t, firstDone); got != http.StatusOK {
		t.Fatalf("first status = %d", got)
	}

	if got := first.chatCount(); got != 2 {
		t.Fatalf("first chat count = %d", got)
	}
	if got := second.chatCount(); got != 1 {
		t.Fatalf("second chat count = %d", got)
	}
}

func TestProxyKeepsFallbackResponseFromLastProvider(t *testing.T) {
	const requestBody = `{"model":"m","stream":false}`

	first := newMockProvider(t, []string{"m"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Upstream-Error", "first")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"first"}}`))
	})
	defer first.Close()
	second := newMockProvider(t, []string{"m"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Upstream-Error", "second")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"second"}}`))
	})
	defer second.Close()

	proxy := newTestProxy(t, first.providerConfig(), second.providerConfig())
	rec := serveChat(proxy, requestBody)
	res := rec.Result()
	defer res.Body.Close()
	gotBody, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusUnauthorized {
		t.Fatalf("status = %d body = %s", res.StatusCode, gotBody)
	}
	if string(gotBody) != `{"error":{"message":"second"}}` {
		t.Fatalf("body = %s", gotBody)
	}
	if res.Header.Get("X-Upstream-Error") != "second" {
		t.Fatalf("missing last upstream error header: %#v", res.Header)
	}
}

func TestProxyRepeatsProviderPoolWhenAttemptsIsGreaterThanOne(t *testing.T) {
	const requestBody = `{"model":"m","stream":false}`

	first := newMockProvider(t, []string{"m"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"first"}}`))
	})
	defer first.Close()
	var secondMu sync.Mutex
	secondAttempts := 0
	second := newMockProvider(t, []string{"m"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		secondMu.Lock()
		secondAttempts++
		attempt := secondAttempts
		secondMu.Unlock()
		w.Header().Set("X-Provider-Attempt", fmt.Sprint(attempt))
		if attempt == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"second failed"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"second succeeded"}`))
	})
	defer second.Close()

	proxy := newTestProxyWithAttempts(t, 2, first.providerConfig(), second.providerConfig())
	rec := serveChat(proxy, requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := first.chatCount(); got != 2 {
		t.Fatalf("first chat count = %d", got)
	}
	if got := second.chatCount(); got != 2 {
		t.Fatalf("second chat count = %d", got)
	}
	if got := rec.Body.String(); got != `{"id":"second succeeded"}` {
		t.Fatalf("body = %s", got)
	}
	if got := rec.Header().Get("X-Provider-Attempt"); got != "2" {
		t.Fatalf("X-Provider-Attempt = %q", got)
	}
}

func TestProxyDelaysBetweenAttemptPasses(t *testing.T) {
	const requestBody = `{"model":"m","stream":false}`

	provider := newMockProvider(t, []string{"m"}, "key", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"try again"}}`))
	})
	defer provider.Close()

	proxy := newTestProxyWithRetryConfig(t, 2, 20*time.Millisecond, provider.providerConfig())
	start := time.Now()
	rec := serveChat(proxy, requestBody)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := provider.chatCount(); got != 2 {
		t.Fatalf("chat count = %d", got)
	}
	if elapsed := time.Since(start); elapsed < 20*time.Millisecond {
		t.Fatalf("expected retry delay, elapsed = %s", elapsed)
	}
}

func TestProxyOnlyUsesProvidersWithRequestedModel(t *testing.T) {
	const requestBody = `{"model":"b","stream":false}`

	first := newMockProvider(t, []string{"a"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("provider without model should not receive chat request")
	})
	defer first.Close()
	second := newMockProvider(t, []string{"b"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"two"}`))
	})
	defer second.Close()

	proxy := newTestProxy(t, first.providerConfig(), second.providerConfig())
	rec := serveChat(proxy, requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := first.chatCount(); got != 0 {
		t.Fatalf("first chat count = %d", got)
	}
	if got := second.chatCount(); got != 1 {
		t.Fatalf("second chat count = %d", got)
	}
}

func TestProxyModelRoutingUsesGlobalProviderOrder(t *testing.T) {
	const requestBody = `{"model":"b","stream":false}`

	first := newMockProvider(t, []string{"a"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("provider without model should not receive chat request")
	})
	defer first.Close()
	second := newMockProvider(t, []string{"b"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"two"}`))
	})
	defer second.Close()
	third := newMockProvider(t, []string{"b"}, "key-3", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"three"}`))
	})
	defer third.Close()

	proxy := newTestProxy(t, first.providerConfig(), second.providerConfig(), third.providerConfig())
	for i := 0; i < 4; i++ {
		rec := serveChat(proxy, requestBody)
		if rec.Code != http.StatusOK {
			t.Fatalf("request %d status = %d body = %s", i+1, rec.Code, rec.Body.String())
		}
	}
	if got := first.chatCount(); got != 0 {
		t.Fatalf("first chat count = %d", got)
	}
	if got := second.chatCount(); got != 2 {
		t.Fatalf("second chat count = %d", got)
	}
	if got := third.chatCount(); got != 2 {
		t.Fatalf("third chat count = %d", got)
	}
	if got := strings.Join(append(second.bodies(), third.bodies()...), ","); got == "" {
		t.Fatal("expected forwarded requests")
	}
}

func TestProxyReturnsModelErrorWhenUnavailable(t *testing.T) {
	provider := newMockProvider(t, []string{"a"}, "key", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("chat request should not be forwarded")
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	rec := serveChat(proxy, `{"model":"missing"}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "model_not_available") {
		t.Fatalf("body = %s", rec.Body.String())
	}
}

func TestProxyReturnsInvalidRequestWhenModelIsMissing(t *testing.T) {
	provider := newMockProvider(t, []string{"a"}, "key", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("chat request should not be forwarded")
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	rec := serveChat(proxy, `{"stream":false}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "invalid_request_error") {
		t.Fatalf("body = %s", rec.Body.String())
	}
}

func TestProxyReturnsInvalidRequestWhenBodyIsMalformed(t *testing.T) {
	provider := newMockProvider(t, []string{"a"}, "key", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("chat request should not be forwarded")
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	rec := serveChat(proxy, `{"model":`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "invalid_request_error") {
		t.Fatalf("body = %s", rec.Body.String())
	}
}

func TestProviderPoolLoadsModels(t *testing.T) {
	provider := newMockProvider(t, []string{"gpt-a", "gpt-b"}, "secret", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"ok"}`))
	})
	defer provider.Close()

	pool := newTestPool(t, provider.providerConfig())
	candidates, err := pool.candidatesForModel("gpt-b")
	if err != nil {
		t.Fatalf("candidatesForModel: %v", err)
	}
	if len(candidates) != 1 || candidates[0] != 0 {
		t.Fatalf("candidates = %#v", candidates)
	}
	if got := provider.joinModelAuthorizations(); got != "Bearer secret" {
		t.Fatalf("model authorizations = %q", got)
	}
}

func TestParseProviderConfig(t *testing.T) {
	cfg, err := parseProviderConfig("p1,https://example.test/v1,sk-key")
	if err != nil {
		t.Fatalf("parseProviderConfig: %v", err)
	}
	if cfg.ID != "p1" || cfg.ProviderURL != "https://example.test/v1" || cfg.APIKey != "sk-key" {
		t.Fatalf("provider = %#v", cfg)
	}
}

func TestParseProviderConfigAllowsCommaInKey(t *testing.T) {
	cfg, err := parseProviderConfig("p1,https://example.test/v1,Bearer key,with,commas")
	if err != nil {
		t.Fatalf("parseProviderConfig: %v", err)
	}
	if cfg.APIKey != "Bearer key,with,commas" {
		t.Fatalf("key = %q", cfg.APIKey)
	}
}

func TestParseProviderConfigRequiresTuple(t *testing.T) {
	if _, err := parseProviderConfig("p1,https://example.test/v1"); err == nil {
		t.Fatal("expected tuple error")
	}
}

func TestValidateRejectsDuplicateProviderIDs(t *testing.T) {
	cfg := cliConfig{providers: providerList{
		{ID: "same", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
		{ID: "same", ProviderURL: "https://two.test/v1", APIKey: "key-two"},
	}, attempts: 1, delay: time.Minute}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected duplicate provider id error")
	}
}

func TestValidateRejectsInvalidAttempts(t *testing.T) {
	cfg := cliConfig{providers: providerList{
		{ID: "p1", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
	}}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected attempts error")
	}
}

func TestValidateRejectsNegativeDelay(t *testing.T) {
	cfg := cliConfig{providers: providerList{
		{ID: "p1", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
	}, attempts: 1, delay: -time.Second}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected delay error")
	}
}

func TestNormalizeChatCompletionsURL(t *testing.T) {
	tests := map[string]string{
		"https://example.test":                     "https://example.test/v1/chat/completions",
		"https://example.test/v1":                  "https://example.test/v1/chat/completions",
		"https://example.test/openai":              "https://example.test/openai/chat/completions",
		"https://example.test/v1/chat/completions": "https://example.test/v1/chat/completions",
	}
	for raw, want := range tests {
		got, err := normalizeChatCompletionsURL(raw)
		if err != nil {
			t.Fatalf("normalizeChatCompletionsURL(%q): %v", raw, err)
		}
		if got != want {
			t.Fatalf("normalizeChatCompletionsURL(%q) = %q, want %q", raw, got, want)
		}
	}
}

func TestNormalizeModelsURL(t *testing.T) {
	tests := map[string]string{
		"https://example.test":                     "https://example.test/v1/models",
		"https://example.test/v1":                  "https://example.test/v1/models",
		"https://example.test/openai":              "https://example.test/openai/models",
		"https://example.test/v1/chat/completions": "https://example.test/v1/models",
		"https://example.test/v1/models":           "https://example.test/v1/models",
	}
	for raw, want := range tests {
		got, err := normalizeModelsURL(raw)
		if err != nil {
			t.Fatalf("normalizeModelsURL(%q): %v", raw, err)
		}
		if got != want {
			t.Fatalf("normalizeModelsURL(%q) = %q, want %q", raw, got, want)
		}
	}
}

func TestRequestedModel(t *testing.T) {
	body := []byte(`{"model":"gpt-x"}`)
	got, err := requestedModel(body)
	if err != nil {
		t.Fatalf("requestedModel: %v", err)
	}
	if got != "gpt-x" {
		t.Fatalf("requestedModel = %q", got)
	}
}

func TestRequestedModelRejectsMissingModel(t *testing.T) {
	if _, err := requestedModel([]byte(`{"stream":false}`)); err == nil {
		t.Fatal("expected missing model error")
	}
}

func TestRequestedModelRejectsMalformedJSON(t *testing.T) {
	if _, err := requestedModel([]byte(`{"model":`)); err == nil {
		t.Fatal("expected malformed JSON error")
	}
}

func TestFetchProviderModelsParsesResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"id": "one"}, {"id": "two"}},
		})
	}))
	defer upstream.Close()

	models, err := fetchProviderModels(context.Background(), upstream.Client(), upstream.URL, "key")
	if err != nil {
		t.Fatalf("fetchProviderModels: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("models = %#v", models)
	}
}

type mockProvider struct {
	*httptest.Server
	t      *testing.T
	apiKey string
	mu     sync.Mutex

	modelAuthorizations []string
	authorizations      []string
	requestBodies       []string
	chatHandler         http.HandlerFunc
}

func newMockProvider(t *testing.T, models []string, apiKey string, chatHandler http.HandlerFunc) *mockProvider {
	t.Helper()
	p := &mockProvider{t: t, apiKey: apiKey, chatHandler: chatHandler}
	p.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			p.mu.Lock()
			p.modelAuthorizations = append(p.modelAuthorizations, r.Header.Get("Authorization"))
			p.mu.Unlock()
			data := make([]map[string]string, 0, len(models))
			for _, model := range models {
				data = append(data, map[string]string{"id": model})
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
		case "/v1/chat/completions":
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatal(err)
			}
			p.mu.Lock()
			p.authorizations = append(p.authorizations, r.Header.Get("Authorization"))
			p.requestBodies = append(p.requestBodies, string(body))
			p.mu.Unlock()
			chatHandler(w, r)
		default:
			t.Fatalf("unexpected path = %s", r.URL.Path)
		}
	}))
	return p
}

func (p *mockProvider) providerConfig() providerConfig {
	return providerConfig{ID: p.apiKey, ProviderURL: p.URL + "/v1", APIKey: p.apiKey}
}

func (p *mockProvider) waitForChats(t *testing.T, want int) {
	t.Helper()
	deadline := time.After(2 * time.Second)
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		if p.chatCount() >= want {
			return
		}
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for %d chat requests, got %d", want, p.chatCount())
		case <-ticker.C:
		}
	}
}

func (p *mockProvider) chatCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.authorizations)
}

func (p *mockProvider) joinAuthorizations() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return strings.Join(p.authorizations, ",")
}

func (p *mockProvider) joinModelAuthorizations() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return strings.Join(p.modelAuthorizations, ",")
}

func (p *mockProvider) bodies() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.requestBodies...)
}

func newTestProxy(t *testing.T, providers ...providerConfig) *proxy {
	t.Helper()
	return newTestProxyWithAttempts(t, 1, providers...)
}

func newTestProxyWithAttempts(t *testing.T, attempts int, providers ...providerConfig) *proxy {
	t.Helper()
	return newTestProxyWithRetryConfig(t, attempts, 0, providers...)
}

func newTestProxyWithRetryConfig(t *testing.T, attempts int, delay time.Duration, providers ...providerConfig) *proxy {
	t.Helper()
	pool := newTestPool(t, providers...)
	return &proxy{pool: pool, client: http.DefaultClient, logger: newTestLogger(), attempts: attempts, delay: delay}
}

func newTestPool(t *testing.T, providers ...providerConfig) *providerPool {
	t.Helper()
	pool, err := newProviderPool(context.Background(), providers, http.DefaultClient, newTestLogger())
	if err != nil {
		t.Fatalf("newProviderPool: %v", err)
	}
	return pool
}

func serveChat(p *proxy, body string) *httptest.ResponseRecorder {
	return serveChatWithHeader(p, body, "", "")
}

func serveChatWithHeader(p *proxy, body string, headerName string, headerValue string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if headerName != "" {
		req.Header.Set(headerName, headerValue)
	}
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)
	return rec
}

func waitStatus(t *testing.T, ch <-chan int) int {
	t.Helper()
	select {
	case status := <-ch:
		return status
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for request")
	}
	return 0
}

func newTestLogger() *zap.Logger {
	return zap.NewNop()
}
