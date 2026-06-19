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
	"go.uber.org/zap/zaptest/observer"
)

const testProxyAPIKey = "lb-test-key"

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

func TestProxyModelRoutingUsesCandidateOrder(t *testing.T) {
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
	var orderMu sync.Mutex
	orderCalls := 0
	proxy.pool.candidateOrder = func(candidates []int) []int {
		if got := fmt.Sprint(candidates); got != "[1 2]" {
			t.Fatalf("candidates = %s", got)
		}
		orderMu.Lock()
		defer orderMu.Unlock()
		orderCalls++
		if orderCalls%2 == 0 {
			return []int{2, 1}
		}
		return []int{1, 2}
	}
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

func TestProxyLogsSuccessfulRequestWithoutBodies(t *testing.T) {
	const requestBody = `{"model":"m","messages":[{"role":"user","content":"secret-input"}],"stream":false}`
	const responseBody = `{"id":"secret-output"}`

	core, logs := observer.New(zap.InfoLevel)
	provider := newMockProvider(t, []string{"m"}, "key", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(responseBody))
	})
	defer provider.Close()

	proxy := newTestProxyWithLogger(t, zap.New(core), provider.providerConfig())
	rec := serveChat(proxy, requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}

	entries := logs.FilterMessage("upstream request succeeded").All()
	if len(entries) != 1 {
		t.Fatalf("success log entries = %d", len(entries))
	}
	fields := entries[0].ContextMap()
	if fields["api"] != string(endpointChatCompletions) || fields["path"] != "/v1/chat/completions" || fields["model"] != "m" {
		t.Fatalf("bad request fields = %#v", fields)
	}
	if fields["provider_id"] != "key" || fields["status"] != int64(http.StatusOK) || fields["request_bytes"] != int64(len(requestBody)) {
		t.Fatalf("bad upstream fields = %#v", fields)
	}
	if fields["response_bytes"] != int64(len(responseBody)) {
		t.Fatalf("response_bytes = %#v", fields["response_bytes"])
	}
	assertLogFieldsDoNotContain(t, fields, "secret-input", "secret-output")
}

func TestProxyRejectsMissingAPIKey(t *testing.T) {
	provider := newMockProvider(t, []string{"m"}, "provider-key", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unauthorized request should not be forwarded")
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "invalid_api_key") {
		t.Fatalf("body = %s", rec.Body.String())
	}
	if got := provider.chatCount(); got != 0 {
		t.Fatalf("chat count = %d", got)
	}
}

func TestProxyRejectsWrongAPIKey(t *testing.T) {
	provider := newMockProvider(t, []string{"m"}, "provider-key", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unauthorized request should not be forwarded")
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	rec := serveChatWithHeader(proxy, `{"model":"m"}`, "Authorization", "Bearer wrong-key")
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := provider.chatCount(); got != 0 {
		t.Fatalf("chat count = %d", got)
	}
}

func TestProxyListsAllModels(t *testing.T) {
	first := newMockProvider(t, []string{"z-model", "a-model"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("models request should not be forwarded as chat")
	})
	defer first.Close()
	second := newMockProvider(t, []string{"a-model", "b-model"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("models request should not be forwarded as chat")
	})
	defer second.Close()

	proxy := newTestProxy(t, first.providerConfig(), second.providerConfig())
	rec := serveModelPath(proxy, "/v1/models", true)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}

	var got modelListResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if got.Object != "list" {
		t.Fatalf("object = %q", got.Object)
	}
	if len(got.Data) != 3 {
		t.Fatalf("data = %#v", got.Data)
	}
	var ids []string
	for _, model := range got.Data {
		ids = append(ids, model.ID)
		if model.Object != "model" || model.OwnedBy != "load-balancer" || model.Created == 0 {
			t.Fatalf("bad model item = %#v", model)
		}
	}
	if strings.Join(ids, ",") != "a-model,b-model,z-model" {
		t.Fatalf("model ids = %v", ids)
	}
}

func TestProxyListsProviderModelMap(t *testing.T) {
	first := newMockProvider(t, []string{"z-model", "a-model"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("models map request should not be forwarded as chat")
	})
	defer first.Close()
	second := newMockProvider(t, []string{"b-model"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("models map request should not be forwarded as chat")
	})
	defer second.Close()

	firstCfg := first.providerConfig()
	firstCfg.ID = "p1"
	secondCfg := second.providerConfig()
	secondCfg.ID = "p2"
	proxy := newTestProxy(t, firstCfg, secondCfg)
	rec := serveModelPath(proxy, "/v1/models/map", true)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}

	var got map[string][]string
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("map = %#v", got)
	}
	if strings.Join(got["p1"], ",") != "a-model,z-model" {
		t.Fatalf("p1 models = %#v", got["p1"])
	}
	if strings.Join(got["p2"], ",") != "b-model" {
		t.Fatalf("p2 models = %#v", got["p2"])
	}
}

func TestProxyProviderStatusReportsBusySuccessAndFailures(t *testing.T) {
	const requestBody = `{"model":"m","stream":false}`

	var modeMu sync.Mutex
	mode := "block"
	release := make(chan struct{})
	provider := newMockProvider(t, []string{"m", "z"}, "key", func(w http.ResponseWriter, r *http.Request) {
		modeMu.Lock()
		currentMode := mode
		modeMu.Unlock()
		switch currentMode {
		case "block":
			<-release
			_, _ = w.Write([]byte(`{"id":"ok"}`))
		case "fail":
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
		default:
			_, _ = w.Write([]byte(`{"id":"ok"}`))
		}
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	done := make(chan int, 1)
	go func() {
		done <- serveChat(proxy, requestBody).Code
	}()
	provider.waitForChats(t, 1)

	status := decodeProviderStatus(t, serveProviderStatus(proxy, "/v1/providers/status", true))
	if len(status.Data) != 1 {
		t.Fatalf("providers = %#v", status.Data)
	}
	got := status.Data[0]
	if got.ID != "key" || strings.Join(got.Models, ",") != "m,z" || got.BusyCount != 1 {
		t.Fatalf("busy status = %#v", got)
	}
	if got.Cooldown.Active || got.Cooldown.Until != nil || got.Cooldown.RemainingMillis != 0 {
		t.Fatalf("cooldown = %#v", got.Cooldown)
	}

	close(release)
	if code := waitStatus(t, done); code != http.StatusOK {
		t.Fatalf("blocked status = %d", code)
	}

	modeMu.Lock()
	mode = "fail"
	modeMu.Unlock()
	rec := serveChat(proxy, requestBody)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("failure status = %d body = %s", rec.Code, rec.Body.String())
	}

	status = decodeProviderStatus(t, serveProviderStatus(proxy, "/v1/providers", true))
	got = status.Data[0]
	if got.BusyCount != 0 || got.LastSuccessAt == nil || got.LastFailureAt == nil {
		t.Fatalf("final status = %#v", got)
	}
	if got.RecentFailureCount != 1 || len(got.RecentFailures) != 1 {
		t.Fatalf("recent failures = %#v", got.RecentFailures)
	}
	if failure := got.RecentFailures[0]; failure.Endpoint != string(endpointChatCompletions) || failure.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("failure = %#v", failure)
	}
	if got.ConsecutiveFailures != 1 {
		t.Fatalf("consecutive failures = %d", got.ConsecutiveFailures)
	}
}

func TestProxySkipsProviderInCooldown(t *testing.T) {
	const requestBody = `{"model":"m","stream":false}`

	first := newMockProvider(t, []string{"m"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	})
	defer first.Close()
	second := newMockProvider(t, []string{"m"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"second"}`))
	})
	defer second.Close()

	proxy, err := newProxy(proxyConfig{
		Providers:            []providerConfig{first.providerConfig(), second.providerConfig()},
		Client:               http.DefaultClient,
		Logger:               newTestLogger(),
		APIKey:               testProxyAPIKey,
		Attempts:             1,
		ProviderCooldown:     time.Minute,
		CooldownFailures:     1,
		ModelRefreshInterval: 0,
	})
	if err != nil {
		t.Fatalf("newProxy: %v", err)
	}
	proxy.pool.candidateOrder = stableCandidateOrder

	rec := serveChat(proxy, requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("first request status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := first.chatCount(); got != 1 {
		t.Fatalf("first chat count after first request = %d", got)
	}
	if got := second.chatCount(); got != 1 {
		t.Fatalf("second chat count after first request = %d", got)
	}

	status := decodeProviderStatus(t, serveProviderStatus(proxy, "/v1/providers/status", true))
	if !status.Data[0].Cooldown.Active || status.Data[0].Cooldown.Until == nil || status.Data[0].Cooldown.RemainingMillis <= 0 {
		t.Fatalf("cooldown status = %#v", status.Data[0].Cooldown)
	}

	rec = serveChat(proxy, requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("second request status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := first.chatCount(); got != 1 {
		t.Fatalf("first provider should have been skipped during cooldown, got %d requests", got)
	}
	if got := second.chatCount(); got != 2 {
		t.Fatalf("second chat count after second request = %d", got)
	}
}

func TestProxyRejectsUnauthorizedProviderStatus(t *testing.T) {
	provider := newMockProvider(t, []string{"m"}, "provider-key", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unauthorized provider status should not be forwarded")
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	rec := serveProviderStatus(proxy, "/v1/providers/status", false)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
}

func TestProxyRejectsUnauthorizedModelList(t *testing.T) {
	provider := newMockProvider(t, []string{"m"}, "provider-key", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("unauthorized model list should not be forwarded")
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	rec := serveModelPath(proxy, "/v1/models", false)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "invalid_api_key") {
		t.Fatalf("body = %s", rec.Body.String())
	}
}

func TestProxyForwardsResponsesRequest(t *testing.T) {
	const requestBody = `{"model":"m","input":"hi","stream":false}`

	provider := newMockProviderWithResponses(t, []string{"m"}, "key", nil, func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.RawQuery; got != "trace=1" {
			t.Fatalf("raw query = %q", got)
		}
		if got := r.Header.Get("X-Req"); got != "responses" {
			t.Fatalf("X-Req = %q", got)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer key" {
			t.Fatalf("authorization = %q", got)
		}
		w.Header().Set("X-Upstream", "responses")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1"}`))
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	rec := serveResponsesWithPathAndHeader(proxy, "/v1/responses?trace=1", requestBody, "X-Req", "responses")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := rec.Body.String(); got != `{"id":"resp_1"}` {
		t.Fatalf("body = %s", got)
	}
	if got := rec.Header().Get("X-Upstream"); got != "responses" {
		t.Fatalf("X-Upstream = %q", got)
	}
	if got := provider.chatCount(); got != 0 {
		t.Fatalf("chat count = %d", got)
	}
	if got := provider.responsesCount(); got != 1 {
		t.Fatalf("responses count = %d", got)
	}
	if got := provider.joinResponseAuthorizations(); got != "Bearer key" {
		t.Fatalf("response authorizations = %q", got)
	}
	if got := strings.Join(provider.responsesBodies(), ","); got != requestBody {
		t.Fatalf("response bodies = %s", got)
	}
}

func TestProxyForwardsResponsesShortPath(t *testing.T) {
	const requestBody = `{"model":"m","input":"hi"}`

	provider := newMockProviderWithResponses(t, []string{"m"}, "key", nil, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"resp_short"}`))
	})
	defer provider.Close()

	proxy := newTestProxy(t, provider.providerConfig())
	rec := serveResponsesWithPath(proxy, "/responses", requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := provider.responsesCount(); got != 1 {
		t.Fatalf("responses count = %d", got)
	}
}

func TestProxyResponsesUsesSiblingURLWhenProviderURLIsChatCompletions(t *testing.T) {
	const requestBody = `{"model":"m","input":"hi"}`

	provider := newMockProviderWithResponses(t, []string{"m"}, "key", nil, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"resp_from_sibling"}`))
	})
	defer provider.Close()

	cfg := provider.providerConfig()
	cfg.ProviderURL = provider.URL + "/v1/chat/completions"
	proxy := newTestProxy(t, cfg)
	rec := serveResponses(proxy, requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := rec.Body.String(); got != `{"id":"resp_from_sibling"}` {
		t.Fatalf("body = %s", got)
	}
	if got := provider.responsesCount(); got != 1 {
		t.Fatalf("responses count = %d", got)
	}
}

func TestProxyResponsesRetriesNextProviderOnFailure(t *testing.T) {
	const requestBody = `{"model":"m","input":"hi","stream":false}`

	first := newMockProviderWithResponses(t, []string{"m"}, "key-1", nil, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Upstream-Error", "first")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"first"}}`))
	})
	defer first.Close()
	second := newMockProviderWithResponses(t, []string{"m"}, "key-2", nil, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Upstream", "second")
		_, _ = w.Write([]byte(`{"id":"resp_second"}`))
	})
	defer second.Close()

	proxy := newTestProxy(t, first.providerConfig(), second.providerConfig())
	rec := serveResponses(proxy, requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := rec.Body.String(); got != `{"id":"resp_second"}` {
		t.Fatalf("body = %s", got)
	}
	if got := rec.Header().Get("X-Upstream"); got != "second" {
		t.Fatalf("X-Upstream = %q", got)
	}
	if got := first.responsesCount(); got != 1 {
		t.Fatalf("first responses count = %d", got)
	}
	if got := second.responsesCount(); got != 1 {
		t.Fatalf("second responses count = %d", got)
	}
}

func TestProxyResponsesOnlyUsesProvidersWithRequestedModel(t *testing.T) {
	const requestBody = `{"model":"b","input":"hi","stream":false}`

	first := newMockProviderWithResponses(t, []string{"a"}, "key-1", nil, func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("provider without model should not receive responses request")
	})
	defer first.Close()
	second := newMockProviderWithResponses(t, []string{"b"}, "key-2", nil, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"two"}`))
	})
	defer second.Close()

	proxy := newTestProxy(t, first.providerConfig(), second.providerConfig())
	rec := serveResponses(proxy, requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := first.responsesCount(); got != 0 {
		t.Fatalf("first responses count = %d", got)
	}
	if got := second.responsesCount(); got != 1 {
		t.Fatalf("second responses count = %d", got)
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

func TestProviderPoolRefreshModelsUpdatesRouting(t *testing.T) {
	provider := newMockProvider(t, []string{"a"}, "secret", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"ok"}`))
	})
	defer provider.Close()

	pool := newTestPool(t, provider.providerConfig())
	provider.setModels("b", "c")
	pool.refreshModels(context.Background(), http.DefaultClient, newTestLogger(), time.Second)

	if _, err := pool.candidatesForModel("a"); err == nil {
		t.Fatal("expected old model to be removed")
	}
	candidates, err := pool.candidatesForModel("b")
	if err != nil {
		t.Fatalf("candidatesForModel: %v", err)
	}
	if len(candidates) != 1 || candidates[0] != 0 {
		t.Fatalf("candidates = %#v", candidates)
	}
	statuses := pool.providerStatuses(time.Now())
	if len(statuses) != 1 {
		t.Fatalf("statuses = %#v", statuses)
	}
	if got := strings.Join(statuses[0].Models, ","); got != "b,c" {
		t.Fatalf("status models = %s", got)
	}
	if statuses[0].ModelRefresh.LastSuccessAt == nil || statuses[0].ModelRefresh.LastFailureAt != nil || statuses[0].ModelRefresh.LastError != "" {
		t.Fatalf("model refresh status = %#v", statuses[0].ModelRefresh)
	}
}

func TestProviderPoolRefreshModelsKeepsPreviousModelsOnFailure(t *testing.T) {
	provider := newMockProvider(t, []string{"a"}, "secret", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"ok"}`))
	})
	pool := newTestPool(t, provider.providerConfig())
	provider.Close()

	pool.refreshModels(context.Background(), http.DefaultClient, newTestLogger(), 50*time.Millisecond)

	candidates, err := pool.candidatesForModel("a")
	if err != nil {
		t.Fatalf("candidatesForModel: %v", err)
	}
	if len(candidates) != 1 || candidates[0] != 0 {
		t.Fatalf("candidates = %#v", candidates)
	}
	statuses := pool.providerStatuses(time.Now())
	if len(statuses[0].RecentFailures) != 1 {
		t.Fatalf("recent failures = %#v", statuses[0].RecentFailures)
	}
	if failure := statuses[0].RecentFailures[0]; failure.Endpoint != "models" || failure.Error == "" {
		t.Fatalf("failure = %#v", failure)
	}
	if statuses[0].ModelRefresh.LastFailureAt == nil || statuses[0].ModelRefresh.LastError == "" {
		t.Fatalf("model refresh status = %#v", statuses[0].ModelRefresh)
	}
}

func TestProxyRefreshesProviderModelsInBackground(t *testing.T) {
	provider := newMockProvider(t, []string{"a"}, "secret", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"ok"}`))
	})
	defer provider.Close()

	proxy, err := newProxy(proxyConfig{
		Providers:            []providerConfig{provider.providerConfig()},
		Client:               http.DefaultClient,
		Logger:               newTestLogger(),
		APIKey:               testProxyAPIKey,
		Attempts:             1,
		ModelRefreshInterval: 10 * time.Millisecond,
		ModelRefreshTimeout:  time.Second,
	})
	if err != nil {
		t.Fatalf("newProxy: %v", err)
	}
	defer proxy.Close()

	provider.setModels("a", "b")
	waitForModel(t, proxy, "b")

	rec := serveChat(proxy, `{"model":"b","stream":false}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
}

func TestProxyUsesShuffledProviderOrder(t *testing.T) {
	const requestBody = `{"model":"m","stream":false}`

	first := newMockProvider(t, []string{"m"}, "key-1", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("first provider should not receive chat request")
	})
	defer first.Close()
	second := newMockProvider(t, []string{"m"}, "key-2", func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("second provider should not receive chat request")
	})
	defer second.Close()
	third := newMockProvider(t, []string{"m"}, "key-3", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"three"}`))
	})
	defer third.Close()

	proxy := newTestProxy(t, first.providerConfig(), second.providerConfig(), third.providerConfig())
	proxy.pool.candidateOrder = func(candidates []int) []int {
		if got := fmt.Sprint(candidates); got != "[0 1 2]" {
			t.Fatalf("candidates = %s", got)
		}
		return []int{2, 0, 1}
	}

	rec := serveChat(proxy, requestBody)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if got := third.chatCount(); got != 1 {
		t.Fatalf("third chat count = %d", got)
	}
	if got := first.chatCount(); got != 0 {
		t.Fatalf("first chat count = %d", got)
	}
	if got := second.chatCount(); got != 0 {
		t.Fatalf("second chat count = %d", got)
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

func TestValidateRejectsMissingAPIKey(t *testing.T) {
	cfg := cliConfig{providers: providerList{
		{ID: "p1", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
	}, attempts: 1, delay: time.Minute}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected api key error")
	}
}

func TestValidateRejectsDuplicateProviderIDs(t *testing.T) {
	cfg := cliConfig{apiKey: testProxyAPIKey, providers: providerList{
		{ID: "same", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
		{ID: "same", ProviderURL: "https://two.test/v1", APIKey: "key-two"},
	}, attempts: 1, delay: time.Minute}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected duplicate provider id error")
	}
}

func TestValidateRejectsInvalidAttempts(t *testing.T) {
	cfg := cliConfig{apiKey: testProxyAPIKey, providers: providerList{
		{ID: "p1", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
	}}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected attempts error")
	}
}

func TestValidateRejectsNegativeDelay(t *testing.T) {
	cfg := cliConfig{apiKey: testProxyAPIKey, providers: providerList{
		{ID: "p1", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
	}, attempts: 1, delay: -time.Second}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected delay error")
	}
}

func TestValidateRejectsNegativeModelRefreshInterval(t *testing.T) {
	cfg := cliConfig{apiKey: testProxyAPIKey, providers: providerList{
		{ID: "p1", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
	}, attempts: 1, refresh: -time.Second}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected model refresh interval error")
	}
}

func TestValidateRejectsNegativeModelRefreshTimeout(t *testing.T) {
	cfg := cliConfig{apiKey: testProxyAPIKey, providers: providerList{
		{ID: "p1", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
	}, attempts: 1, refreshTimeout: -time.Second}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected model refresh timeout error")
	}
}

func TestValidateRejectsNegativeProviderCooldown(t *testing.T) {
	cfg := cliConfig{apiKey: testProxyAPIKey, providers: providerList{
		{ID: "p1", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
	}, attempts: 1, cooldown: -time.Second}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected provider cooldown error")
	}
}

func TestValidateRejectsNegativeCooldownFailures(t *testing.T) {
	cfg := cliConfig{apiKey: testProxyAPIKey, providers: providerList{
		{ID: "p1", ProviderURL: "https://one.test/v1", APIKey: "key-one"},
	}, attempts: 1, cooldownFailures: -1}
	if err := cfg.validate(); err == nil {
		t.Fatal("expected provider cooldown failures error")
	}
}

func TestNormalizeChatCompletionsURL(t *testing.T) {
	tests := map[string]string{
		"https://example.test":                     "https://example.test/v1/chat/completions",
		"https://example.test/v1":                  "https://example.test/v1/chat/completions",
		"https://example.test/openai":              "https://example.test/openai/chat/completions",
		"https://example.test/v1/chat/completions": "https://example.test/v1/chat/completions",
		"https://example.test/v1/models":           "https://example.test/v1/chat/completions",
		"https://example.test/v1/responses":        "https://example.test/v1/chat/completions",
		"https://example.test/responses":           "https://example.test/chat/completions",
		"https://example.test/v1/responses?x=1":    "https://example.test/v1/chat/completions?x=1",
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

func TestNormalizeResponsesURL(t *testing.T) {
	tests := map[string]string{
		"https://example.test":                         "https://example.test/v1/responses",
		"https://example.test/v1":                      "https://example.test/v1/responses",
		"https://example.test/openai":                  "https://example.test/openai/responses",
		"https://example.test/v1/chat/completions":     "https://example.test/v1/responses",
		"https://example.test/chat/completions":        "https://example.test/responses",
		"https://example.test/v1/models":               "https://example.test/v1/responses",
		"https://example.test/v1/responses":            "https://example.test/v1/responses",
		"https://example.test/v1/chat/completions?x=1": "https://example.test/v1/responses?x=1",
	}
	for raw, want := range tests {
		got, err := normalizeResponsesURL(raw)
		if err != nil {
			t.Fatalf("normalizeResponsesURL(%q): %v", raw, err)
		}
		if got != want {
			t.Fatalf("normalizeResponsesURL(%q) = %q, want %q", raw, got, want)
		}
	}
}

func TestNormalizeModelsURL(t *testing.T) {
	tests := map[string]string{
		"https://example.test":                     "https://example.test/v1/models",
		"https://example.test/v1":                  "https://example.test/v1/models",
		"https://example.test/openai":              "https://example.test/openai/models",
		"https://example.test/v1/chat/completions": "https://example.test/v1/models",
		"https://example.test/v1/responses":        "https://example.test/v1/models",
		"https://example.test/v1/models":           "https://example.test/v1/models",
		"https://example.test/responses":           "https://example.test/models",
		"https://example.test/v1/responses?x=1":    "https://example.test/v1/models",
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

	models                 []string
	modelAuthorizations    []string
	authorizations         []string
	requestBodies          []string
	responseAuthorizations []string
	responseBodies         []string
	chatHandler            http.HandlerFunc
	responsesHandler       http.HandlerFunc
}

func newMockProvider(t *testing.T, models []string, apiKey string, chatHandler http.HandlerFunc) *mockProvider {
	t.Helper()
	return newMockProviderWithResponses(t, models, apiKey, chatHandler, nil)
}

func newMockProviderWithResponses(t *testing.T, models []string, apiKey string, chatHandler http.HandlerFunc, responsesHandler http.HandlerFunc) *mockProvider {
	t.Helper()
	if chatHandler == nil {
		chatHandler = func(w http.ResponseWriter, r *http.Request) {
			t.Fatal("unexpected chat completions request")
		}
	}
	if responsesHandler == nil {
		responsesHandler = func(w http.ResponseWriter, r *http.Request) {
			t.Fatal("unexpected responses request")
		}
	}
	p := &mockProvider{
		t:                t,
		apiKey:           apiKey,
		models:           append([]string(nil), models...),
		chatHandler:      chatHandler,
		responsesHandler: responsesHandler,
	}
	p.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			p.mu.Lock()
			p.modelAuthorizations = append(p.modelAuthorizations, r.Header.Get("Authorization"))
			models := append([]string(nil), p.models...)
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
		case "/v1/responses":
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatal(err)
			}
			p.mu.Lock()
			p.responseAuthorizations = append(p.responseAuthorizations, r.Header.Get("Authorization"))
			p.responseBodies = append(p.responseBodies, string(body))
			p.mu.Unlock()
			responsesHandler(w, r)
		default:
			t.Fatalf("unexpected path = %s", r.URL.Path)
		}
	}))
	return p
}

func (p *mockProvider) setModels(models ...string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.models = append([]string(nil), models...)
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

func (p *mockProvider) responsesCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.responseAuthorizations)
}

func (p *mockProvider) joinAuthorizations() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return strings.Join(p.authorizations, ",")
}

func (p *mockProvider) joinResponseAuthorizations() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return strings.Join(p.responseAuthorizations, ",")
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

func (p *mockProvider) responsesBodies() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.responseBodies...)
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
	return &proxy{pool: pool, client: http.DefaultClient, logger: newTestLogger(), apiKey: testProxyAPIKey, attempts: attempts, delay: delay}
}

func newTestProxyWithLogger(t *testing.T, logger *zap.Logger, providers ...providerConfig) *proxy {
	t.Helper()
	pool := newTestPool(t, providers...)
	return &proxy{pool: pool, client: http.DefaultClient, logger: logger, apiKey: testProxyAPIKey, attempts: 1}
}

func newTestPool(t *testing.T, providers ...providerConfig) *providerPool {
	t.Helper()
	pool, err := newProviderPool(context.Background(), providers, http.DefaultClient, newTestLogger())
	if err != nil {
		t.Fatalf("newProviderPool: %v", err)
	}
	pool.candidateOrder = stableCandidateOrder
	return pool
}

func stableCandidateOrder(candidates []int) []int {
	return append([]int(nil), candidates...)
}

func serveChat(p *proxy, body string) *httptest.ResponseRecorder {
	return serveChatWithHeader(p, body, "", "")
}

func serveChatWithHeader(p *proxy, body string, headerName string, headerValue string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", authorizationHeader(testProxyAPIKey))
	if headerName != "" {
		req.Header.Set(headerName, headerValue)
	}
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)
	return rec
}

func serveResponses(p *proxy, body string) *httptest.ResponseRecorder {
	return serveResponsesWithPath(p, "/v1/responses", body)
}

func serveResponsesWithPath(p *proxy, path string, body string) *httptest.ResponseRecorder {
	return serveResponsesWithPathAndHeader(p, path, body, "", "")
}

func serveResponsesWithPathAndHeader(p *proxy, path string, body string, headerName string, headerValue string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", authorizationHeader(testProxyAPIKey))
	if headerName != "" {
		req.Header.Set(headerName, headerValue)
	}
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)
	return rec
}

func serveModelPath(p *proxy, path string, authorized bool) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	if authorized {
		req.Header.Set("Authorization", authorizationHeader(testProxyAPIKey))
	}
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)
	return rec
}

func serveProviderStatus(p *proxy, path string, authorized bool) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	if authorized {
		req.Header.Set("Authorization", authorizationHeader(testProxyAPIKey))
	}
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)
	return rec
}

func decodeProviderStatus(t *testing.T, rec *httptest.ResponseRecorder) providersStatusResponse {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var got providersStatusResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal provider status: %v", err)
	}
	if got.Object != "list" {
		t.Fatalf("object = %q", got.Object)
	}
	return got
}

func waitForModel(t *testing.T, p *proxy, model string) {
	t.Helper()
	deadline := time.After(2 * time.Second)
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		rec := serveModelPath(p, "/v1/models", true)
		if rec.Code == http.StatusOK && strings.Contains(rec.Body.String(), `"`+model+`"`) {
			return
		}
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for model %q", model)
		case <-ticker.C:
		}
	}
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

func assertLogFieldsDoNotContain(t *testing.T, fields map[string]any, forbidden ...string) {
	t.Helper()
	got := fmt.Sprint(fields)
	for _, value := range forbidden {
		if strings.Contains(got, value) {
			t.Fatalf("log fields leaked %q: %#v", value, fields)
		}
	}
}
