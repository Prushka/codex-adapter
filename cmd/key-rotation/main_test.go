package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"go.uber.org/zap"
)

func TestProxyRetriesNextKeyWithoutChangingBody(t *testing.T) {
	const requestBody = `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}`

	var mu sync.Mutex
	var authorizations []string
	var bodies []string
	var traces []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		authorizations = append(authorizations, r.Header.Get("Authorization"))
		bodies = append(bodies, string(body))
		traces = append(traces, r.Header.Get("X-Trace"))
		mu.Unlock()

		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") == "Bearer key-a" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Upstream", "ok")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-ok"}`))
	}))
	defer upstream.Close()

	proxy := newProxy(proxyConfig{
		ChatURL: upstream.URL + "/v1/chat/completions",
		Keys:    []string{"key-a", "key-b"},
		Client:  upstream.Client(),
		Logger:  zap.NewNop(),
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions?beta=1", strings.NewReader(requestBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Trace", "trace-1")
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)

	res := rec.Result()
	defer res.Body.Close()
	gotBody, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status = %d body = %s", res.StatusCode, gotBody)
	}
	if string(gotBody) != `{"id":"chatcmpl-ok"}` {
		t.Fatalf("body = %s", gotBody)
	}
	if res.Header.Get("X-Upstream") != "ok" {
		t.Fatalf("missing upstream response header: %#v", res.Header)
	}

	mu.Lock()
	defer mu.Unlock()
	if strings.Join(authorizations, ",") != "Bearer key-a,Bearer key-b" {
		t.Fatalf("authorizations = %#v", authorizations)
	}
	for _, body := range bodies {
		if body != requestBody {
			t.Fatalf("forwarded body = %s", body)
		}
	}
	for _, trace := range traces {
		if trace != "trace-1" {
			t.Fatalf("forwarded trace = %q", trace)
		}
	}
}

func TestProxyReturnsLastUpstreamFailureAfterAllKeysFail(t *testing.T) {
	var attempts int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Upstream-Error", "rate-limit")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"quota"}}`))
	}))
	defer upstream.Close()

	proxy := newProxy(proxyConfig{
		ChatURL: upstream.URL + "/v1/chat/completions",
		Keys:    []string{"key-a", "key-b"},
		Client:  upstream.Client(),
		Logger:  zap.NewNop(),
	})

	req := httptest.NewRequest(http.MethodPost, "/chat/completions", strings.NewReader(`{"stream":false}`))
	rec := httptest.NewRecorder()
	proxy.ServeHTTP(rec, req)

	res := rec.Result()
	defer res.Body.Close()
	gotBody, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status = %d body = %s", res.StatusCode, gotBody)
	}
	if string(gotBody) != `{"error":{"message":"quota"}}` {
		t.Fatalf("body = %s", gotBody)
	}
	if res.Header.Get("X-Upstream-Error") != "rate-limit" {
		t.Fatalf("missing upstream error header: %#v", res.Header)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d", attempts)
	}
}

func TestProxyStartsNextRequestWithRotatedKey(t *testing.T) {
	var mu sync.Mutex
	var authorizations []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		authorizations = append(authorizations, r.Header.Get("Authorization"))
		mu.Unlock()
		if r.Header.Get("Authorization") == "Bearer key-a" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	proxy := newProxy(proxyConfig{
		ChatURL: upstream.URL + "/v1/chat/completions",
		Keys:    []string{"key-a", "key-b"},
		Client:  upstream.Client(),
		Logger:  zap.NewNop(),
	})

	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
		rec := httptest.NewRecorder()
		proxy.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("request %d status = %d", i+1, rec.Code)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if strings.Join(authorizations, ",") != "Bearer key-a,Bearer key-b,Bearer key-b" {
		t.Fatalf("authorizations = %#v", authorizations)
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
