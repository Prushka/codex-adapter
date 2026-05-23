package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func TestBuildChatRequestForcesModelReasoningAndMessages(t *testing.T) {
	adapter := testAdapter(t, "http://example.test/v1", nil)
	req := map[string]any{
		"model":            "ignored-model",
		"instructions":     "Be precise.",
		"prompt_cache_key": "cache-123",
		"client_metadata": map[string]any{
			"session": "s-1",
			"attempt": 2,
		},
		"reasoning": map[string]any{"effort": "high"},
		"input": []any{
			map[string]any{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Hello"},
				},
			},
		},
		"stream": true,
	}

	chatReq, _, err := adapter.buildChatRequest(req, true)
	if err != nil {
		t.Fatal(err)
	}
	if got := chatReq["model"]; got != "forced-model" {
		t.Fatalf("model = %v", got)
	}
	if got := chatReq["reasoning_effort"]; got != "low" {
		t.Fatalf("reasoning_effort = %v", got)
	}
	messages := chatReq["messages"].([]map[string]any)
	if len(messages) != 2 {
		t.Fatalf("messages length = %d", len(messages))
	}
	if messages[0]["role"] != "system" || messages[0]["content"] != "Be precise." {
		t.Fatalf("bad system message: %#v", messages[0])
	}
	if messages[1]["role"] != "user" || messages[1]["content"] != "Hello" {
		t.Fatalf("bad user message: %#v", messages[1])
	}
	if chatReq["prompt_cache_key"] != "cache-123" {
		t.Fatalf("prompt_cache_key = %v", chatReq["prompt_cache_key"])
	}
	metadata := chatReq["metadata"].(map[string]string)
	if metadata["session"] != "s-1" || metadata["attempt"] != "2" {
		t.Fatalf("metadata = %#v", metadata)
	}
}

func TestStreamingFunctionCallToResponsesFunctionCall(t *testing.T) {
	var upstreamReq map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("upstream path = %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstreamReq); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"shell_command\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5,\"total_tokens\":9}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools([]any{
		map[string]any{
			"type":        "function",
			"name":        "shell_command",
			"description": "Run a command.",
			"parameters":  objectSchema(),
		},
	})

	events := callResponses(t, adapter, body)
	if upstreamReq["model"] != "forced-model" || upstreamReq["reasoning_effort"] != "low" {
		t.Fatalf("forced upstream fields missing: %#v", upstreamReq)
	}
	item := firstDoneItem(t, events, "function_call")
	if item["name"] != "shell_command" {
		t.Fatalf("function name = %v", item["name"])
	}
	if item["call_id"] != "call-1" {
		t.Fatalf("call_id = %v", item["call_id"])
	}
	if item["arguments"] != "{\"command\":\"pwd\"}" {
		t.Fatalf("arguments = %v", item["arguments"])
	}
	assertCompleted(t, events)
}

func TestNamespaceToolCallRoundTripsNamespace(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		tools := req["tools"].([]any)
		fn := tools[0].(map[string]any)["function"].(map[string]any)
		if fn["name"] != "mcp__rmcp__echo" {
			t.Fatalf("flattened tool name = %v", fn["name"])
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"tool_calls": []any{
							map[string]any{
								"id":   "call-ns",
								"type": "function",
								"function": map[string]any{
									"name":      "mcp__rmcp__echo",
									"arguments": "{\"text\":\"hi\"}",
								},
							},
						},
					},
				},
			},
		})
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools([]any{
		map[string]any{
			"type":        "namespace",
			"name":        "mcp__rmcp__",
			"description": "MCP server",
			"tools": []any{
				map[string]any{
					"type":        "function",
					"name":        "echo",
					"description": "Echo.",
					"parameters":  objectSchema(),
				},
			},
		},
	})

	events := callResponses(t, adapter, body)
	item := firstDoneItem(t, events, "function_call")
	if item["namespace"] != "mcp__rmcp__" || item["name"] != "echo" {
		t.Fatalf("bad namespace item: %#v", item)
	}
}

func TestCustomApplyPatchToolCallRoundTripsAsCustomToolCall(t *testing.T) {
	patch := "*** Begin Patch\n*** Add File: note.txt\n+hi\n*** End Patch\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		tools := req["tools"].([]any)
		fn := tools[0].(map[string]any)["function"].(map[string]any)
		if fn["name"] != "apply_patch" {
			t.Fatalf("custom function name = %v", fn["name"])
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"tool_calls": []any{
							map[string]any{
								"id":   "call-patch",
								"type": "function",
								"function": map[string]any{
									"name":      "apply_patch",
									"arguments": mustJSON(map[string]string{"input": patch}),
								},
							},
						},
					},
				},
			},
		})
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools([]any{
		map[string]any{
			"type":        "custom",
			"name":        "apply_patch",
			"description": "Apply a patch.",
			"format": map[string]any{
				"type":       "grammar",
				"syntax":     "lark",
				"definition": "start: /.+/",
			},
		},
	})

	events := callResponses(t, adapter, body)
	item := firstDoneItem(t, events, "custom_tool_call")
	if item["name"] != "apply_patch" || item["input"] != patch {
		t.Fatalf("bad custom item: %#v", item)
	}
}

func TestToolSearchCallRoundTripsAsClientToolSearch(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"tool_calls": []any{
							map[string]any{
								"id":   "call-search",
								"type": "function",
								"function": map[string]any{
									"name":      "tool_search",
									"arguments": "{\"query\":\"calendar\",\"limit\":3}",
								},
							},
						},
					},
				},
			},
		})
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools([]any{
		map[string]any{
			"type":        "tool_search",
			"execution":   "client",
			"description": "Search tools.",
			"parameters":  objectSchema(),
		},
	})

	events := callResponses(t, adapter, body)
	item := firstDoneItem(t, events, "tool_search_call")
	if item["execution"] != "client" || item["call_id"] != "call-search" {
		t.Fatalf("bad tool search item: %#v", item)
	}
	args := item["arguments"].(map[string]any)
	if args["query"] != "calendar" {
		t.Fatalf("query = %v", args["query"])
	}
}

func TestWebSearchCallRoundTripsAsResponsesWebSearchCall(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"tool_calls": []any{
							map[string]any{
								"id":   "call-web",
								"type": "function",
								"function": map[string]any{
									"name":      "web_search",
									"arguments": "{\"action\":\"search\",\"query\":\"codex responses api\"}",
								},
							},
						},
					},
				},
			},
		})
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools([]any{
		map[string]any{
			"type":                   "web_search",
			"external_web_access":    true,
			"search_context_size":    "medium",
			"search_content_types":   []any{"webpage"},
			"additional_unmodeled_x": "kept out of chat tool schema",
		},
	})

	events := callResponses(t, adapter, body)
	item := firstDoneItem(t, events, "web_search_call")
	if item["status"] != "completed" {
		t.Fatalf("web search status = %v", item["status"])
	}
	action := item["action"].(map[string]any)
	if action["type"] != "search" || action["query"] != "codex responses api" {
		t.Fatalf("bad web search action: %#v", action)
	}
}

func TestCompactEndpointReturnsResponsesOutputItems(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req["stream"] != false {
			t.Fatalf("compact upstream stream = %v", req["stream"])
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"content": "summary",
					},
				},
			},
		})
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	raw, _ := json.Marshal(responsesRequestWithTools(nil))
	req := httptest.NewRequest(http.MethodPost, "/v1/responses/compact", bytes.NewReader(raw))
	rec := httptest.NewRecorder()
	adapter.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var response map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	output := response["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output length = %d", len(output))
	}
	item := output[0].(map[string]any)
	if item["type"] != "message" || item["role"] != "assistant" {
		t.Fatalf("bad compact item: %#v", item)
	}
}

func TestDebugRecorderWritesOrderedJSONFiles(t *testing.T) {
	dir := t.TempDir()
	rec, err := NewDebugRecorder(dir)
	if err != nil {
		t.Fatal(err)
	}
	rec.SaveJSON("first request", map[string]any{"a": 1})
	rec.SaveJSON("second response", map[string]any{"b": 2})

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	sort.Strings(names)
	want := []string{"000001-first-request.json", "000002-second-response.json"}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("debug files = %v", names)
	}
	if _, err := os.Stat(filepath.Join(dir, want[0])); err != nil {
		t.Fatal(err)
	}
}

func testAdapter(t *testing.T, providerURL string, debug *DebugRecorder) *Adapter {
	t.Helper()
	adapter, err := NewAdapter(AdapterConfig{
		ProviderURL:     providerURL,
		Model:           "forced-model",
		ReasoningEffort: "low",
		Debug:           debug,
		HTTPClient:      http.DefaultClient,
	})
	if err != nil {
		t.Fatal(err)
	}
	return adapter
}

func responsesRequestWithTools(tools []any) map[string]any {
	return map[string]any{
		"model":               "ignored",
		"instructions":        "Use tools when needed.",
		"input":               []any{map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "go"}}}},
		"tools":               tools,
		"tool_choice":         "auto",
		"parallel_tool_calls": true,
		"stream":              true,
		"store":               false,
		"include":             []any{},
		"prompt_cache_key":    "thread",
		"client_metadata":     map[string]any{"x": "y"},
		"service_tier":        "default",
		"reasoning":           map[string]any{"effort": "high"},
		"text":                nil,
	}
}

func objectSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"properties":           map[string]any{},
		"additionalProperties": true,
	}
}

func callResponses(t *testing.T, adapter *Adapter, body map[string]any) []map[string]any {
	t.Helper()
	raw, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(raw))
	req.Header.Set("Authorization", "Bearer test")
	rec := httptest.NewRecorder()
	adapter.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	return parseSSEEvents(t, rec.Body.String())
}

func parseSSEEvents(t *testing.T, data string) []map[string]any {
	t.Helper()
	var events []map[string]any
	scanner := bufio.NewScanner(strings.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		var event map[string]any
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			t.Fatalf("bad event %q: %v", payload, err)
		}
		events = append(events, event)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return events
}

func firstDoneItem(t *testing.T, events []map[string]any, typ string) map[string]any {
	t.Helper()
	for _, event := range events {
		if event["type"] != "response.output_item.done" {
			continue
		}
		item := event["item"].(map[string]any)
		if item["type"] == typ {
			return item
		}
	}
	t.Fatalf("missing done item type %s in %#v", typ, events)
	return nil
}

func assertCompleted(t *testing.T, events []map[string]any) {
	t.Helper()
	for _, event := range events {
		if event["type"] == "response.completed" {
			return
		}
	}
	t.Fatalf("missing response.completed in %#v", events)
}

func mustJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return string(data)
}
