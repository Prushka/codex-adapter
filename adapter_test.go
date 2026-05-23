package adapter

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zaptest/observer"
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
	for _, key := range []string{"metadata", "prompt_cache_key", "store", "service_tier"} {
		if _, ok := chatReq[key]; ok {
			t.Fatalf("nonessential provider compatibility field %q was forwarded: %#v", key, chatReq[key])
		}
	}
}

func TestBuildChatRequestFlattensReasoningItems(t *testing.T) {
	adapter := testAdapter(t, "http://example.test/v1", nil)
	req := map[string]any{
		"input": []any{
			map[string]any{
				"type":    "reasoning",
				"summary": []any{},
				"content": []any{
					map[string]any{"type": "reasoning_text", "text": "hidden detail"},
				},
			},
		},
	}

	chatReq, _, err := adapter.buildChatRequest(req, false)
	if err != nil {
		t.Fatal(err)
	}
	messages := chatReq["messages"].([]map[string]any)
	if len(messages) != 1 {
		t.Fatalf("messages length = %d", len(messages))
	}
	if messages[0]["role"] != "assistant" {
		t.Fatalf("role = %v", messages[0]["role"])
	}
	if got := messages[0]["content"]; got != "hidden detail" {
		t.Fatalf("content = %v", got)
	}
	if got, ok := messages[0]["content"].(string); !ok || strings.HasPrefix(got, "[Responses API item]") {
		t.Fatalf("reasoning content still marked: %#v", messages[0]["content"])
	}
}

func TestChatContentFromResponsesContentFlattensReasoningTextTypes(t *testing.T) {
	got := chatContentFromResponsesContent([]any{
		map[string]any{"type": "reasoning_text", "text": "thinking"},
		map[string]any{"type": "summary_text", "text": " summary"},
	}, "assistant")
	if got != "thinking summary" {
		t.Fatalf("content = %#v", got)
	}
}

func TestPostChatForwardsInboundAuthorizationByDefault(t *testing.T) {
	authCh := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authCh <- r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	inbound := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	inbound.Header.Set("Authorization", "Bearer codex-key")
	resp, err := adapter.postChat(inbound, map[string]any{"stream": false})
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()

	if got := <-authCh; got != "Bearer codex-key" {
		t.Fatalf("authorization = %q", got)
	}
}

func TestPostChatConfiguredAPIKeyOverridesInboundAuthorization(t *testing.T) {
	authCh := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authCh <- r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer upstream.Close()

	adapter, err := NewAdapter(AdapterConfig{
		ProviderURL:     upstream.URL,
		Model:           "forced-model",
		ReasoningEffort: "low",
		APIKey:          "upstream-key",
		HTTPClient:      http.DefaultClient,
	})
	if err != nil {
		t.Fatal(err)
	}
	inbound := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	inbound.Header.Set("Authorization", "Bearer codex-key")
	resp, err := adapter.postChat(inbound, map[string]any{"stream": false})
	if err != nil {
		t.Fatal(err)
	}
	_ = resp.Body.Close()

	if got := <-authCh; got != "Bearer upstream-key" {
		t.Fatalf("authorization = %q", got)
	}
}

func TestUpstreamHTTPErrorIncludesProviderDetailsInLogsAndResponse(t *testing.T) {
	core, logs := observer.New(zap.WarnLevel)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("X-Goog-Request-Id", "goog-req-123")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT"}}`))
	}))
	defer upstream.Close()

	adapter, err := NewAdapter(AdapterConfig{
		ProviderURL:     upstream.URL,
		Model:           "forced-model",
		ReasoningEffort: "low",
		HTTPClient:      http.DefaultClient,
		Logger:          zap.New(core),
	})
	if err != nil {
		t.Fatal(err)
	}

	events := callResponses(t, adapter, responsesRequestWithTools(nil))
	failed := firstResponseEvent(t, events, "response.failed")
	response := failed["response"].(map[string]any)
	respErr := response["error"].(map[string]any)
	if got := respErr["message"].(string); !strings.Contains(got, "API key not valid") {
		t.Fatalf("failed response message = %q", got)
	}

	entries := logs.FilterMessage("upstream chat request failed").All()
	if len(entries) != 1 {
		t.Fatalf("log entries = %d", len(entries))
	}
	fields := entries[0].ContextMap()
	if fields["status"] != int64(400) && fields["status"] != 400 {
		t.Fatalf("status field = %#v", fields["status"])
	}
	if got := fields["upstream_error"].(string); !strings.Contains(got, "API key not valid") {
		t.Fatalf("upstream_error = %q", got)
	}
	if got := fields["upstream_response_body"].(string); !strings.Contains(got, "INVALID_ARGUMENT") {
		t.Fatalf("upstream_response_body = %q", got)
	}
	if fields["upstream_request_id"] != "goog-req-123" {
		t.Fatalf("upstream_request_id = %#v", fields["upstream_request_id"])
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
	addedIndex := firstEventIndex(t, events, "response.output_item.added", "function_call")
	deltaIndex := firstEventIndex(t, events, "response.function_call_arguments.delta", "")
	doneIndex := firstEventIndex(t, events, "response.output_item.done", "function_call")
	if !(addedIndex < deltaIndex && deltaIndex < doneIndex) {
		t.Fatalf("bad function call event order: added=%d delta=%d done=%d", addedIndex, deltaIndex, doneIndex)
	}
	if events[deltaIndex]["delta"] != "{\"command\":\"pwd\"}" {
		t.Fatalf("function delta = %v", events[deltaIndex]["delta"])
	}
	assertCompleted(t, events)
}

func TestGeminiExtraContentRoundTripsThroughAdapterCache(t *testing.T) {
	var requestCount atomic.Int32
	extraContent := map[string]any{
		"google": map[string]any{
			"thought_signature": "sig-123",
		},
	}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch requestCount.Add(1) {
		case 1:
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: " + mustJSON(map[string]any{
				"choices": []any{
					map[string]any{
						"delta": map[string]any{
							"tool_calls": []any{
								map[string]any{
									"index":         0,
									"id":            "call-gemini",
									"type":          "function",
									"extra_content": extraContent,
									"function": map[string]any{
										"name":      "exec_command",
										"arguments": "{\"cmd\":\"pwd\"}",
									},
								},
							},
						},
					},
				},
			}) + "\n\n"))
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 2:
			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatal(err)
			}
			got := firstAssistantToolCallExtraContent(t, req)
			gotGoogle := got["google"].(map[string]any)
			if gotGoogle["thought_signature"] != "sig-123" {
				t.Fatalf("thought_signature = %#v", got)
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{"content": "done"},
					},
				},
			})
		default:
			t.Fatalf("unexpected upstream request")
		}
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools([]any{
		map[string]any{
			"type":       "function",
			"name":       "exec_command",
			"parameters": objectSchema(),
		},
	})

	events := callResponses(t, adapter, body)
	item := firstDoneItem(t, events, "function_call")
	emittedExtra := item["extra_content"].(map[string]any)
	if emittedExtra["google"].(map[string]any)["thought_signature"] != "sig-123" {
		t.Fatalf("emitted extra_content = %#v", emittedExtra)
	}

	// Existing Codex builds drop unknown ResponseItem fields. The adapter cache
	// still needs to restore Gemini's extra_content on the follow-up chat request.
	delete(item, "extra_content")
	body["input"] = []any{
		item,
		map[string]any{
			"type":    "function_call_output",
			"call_id": "call-gemini",
			"output":  "ok",
		},
	}
	_ = callResponses(t, adapter, body)
	if requestCount.Load() != 2 {
		t.Fatalf("upstream request count = %d", requestCount.Load())
	}
}

func TestStreamingTextAndReasoningAddBeforeDeltaAndDone(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	events := callResponses(t, adapter, responsesRequestWithTools(nil))

	reasoningAdded := firstEventIndex(t, events, "response.output_item.added", "reasoning")
	reasoningDelta := firstEventIndex(t, events, "response.reasoning_text.delta", "")
	reasoningDone := firstEventIndex(t, events, "response.output_item.done", "reasoning")
	if !(reasoningAdded < reasoningDelta && reasoningDelta < reasoningDone) {
		t.Fatalf("bad reasoning event order: added=%d delta=%d done=%d", reasoningAdded, reasoningDelta, reasoningDone)
	}

	messageAdded := firstEventIndex(t, events, "response.output_item.added", "message")
	textDelta := firstEventIndex(t, events, "response.output_text.delta", "")
	messageDone := firstEventIndex(t, events, "response.output_item.done", "message")
	if !(messageAdded < textDelta && textDelta < messageDone) {
		t.Fatalf("bad message event order: added=%d delta=%d done=%d", messageAdded, textDelta, messageDone)
	}
	if events[reasoningDelta]["delta"] != "thinking" || events[textDelta]["delta"] != "answer" {
		t.Fatalf("bad deltas: %#v %#v", events[reasoningDelta], events[textDelta])
	}
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
	const applyPatchDescription = "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON."
	const applyPatchGrammar = "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\nend_patch: \"*** End Patch\" LF?\n\nhunk: add_hunk | delete_hunk | update_hunk\nadd_hunk: \"*** Add File: \" filename LF add_line+\ndelete_hunk: \"*** Delete File: \" filename LF\nupdate_hunk: \"*** Update File: \" filename LF change_move? change?\n\nfilename: /(.+)/\nadd_line: \"+\" /(.*)/ LF -> line\n\nchange_move: \"*** Move to: \" filename LF\nchange: (change_context | change_line)+ eof_line?\nchange_context: (\"@@\" | \"@@ \" /(.+)/) LF\nchange_line: (\"+\" | \"-\" | \" \") /(.*)/ LF\neof_line: \"*** End of File\" LF\n\n%import common.LF"
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
		if fn["strict"] != true {
			t.Fatalf("custom function strict = %#v", fn["strict"])
		}
		description, _ := fn["description"].(string)
		for _, want := range []string{
			"Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
			"Responses custom tool format:",
			"type: grammar",
			"syntax: lark",
			"start: begin_patch hunk+ end_patch",
			"*** Add File: ",
			"*** End Patch",
			"*** End of File",
			"Call it with a JSON object containing exactly one string field named input.",
		} {
			if !strings.Contains(description, want) {
				t.Fatalf("custom function description missing %q:\n%s", want, description)
			}
		}
		parameters := fn["parameters"].(map[string]any)
		if parameters["additionalProperties"] != false {
			t.Fatalf("custom function parameters.additionalProperties = %#v", parameters["additionalProperties"])
		}
		if got := parameters["required"].([]any); len(got) != 1 || got[0] != "input" {
			t.Fatalf("custom function parameters.required = %#v", got)
		}
		input := parameters["properties"].(map[string]any)["input"].(map[string]any)
		if input["type"] != "string" {
			t.Fatalf("custom function input type = %#v", input["type"])
		}
		if got := input["description"].(string); !strings.Contains(got, "freeform input for the custom tool") {
			t.Fatalf("custom function input description = %q", got)
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
			"description": applyPatchDescription,
			"format": map[string]any{
				"type":       "grammar",
				"syntax":     "lark",
				"definition": applyPatchGrammar,
			},
		},
	})

	events := callResponses(t, adapter, body)
	item := firstDoneItem(t, events, "custom_tool_call")
	if item["name"] != "apply_patch" || item["input"] != patch {
		t.Fatalf("bad custom item: %#v", item)
	}
}

func TestStreamingApplyPatchCustomToolEmitsInputDeltas(t *testing.T) {
	patch1 := "*** Begin Patch\n*** Add File: live.txt\n+live"
	patch2 := " line\n*** End Patch\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: " + mustJSON(map[string]any{
			"choices": []any{
				map[string]any{
					"delta": map[string]any{
						"tool_calls": []any{
							map[string]any{
								"index": 0,
								"id":    "patch-call",
								"type":  "function",
								"function": map[string]any{
									"name":      "apply_patch",
									"arguments": `{"input":"` + strings.ReplaceAll(patch1, "\n", `\n`),
								},
							},
						},
					},
				},
			},
		}) + "\n\n"))
		_, _ = w.Write([]byte("data: " + mustJSON(map[string]any{
			"choices": []any{
				map[string]any{
					"delta": map[string]any{
						"tool_calls": []any{
							map[string]any{
								"index": 0,
								"function": map[string]any{
									"arguments": strings.ReplaceAll(patch2, "\n", `\n`) + `"}`,
								},
							},
						},
					},
				},
			},
		}) + "\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools([]any{
		map[string]any{
			"type":        "custom",
			"name":        "apply_patch",
			"description": "Apply a patch.",
		},
	})

	events := callResponses(t, adapter, body)
	addedIndex := firstEventIndex(t, events, "response.output_item.added", "custom_tool_call")
	firstDelta := firstEventIndex(t, events, "response.custom_tool_call_input.delta", "")
	doneIndex := firstEventIndex(t, events, "response.output_item.done", "custom_tool_call")
	if !(addedIndex < firstDelta && firstDelta < doneIndex) {
		t.Fatalf("bad custom tool event order: added=%d delta=%d done=%d", addedIndex, firstDelta, doneIndex)
	}

	var streamed strings.Builder
	for _, event := range events {
		if event["type"] == "response.custom_tool_call_input.delta" {
			streamed.WriteString(event["delta"].(string))
		}
	}
	item := firstDoneItem(t, events, "custom_tool_call")
	if streamed.String() != patch1+patch2 || item["input"] != patch1+patch2 {
		t.Fatalf("bad streamed patch: streamed=%q item=%#v", streamed.String(), item)
	}
}

func TestStreamingInterleavedToolCallsDoneAfterAllArgumentDeltas(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-a\",\"type\":\"function\",\"function\":{\"name\":\"first_tool\",\"arguments\":\"{\\\"a\\\":\"}},{\"index\":1,\"id\":\"call-b\",\"type\":\"function\",\"function\":{\"name\":\"second_tool\",\"arguments\":\"{\\\"b\\\":\"}}]}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}},{\"index\":1,\"function\":{\"arguments\":\"2}\"}}]}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools([]any{
		map[string]any{"type": "function", "name": "first_tool", "parameters": objectSchema()},
		map[string]any{"type": "function", "name": "second_tool", "parameters": objectSchema()},
	})

	events := callResponses(t, adapter, body)
	var done []map[string]any
	for _, event := range events {
		if event["type"] == "response.output_item.done" {
			item := event["item"].(map[string]any)
			if item["type"] == "function_call" {
				done = append(done, item)
			}
		}
	}
	if len(done) != 2 {
		t.Fatalf("done function calls = %#v", done)
	}
	if done[0]["arguments"] != "{\"a\":1}" || done[1]["arguments"] != "{\"b\":2}" {
		t.Fatalf("bad interleaved arguments: %#v", done)
	}
}

func TestStreamingLengthFinishReasonEmitsIncomplete(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	events := callResponses(t, adapter, responsesRequestWithTools(nil))
	for _, event := range events {
		if event["type"] == "response.completed" {
			t.Fatalf("unexpected completed event for length finish: %#v", events)
		}
	}
	for _, event := range events {
		if event["type"] != "response.incomplete" {
			continue
		}
		response := event["response"].(map[string]any)
		details := response["incomplete_details"].(map[string]any)
		if details["reason"] != "max_output_tokens" {
			t.Fatalf("incomplete reason = %v", details["reason"])
		}
		return
	}
	t.Fatalf("missing response.incomplete in %#v", events)
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

func TestToolSearchOutputRegistersDiscoveredNamespaceToolsForChatFollowUp(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		tools := req["tools"].([]any)
		var names []string
		for _, raw := range tools {
			tool := raw.(map[string]any)
			fn := tool["function"].(map[string]any)
			names = append(names, fn["name"].(string))
		}
		if !stringSliceContains(names, "tool_search") {
			t.Fatalf("missing tool_search in %v", names)
		}
		if !stringSliceContains(names, "multi_agent_v1_spawn_agent") {
			t.Fatalf("missing discovered namespace child in %v", names)
		}
		messages := req["messages"].([]any)
		toolMsg := messages[len(messages)-2].(map[string]any)
		if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "search-1" {
			t.Fatalf("bad tool_search_output chat message: %#v", toolMsg)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"tool_calls": []any{
							map[string]any{
								"id":   "call-spawn",
								"type": "function",
								"function": map[string]any{
									"name":      "multi_agent_v1_spawn_agent",
									"arguments": "{\"message\":\"Explore the repo\",\"fork_context\":true}",
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
	body["input"] = []any{
		map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": "Find spawn agent tooling."}},
		},
		map[string]any{
			"type":      "tool_search_call",
			"call_id":   "search-1",
			"execution": "client",
			"arguments": map[string]any{"query": "spawn agent", "limit": 5},
		},
		map[string]any{
			"type":      "tool_search_output",
			"call_id":   "search-1",
			"execution": "client",
			"status":    "completed",
			"tools": []any{
				map[string]any{
					"type":        "namespace",
					"name":        "multi_agent_v1",
					"description": "Tools for spawning and managing sub-agents.",
					"tools": []any{
						map[string]any{
							"type":          "function",
							"name":          "spawn_agent",
							"description":   "Spawn a sub-agent for a well-scoped task.",
							"defer_loading": true,
							"parameters": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"message":      map[string]any{"type": "string"},
									"fork_context": map[string]any{"type": "boolean"},
								},
								"required":             []any{"message"},
								"additionalProperties": false,
							},
						},
					},
				},
			},
		},
		map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": "Spawn one agent."}},
		},
	}

	events := callResponses(t, adapter, body)
	item := firstDoneItem(t, events, "function_call")
	if item["namespace"] != "multi_agent_v1" || item["name"] != "spawn_agent" {
		t.Fatalf("bad discovered namespace call: %#v", item)
	}
	if item["call_id"] != "call-spawn" {
		t.Fatalf("call_id = %v", item["call_id"])
	}
}

func TestStreamingDiscoveredToolCallsWithoutIndexesStaySeparate(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		tools := req["tools"].([]any)
		var names []string
		for _, raw := range tools {
			tool := raw.(map[string]any)
			fn := tool["function"].(map[string]any)
			names = append(names, fn["name"].(string))
		}
		if !stringSliceContains(names, "multi_agent_v1_spawn_agent") {
			t.Fatalf("missing discovered spawn tool in %v", names)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		for _, call := range []struct {
			id      string
			message string
		}{
			{id: "call-spawn-a", message: "Explore adapter.go."},
			{id: "call-spawn-b", message: "Explore sse.go."},
			{id: "call-spawn-c", message: "Explore README.md."},
		} {
			_, _ = w.Write([]byte("data: " + mustJSON(map[string]any{
				"choices": []any{
					map[string]any{
						"delta": map[string]any{
							"tool_calls": []any{
								map[string]any{
									"id":   call.id,
									"type": "function",
									"function": map[string]any{
										"name": "multi_agent_v1_spawn_agent",
										"arguments": mustJSON(map[string]any{
											"agent_type": "explorer",
											"message":    call.message,
										}),
									},
								},
							},
						},
					},
				},
			}) + "\n\n"))
		}
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
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
	body["input"] = []any{
		map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": "Find spawn agent tooling."}},
		},
		map[string]any{
			"type":      "tool_search_call",
			"call_id":   "search-1",
			"execution": "client",
			"arguments": map[string]any{"query": "spawn agent", "limit": 5},
		},
		map[string]any{
			"type":      "tool_search_output",
			"call_id":   "search-1",
			"execution": "client",
			"status":    "completed",
			"tools": []any{
				map[string]any{
					"type":        "namespace",
					"name":        "multi_agent_v1",
					"description": "Tools for spawning and managing sub-agents.",
					"tools": []any{
						map[string]any{
							"type":        "function",
							"name":        "spawn_agent",
							"description": "Spawn a sub-agent for a well-scoped task.",
							"parameters": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"message":    map[string]any{"type": "string"},
									"agent_type": map[string]any{"type": "string"},
								},
								"additionalProperties": false,
							},
						},
					},
				},
			},
		},
		map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": "Spawn 3 agents."}},
		},
	}

	events := callResponses(t, adapter, body)
	var calls []map[string]any
	for _, event := range events {
		if event["type"] != "response.output_item.done" {
			continue
		}
		item := event["item"].(map[string]any)
		if item["type"] == "function_call" {
			calls = append(calls, item)
		}
	}
	if len(calls) != 3 {
		t.Fatalf("function calls = %#v", calls)
	}
	for i, call := range calls {
		if call["namespace"] != "multi_agent_v1" || call["name"] != "spawn_agent" {
			t.Fatalf("bad discovered call %d: %#v", i, call)
		}
		args := jsonObject(call["arguments"].(string))
		if args["message"] == "" {
			t.Fatalf("call %d missing message args: %#v", i, call)
		}
	}
	if calls[0]["call_id"] != "call-spawn-a" || calls[1]["call_id"] != "call-spawn-b" || calls[2]["call_id"] != "call-spawn-c" {
		t.Fatalf("call ids = %#v", calls)
	}
}

func TestUnknownHistoricalToolCallIsNotForwardedAsChatToolCall(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		messages := req["messages"].([]any)
		for _, raw := range messages {
			msg := raw.(map[string]any)
			if _, ok := msg["tool_calls"]; ok {
				t.Fatalf("unknown historical call was forwarded as chat tool call: %#v", msg)
			}
			if msg["role"] == "tool" {
				t.Fatalf("unknown historical output was forwarded as chat tool output: %#v", msg)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{"content": "recovered"},
				},
			},
		})
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools(nil)
	body["input"] = []any{
		map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": "Spawn agents."}},
		},
		map[string]any{
			"type":      "function_call",
			"call_id":   "bad-wrapper",
			"name":      "multi_agent_v1",
			"arguments": "{\"spawn_agent\":{\"message\":\"one\"}}{\"spawn_agent\":{\"message\":\"two\"}}",
		},
		map[string]any{
			"type":    "function_call_output",
			"call_id": "bad-wrapper",
			"output":  "unsupported call: multi_agent_v1",
		},
	}

	events := callResponses(t, adapter, body)
	message := firstDoneItem(t, events, "message")
	content := message["content"].([]any)
	if got := content[0].(map[string]any)["text"].(string); got != "recovered" {
		t.Fatalf("final message = %q", got)
	}
}

func TestMalformedHistoricalToolCallIsNotForwardedAsChatToolCall(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		messages := req["messages"].([]any)
		for _, raw := range messages {
			msg := raw.(map[string]any)
			if _, ok := msg["tool_calls"]; ok {
				t.Fatalf("malformed historical call was forwarded as chat tool call: %#v", msg)
			}
			if msg["role"] == "tool" {
				t.Fatalf("malformed historical output was forwarded as chat tool output: %#v", msg)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{"content": "recovered"},
				},
			},
		})
	}))
	defer upstream.Close()

	adapter := testAdapter(t, upstream.URL, nil)
	body := responsesRequestWithTools([]any{
		map[string]any{
			"type":       "function",
			"name":       "known_tool",
			"parameters": objectSchema(),
		},
	})
	body["input"] = []any{
		map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": "Run tools."}},
		},
		map[string]any{
			"type":      "function_call",
			"call_id":   "bad-known",
			"name":      "known_tool",
			"arguments": "{\"a\":1}{\"a\":2}",
		},
		map[string]any{
			"type":    "function_call_output",
			"call_id": "bad-known",
			"output":  "failed to parse function arguments: trailing characters",
		},
	}

	events := callResponses(t, adapter, body)
	message := firstDoneItem(t, events, "message")
	content := message["content"].([]any)
	if got := content[0].(map[string]any)["text"].(string); got != "recovered" {
		t.Fatalf("final message = %q", got)
	}
}

func TestWebSearchCallTriggersSearchFollowUpAndFinalAnswer(t *testing.T) {
	var upstreamHits atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit := upstreamHits.Add(1)
		switch hit {
		case 1:
			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatal(err)
			}
			if req["stream"] != true {
				t.Fatalf("first upstream request stream = %#v", req["stream"])
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call-web\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"action\\\":\\\"search\\\",\\\"query\\\":\\\"Tesla stock price\\\"}\"},\"extra_content\":{\"google\":{\"thought_signature\":\"sig-123\"}}}]}}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5,\"total_tokens\":9}}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 2:
			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatal(err)
			}
			messages := req["messages"].([]any)
			if len(messages) < 3 {
				t.Fatalf("follow-up messages = %#v", messages)
			}
			assistantMsg := messages[len(messages)-2].(map[string]any)
			calls := assistantMsg["tool_calls"].([]any)
			extra := calls[0].(map[string]any)["extra_content"].(map[string]any)
			if got := extra["google"].(map[string]any)["thought_signature"]; got != "sig-123" {
				t.Fatalf("follow-up thought_signature = %#v", got)
			}
			toolMsg := messages[len(messages)-1].(map[string]any)
			if toolMsg["role"] != "tool" {
				t.Fatalf("follow-up tool message = %#v", toolMsg)
			}
			if got := toolMsg["content"].(string); !strings.Contains(got, "TSLA trading near $123.45") {
				t.Fatalf("follow-up tool content = %q", got)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Tesla is trading near $123.45.\"}}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":12,\"total_tokens\":22}}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			t.Fatalf("unexpected upstream hit %d", hit)
		}
	}))
	defer upstream.Close()

	searcher := &mockWebSearcher{
		results: []searchResult{
			{
				Title:   "Tesla stock price - Example Finance",
				URL:     "https://finance.example.com/tsla",
				Snippet: "TSLA trading near $123.45 with a positive move.",
			},
		},
	}

	adapter := testAdapterWithClientAndSearcher(t, upstream.URL, nil, http.DefaultClient, searcher)
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
	if action["type"] != "search" || action["query"] != "Tesla stock price" {
		t.Fatalf("bad web search action: %#v", action)
	}
	message := firstDoneItem(t, events, "message")
	content := message["content"].([]any)
	if got := content[0].(map[string]any)["text"].(string); !strings.Contains(got, "Tesla is trading near $123.45.") {
		t.Fatalf("final assistant text = %q", got)
	}
	assertCompleted(t, events)
	if upstreamHits.Load() != 2 {
		t.Fatalf("upstream hits = %d", upstreamHits.Load())
	}
	if len(searcher.queries) != 1 || searcher.queries[0] != "Tesla stock price" {
		t.Fatalf("search queries = %#v", searcher.queries)
	}
}

func TestWebSearchBackendErrorBecomesToolContent(t *testing.T) {
	var upstreamHits atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit := upstreamHits.Add(1)
		switch hit {
		case 1:
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call-web\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"generic query\\\"}\"}}]}}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case 2:
			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatal(err)
			}
			messages := req["messages"].([]any)
			toolMsg := messages[len(messages)-1].(map[string]any)
			if toolMsg["role"] != "tool" {
				t.Fatalf("follow-up tool message = %#v", toolMsg)
			}
			content := toolMsg["content"].(string)
			if !strings.Contains(content, "Search backend errors") || !strings.Contains(content, "backend offline") {
				t.Fatalf("tool content = %q", content)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"I could not retrieve search results.\"}}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			t.Fatalf("unexpected upstream hit %d", hit)
		}
	}))
	defer upstream.Close()

	searcher := &mockWebSearcher{err: errors.New("backend offline")}
	adapter := testAdapterWithClientAndSearcher(t, upstream.URL, nil, http.DefaultClient, searcher)
	body := responsesRequestWithTools([]any{
		map[string]any{
			"type":                "web_search",
			"external_web_access": true,
		},
	})

	events := callResponses(t, adapter, body)
	message := firstDoneItem(t, events, "message")
	content := message["content"].([]any)
	if got := content[0].(map[string]any)["text"].(string); !strings.Contains(got, "could not retrieve") {
		t.Fatalf("final assistant text = %q", got)
	}
	assertCompleted(t, events)
	if upstreamHits.Load() != 2 {
		t.Fatalf("upstream hits = %d", upstreamHits.Load())
	}
}

func TestDuckDuckGoSearchBackendParsesLiteHTML(t *testing.T) {
	duckHTML := `<!doctype html>
<html>
  <body>
    <table>
      <tr class="result">
        <td class="result-snippet">
          <a class="result-link" href="/l/?uddg=https%3A%2F%2Ffinance.example.com%2Ftsla">Tesla stock price - Example Finance</a>
          <div class="result-snippet">TSLA trading near $123.45 with a positive move.</div>
        </td>
      </tr>
    </table>
  </body>
</html>`

	client := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			if strings.Contains(req.URL.Host, "duckduckgo.com") {
				return &http.Response{
					StatusCode: http.StatusOK,
					Status:     "200 OK",
					Header:     http.Header{"Content-Type": []string{"text/html; charset=utf-8"}},
					Body:       io.NopCloser(strings.NewReader(duckHTML)),
				}, nil
			}
			return http.DefaultTransport.RoundTrip(req)
		}),
	}

	searcher := newDuckDuckGoSearcher(client, zap.NewNop())
	results, err := searcher.Search(context.Background(), "Tesla stock price", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %#v", results)
	}
	if results[0].URL != "https://finance.example.com/tsla" {
		t.Fatalf("result url = %q", results[0].URL)
	}
	if !strings.Contains(results[0].Snippet, "TSLA trading near $123.45") {
		t.Fatalf("result snippet = %q", results[0].Snippet)
	}
}

func TestBingSearchBackendParsesHTMLResults(t *testing.T) {
	bingHTML := `<!doctype html>
<html>
  <body>
    <ol id="b_results">
      <li class="b_algo">
        <h2><a href="https://www.bing.com/ck/a?!&amp;u=a1aHR0cHM6Ly9jb2RlLnZpc3VhbHN0dWRpby5jb20v&amp;ntb=1">Visual Studio Code</a></h2>
        <div class="b_caption"><p>A lightweight source code editor for building software.</p></div>
      </li>
    </ol>
  </body>
</html>`

	client := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			if strings.Contains(req.URL.Host, "bing.com") {
				return &http.Response{
					StatusCode: http.StatusOK,
					Status:     "200 OK",
					Header:     http.Header{"Content-Type": []string{"text/html; charset=utf-8"}},
					Body:       io.NopCloser(strings.NewReader(bingHTML)),
				}, nil
			}
			return http.DefaultTransport.RoundTrip(req)
		}),
	}

	searcher := newBingSearcher(client, zap.NewNop())
	results, err := searcher.Search(context.Background(), "open source code editor", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %#v", results)
	}
	if results[0].URL != "https://code.visualstudio.com/" {
		t.Fatalf("result url = %q", results[0].URL)
	}
	if !strings.Contains(results[0].Snippet, "source code editor") {
		t.Fatalf("result snippet = %q", results[0].Snippet)
	}
}

func TestYahooSearchBackendParsesHTMLResults(t *testing.T) {
	yahooHTML := `<!doctype html>
<html>
  <body>
    <div id="web">
      <ol class="reg searchCenterMiddle">
        <li>
          <div class="dd algo algo-sr">
            <div class="compTitle">
              <a href="https://r.search.yahoo.com/_ylt=abc/RU=https%3a%2f%2fcode.visualstudio.com%2f/RK=2/RS=def">
                <h3 class="title"><span>Visual Studio Code - Code Editing</span></h3>
              </a>
            </div>
            <div class="compText"><p>AI-powered editing and completions in one code editor.</p></div>
          </div>
        </li>
      </ol>
    </div>
  </body>
</html>`

	client := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			if strings.Contains(req.URL.Host, "yahoo.com") {
				return &http.Response{
					StatusCode: http.StatusOK,
					Status:     "200 OK",
					Header:     http.Header{"Content-Type": []string{"text/html; charset=utf-8"}},
					Body:       io.NopCloser(strings.NewReader(yahooHTML)),
				}, nil
			}
			return http.DefaultTransport.RoundTrip(req)
		}),
	}

	searcher := newYahooSearcher(client, zap.NewNop())
	results, err := searcher.Search(context.Background(), "open source code editor", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %#v", results)
	}
	if results[0].URL != "https://code.visualstudio.com/" {
		t.Fatalf("result url = %q", results[0].URL)
	}
	if !strings.Contains(results[0].Snippet, "AI-powered editing") {
		t.Fatalf("result snippet = %q", results[0].Snippet)
	}
}

func TestDefaultSearchBackendFallsBackAfterChallenge(t *testing.T) {
	bingHTML := `<!doctype html>
<html><body><ol id="b_results"><li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&amp;u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9yZXN1bHQ=&amp;ntb=1">Generic result</a></h2><div class="b_caption"><p>Recovered from the fallback backend.</p></div></li></ol></body></html>`
	challengeHTML := `<!doctype html><html><body>Verify you are human. CAPTCHA required.</body></html>`

	client := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			switch {
			case strings.Contains(req.URL.Host, "duckduckgo.com"):
				return &http.Response{
					StatusCode: http.StatusOK,
					Status:     "200 OK",
					Header:     http.Header{"Content-Type": []string{"text/html; charset=utf-8"}},
					Body:       io.NopCloser(strings.NewReader(challengeHTML)),
				}, nil
			case strings.Contains(req.URL.Host, "bing.com"):
				return &http.Response{
					StatusCode: http.StatusOK,
					Status:     "200 OK",
					Header:     http.Header{"Content-Type": []string{"text/html; charset=utf-8"}},
					Body:       io.NopCloser(strings.NewReader(bingHTML)),
				}, nil
			case strings.Contains(req.URL.Host, "yahoo.com"):
				t.Fatalf("yahoo should not be called after bing succeeds")
			}
			return http.DefaultTransport.RoundTrip(req)
		}),
	}

	searcher, err := NewWebSearcher(WebSearchConfig{Provider: "duckduckgo"}, client, zap.NewNop())
	if err != nil {
		t.Fatal(err)
	}
	results, err := searcher.Search(context.Background(), "generic query", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %#v", results)
	}
	if results[0].URL != "https://example.com/result" {
		t.Fatalf("result url = %q", results[0].URL)
	}
}

func TestNormalizeSearchResultURLDecodesSearchRedirects(t *testing.T) {
	cases := map[string]string{
		"https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fduck":                      "https://example.com/duck",
		"https://www.bing.com/ck/a?!&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9iaW5n&ntb=1":               "https://example.com/bing",
		"https://r.search.yahoo.com/_ylt=abc/RU=https%3a%2f%2fexample.com%2fyahoo/RK=2/RS=def": "https://example.com/yahoo",
	}
	for raw, want := range cases {
		if got := normalizeSearchResultURL(raw); got != want {
			t.Fatalf("normalizeSearchResultURL(%q) = %q, want %q", raw, got, want)
		}
	}
}

func TestSearxngSearchBackendParsesJSONResults(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("format"); got != "json" {
			t.Fatalf("format = %q", got)
		}
		if got := r.URL.Query().Get("q"); got != "Tesla stock price" {
			t.Fatalf("q = %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"results": []map[string]any{
				{
					"title":   "Tesla stock price - Example Finance",
					"url":     "https://finance.example.com/tsla",
					"content": "TSLA trading near $123.45 with a positive move.",
				},
			},
		})
	}))
	defer upstream.Close()

	searcher, err := NewWebSearcher(WebSearchConfig{
		Provider: "searxng",
		Endpoint: upstream.URL + "/search",
	}, upstream.Client(), zap.NewNop())
	if err != nil {
		t.Fatal(err)
	}
	results, err := searcher.Search(context.Background(), "Tesla stock price", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %#v", results)
	}
	if results[0].URL != "https://finance.example.com/tsla" {
		t.Fatalf("result url = %q", results[0].URL)
	}
	if !strings.Contains(results[0].Snippet, "TSLA trading near $123.45") {
		t.Fatalf("result snippet = %q", results[0].Snippet)
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
	return testAdapterWithClient(t, providerURL, debug, http.DefaultClient)
}

func testAdapterWithClient(t *testing.T, providerURL string, debug *DebugRecorder, client *http.Client) *Adapter {
	t.Helper()
	return testAdapterWithClientAndSearcher(t, providerURL, debug, client, nil)
}

func testAdapterWithClientAndSearcher(t *testing.T, providerURL string, debug *DebugRecorder, client *http.Client, searcher WebSearcher) *Adapter {
	t.Helper()
	adapter, err := NewAdapter(AdapterConfig{
		ProviderURL:     providerURL,
		Model:           "forced-model",
		ReasoningEffort: "low",
		WebSearcher:     searcher,
		Debug:           debug,
		HTTPClient:      client,
	})
	if err != nil {
		t.Fatal(err)
	}
	return adapter
}

type mockWebSearcher struct {
	results []searchResult
	queries []string
	err     error
}

func (m *mockWebSearcher) Name() string {
	return "mock"
}

func (m *mockWebSearcher) Search(_ context.Context, query string, limit int) ([]searchResult, error) {
	m.queries = append(m.queries, query)
	if m.err != nil {
		return nil, m.err
	}
	results := append([]searchResult(nil), m.results...)
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
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

func firstResponseEvent(t *testing.T, events []map[string]any, typ string) map[string]any {
	t.Helper()
	for _, event := range events {
		if event["type"] == typ {
			return event
		}
	}
	t.Fatalf("missing response event type %s in %#v", typ, events)
	return nil
}

func firstAssistantToolCallExtraContent(t *testing.T, req map[string]any) map[string]any {
	t.Helper()
	messages, ok := req["messages"].([]any)
	if !ok {
		t.Fatalf("messages missing in %#v", req)
	}
	for _, raw := range messages {
		message, ok := raw.(map[string]any)
		if !ok || message["role"] != "assistant" {
			continue
		}
		calls, ok := message["tool_calls"].([]any)
		if !ok || len(calls) == 0 {
			continue
		}
		call, ok := calls[0].(map[string]any)
		if !ok {
			continue
		}
		extra, ok := call["extra_content"].(map[string]any)
		if !ok {
			t.Fatalf("extra_content missing on tool call: %#v", call)
		}
		return extra
	}
	t.Fatalf("assistant tool call missing in %#v", req)
	return nil
}

func firstEventIndex(t *testing.T, events []map[string]any, eventType, itemType string) int {
	t.Helper()
	for i, event := range events {
		if event["type"] != eventType {
			continue
		}
		if itemType == "" {
			return i
		}
		item, _ := event["item"].(map[string]any)
		if item["type"] == itemType {
			return i
		}
	}
	t.Fatalf("missing event type=%s item_type=%s in %#v", eventType, itemType, events)
	return -1
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

func stringSliceContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func mustJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return string(data)
}
