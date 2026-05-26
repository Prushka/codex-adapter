package adapter

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

const (
	maxUpstreamErrorBodyLogBytes  = 16 << 10
	maxUpstreamErrorMessageBytes  = 4 << 10
	maxToolExtraContentEntries    = 40960
	maxMessageExtraContentEntries = 40960
	maxWebSearchHistoryEntries    = 40960

	reasoningHistoryAuto             = "auto"
	reasoningHistoryDrop             = "drop"
	reasoningHistoryReasoningContent = "reasoning-content"
	reasoningHistoryAssistantContent = "assistant-content"
)

type AdapterConfig struct {
	ProviderURL              string
	Model                    string
	ReasoningEffort          string
	ReasoningHistory         string
	APIKey                   string
	DisableUpstreamStreaming bool
	WebSearcher              WebSearcher
	Debug                    *DebugRecorder
	HTTPClient               *http.Client
	Logger                   *zap.Logger
}

type Adapter struct {
	chatURL                  string
	model                    string
	reasoningEffort          string
	reasoningHistory         string
	apiKey                   string
	disableUpstreamStreaming bool
	search                   WebSearcher
	debug                    *DebugRecorder
	client                   *http.Client
	logger                   *zap.Logger
	extraMu                  sync.Mutex
	extraByCallID            map[string]any
	extraOrder               []string
	messageExtraByKey        map[string][]any
	messageExtraOrder        []string
	webSearchByKey           map[string][]webSearchHistoryEntry
	webSearchOrder           []string
}

type webSearchHistoryEntry struct {
	CallID       string
	Name         string
	Arguments    string
	Result       string
	ExtraContent any
}

func NewAdapter(cfg AdapterConfig) (*Adapter, error) {
	if cfg.ProviderURL == "" {
		return nil, errors.New("provider URL is required")
	}
	if cfg.Model == "" {
		return nil, errors.New("model is required")
	}
	if cfg.ReasoningEffort == "" {
		return nil, errors.New("reasoning effort is required")
	}
	reasoningHistory, err := resolveReasoningHistoryMode(cfg.ReasoningHistory, cfg.Model)
	if err != nil {
		return nil, err
	}
	chatURL, err := normalizeChatCompletionsURL(cfg.ProviderURL)
	if err != nil {
		return nil, err
	}
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}
	search := cfg.WebSearcher
	if search == nil {
		search = newGenericWebSearcher(client, logger, false)
	}
	return &Adapter{
		chatURL:                  chatURL,
		model:                    cfg.Model,
		reasoningEffort:          cfg.ReasoningEffort,
		reasoningHistory:         reasoningHistory,
		apiKey:                   strings.TrimSpace(cfg.APIKey),
		disableUpstreamStreaming: cfg.DisableUpstreamStreaming,
		search:                   search,
		debug:                    cfg.Debug,
		client:                   client,
		logger:                   logger,
		extraByCallID:            map[string]any{},
		messageExtraByKey:        map[string][]any{},
		webSearchByKey:           map[string][]webSearchHistoryEntry{},
	}, nil
}

func (a *Adapter) ChatCompletionsURL() string {
	return a.chatURL
}

func resolveReasoningHistoryMode(raw, model string) (string, error) {
	mode := strings.TrimSpace(strings.ToLower(raw))
	if mode == "" {
		mode = reasoningHistoryAuto
	}
	switch mode {
	case reasoningHistoryAuto:
		if modelUsesKimiReasoningContent(model) {
			return reasoningHistoryReasoningContent, nil
		}
		return reasoningHistoryDrop, nil
	case reasoningHistoryDrop, reasoningHistoryReasoningContent, reasoningHistoryAssistantContent:
		return mode, nil
	default:
		return "", fmt.Errorf("invalid reasoning history mode %q: expected auto, drop, reasoning-content, or assistant-content", raw)
	}
}

func modelUsesKimiReasoningContent(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "kimi-k2-thinking") || strings.Contains(model, "kimi-k2.6")
}

func modelNeedsPreservedThinking(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "kimi-k2.6")
}

func (a *Adapter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimRight(r.URL.Path, "/")
	if path == "" {
		path = "/"
	}
	switch path {
	case "/healthz":
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	case "/models", "/v1/models":
		a.handleModels(w, r)
	case "/responses", "/v1/responses":
		a.handleResponses(w, r)
	case "/responses/compact", "/v1/responses/compact":
		a.handleCompact(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (a *Adapter) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data": []any{
			map[string]any{
				"id":       a.model,
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "codex-adapter",
			},
		},
	})
}

func (a *Adapter) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := a.readRequestBody(r)
	if err != nil {
		a.logger.Warn("failed to read responses request", zap.String("path", r.URL.Path), zap.Error(err))
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	a.debug.SaveRawJSON("inbound responses request", body)

	respID := newID("resp")
	sse := newResponseSSEWriter(w, a.debug)
	_ = sse.Event("response.created", map[string]any{
		"response": map[string]any{
			"id":         respID,
			"object":     "response",
			"created_at": time.Now().Unix(),
			"status":     "in_progress",
			"model":      a.model,
			"output":     []any{},
		},
	})

	var responsesReq map[string]any
	if err := json.Unmarshal(body, &responsesReq); err != nil {
		a.logger.Warn("invalid responses request json", zap.String("response_id", respID), zap.Error(err))
		_ = sse.Event("response.failed", failedResponse(respID, "invalid_request_error", "invalid_json", err.Error()))
		return
	}

	streamUpstream := !a.disableUpstreamStreaming
	chatReq, ctx, err := a.buildChatRequest(responsesReq, streamUpstream)
	if err != nil {
		a.logger.Warn("failed to translate responses request", zap.String("response_id", respID), zap.Error(err))
		_ = sse.Event("response.failed", failedResponse(respID, "invalid_request_error", "translation_error", err.Error()))
		return
	}
	a.debug.SaveJSON("upstream chat request", chatReq)

	upstream, err := a.postChat(r, chatReq)
	if err != nil {
		a.logger.Error("failed to send upstream chat request",
			zap.String("response_id", respID),
			zap.String("upstream_url", a.chatURL),
			zap.Error(err),
		)
		_ = sse.Event("response.failed", failedResponse(respID, "server_error", "upstream_request_error", err.Error()))
		return
	}
	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {
			a.logger.Warn("failed to close response body", zap.String("response_id", respID), zap.Error(err))
		}
	}(upstream.Body)

	if upstream.StatusCode < 200 || upstream.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(upstream.Body, 4<<20))
		a.debug.SaveRawJSON("upstream chat error", raw)
		errInfo := newUpstreamErrorInfo(upstream.StatusCode, upstream.Header, raw)
		a.logger.Warn("upstream chat request failed",
			errInfo.logFields(respID, a.chatURL)...,
		)
		_ = sse.Event("response.failed", failedResponse(respID, "server_error", "upstream_http_error", errInfo.message))
		return
	}

	if isEventStream(upstream.Header.Get("Content-Type")) {
		if err := a.translateChatStream(r, upstream.Body, chatReq, ctx, sse, respID, 0); err != nil {
			a.logger.Error("failed to translate upstream chat stream", zap.String("response_id", respID), zap.Error(err))
			_ = sse.Event("response.failed", failedResponse(respID, "server_error", "stream_translation_error", err.Error()))
		}
		return
	}

	raw, err := io.ReadAll(upstream.Body)
	if err != nil {
		a.logger.Error("failed to read upstream chat response", zap.String("response_id", respID), zap.Error(err))
		_ = sse.Event("response.failed", failedResponse(respID, "server_error", "upstream_read_error", err.Error()))
		return
	}
	a.debug.SaveRawJSON("upstream chat response", raw)
	gen, err := generationFromChatResponse(raw, ctx)
	if err != nil {
		a.logger.Error("failed to translate upstream chat response", zap.String("response_id", respID), zap.Error(err))
		_ = sse.Event("response.failed", failedResponse(respID, "server_error", "response_translation_error", err.Error()))
		return
	}
	if err := a.handleGeneration(r, chatReq, gen, ctx, sse, respID, 0); err != nil {
		a.logger.Error("failed to handle upstream chat response", zap.String("response_id", respID), zap.Error(err))
		_ = sse.Event("response.failed", failedResponse(respID, "server_error", "response_handling_error", err.Error()))
	}
}

func (a *Adapter) handleCompact(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := a.readRequestBody(r)
	if err != nil {
		a.logger.Warn("failed to read compact request", zap.String("path", r.URL.Path), zap.Error(err))
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	a.debug.SaveRawJSON("inbound compact request", body)

	var responsesReq map[string]any
	if err := json.Unmarshal(body, &responsesReq); err != nil {
		a.logger.Warn("invalid compact request json", zap.Error(err))
		writeJSONError(w, http.StatusBadRequest, "invalid_json", err.Error())
		return
	}
	chatReq, ctx, err := a.buildChatRequest(responsesReq, false)
	if err != nil {
		a.logger.Warn("failed to translate compact request", zap.Error(err))
		writeJSONError(w, http.StatusBadRequest, "translation_error", err.Error())
		return
	}
	a.debug.SaveJSON("upstream compact chat request", chatReq)

	upstream, err := a.postChat(r, chatReq)
	if err != nil {
		a.logger.Error("failed to send upstream compact chat request",
			zap.String("upstream_url", a.chatURL),
			zap.Error(err),
		)
		writeJSONError(w, http.StatusBadGateway, "upstream_request_error", err.Error())
		return
	}
	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {
			a.logger.Warn("failed to close response body", zap.Error(err))
		}
	}(upstream.Body)

	raw, err := io.ReadAll(upstream.Body)
	if err != nil {
		a.logger.Error("failed to read upstream compact chat response", zap.Error(err))
		writeJSONError(w, http.StatusBadGateway, "upstream_read_error", err.Error())
		return
	}
	if upstream.StatusCode < 200 || upstream.StatusCode >= 300 {
		a.debug.SaveRawJSON("upstream compact chat error", raw)
		errInfo := newUpstreamErrorInfo(upstream.StatusCode, upstream.Header, raw)
		a.logger.Warn("upstream compact chat request failed",
			errInfo.logFields("", a.chatURL)...,
		)
		writeJSONError(w, http.StatusBadGateway, "upstream_http_error", errInfo.message)
		return
	}
	a.debug.SaveRawJSON("upstream compact chat response", raw)

	gen, err := generationFromChatResponse(raw, ctx)
	if err != nil {
		a.logger.Error("failed to translate upstream compact chat response", zap.Error(err))
		writeJSONError(w, http.StatusBadGateway, "response_translation_error", err.Error())
		return
	}
	a.rememberToolExtraContent(gen)
	a.rememberMessageExtraContent(gen)
	out := map[string]any{"output": gen.outputItems()}
	a.debug.SaveJSON("outbound compact response", out)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func (a *Adapter) postChat(inbound *http.Request, chatReq map[string]any) (*http.Response, error) {
	data, err := json.Marshal(chatReq)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(inbound.Context(), http.MethodPost, a.chatURL, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	copyForwardHeaders(req.Header, inbound.Header)
	if a.apiKey != "" {
		req.Header.Set("Authorization", authorizationHeader(a.apiKey))
	}
	req.Header.Set("Content-Type", "application/json")
	if chatReq["stream"] == true {
		req.Header.Set("Accept", "text/event-stream")
	} else {
		req.Header.Set("Accept", "application/json")
	}
	return a.client.Do(req)
}

func authorizationHeader(apiKey string) string {
	apiKey = strings.TrimSpace(apiKey)
	if strings.Contains(apiKey, " ") {
		return apiKey
	}
	return "Bearer " + apiKey
}

func (a *Adapter) rememberToolExtraContent(gen *chatGeneration) {
	if gen == nil {
		return
	}
	indexes := make([]int, 0, len(gen.tools))
	for index := range gen.tools {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	for _, index := range indexes {
		call := gen.tools[index]
		if call == nil || call.ID == "" || call.ExtraContent == nil {
			continue
		}
		a.rememberExtraContent(call.ID, call.ExtraContent)
	}
}

func (a *Adapter) rememberMessageExtraContent(gen *chatGeneration) {
	if gen == nil || gen.message == nil || gen.message.ExtraContent == nil || gen.message.text.Len() == 0 {
		return
	}
	key := messageExtraContentKey("assistant", gen.message.text.String())
	if key == "" {
		return
	}
	a.extraMu.Lock()
	defer a.extraMu.Unlock()
	a.messageExtraByKey[key] = append(a.messageExtraByKey[key], cloneJSONValue(gen.message.ExtraContent))
	a.messageExtraOrder = append(a.messageExtraOrder, key)
	for len(a.messageExtraOrder) > maxMessageExtraContentEntries {
		oldest := a.messageExtraOrder[0]
		a.messageExtraOrder = a.messageExtraOrder[1:]
		items := a.messageExtraByKey[oldest]
		if len(items) <= 1 {
			delete(a.messageExtraByKey, oldest)
		} else {
			a.messageExtraByKey[oldest] = items[1:]
		}
	}
}

func (a *Adapter) rememberWebSearchHistory(call *chatToolCall, result string) {
	if call == nil {
		return
	}
	action := webSearchActionFromArguments(call.Arguments.String())
	key := webSearchHistoryKey(action)
	if key == "" {
		return
	}
	entry := webSearchHistoryEntry{
		CallID:       call.ID,
		Name:         call.Name,
		Arguments:    call.Arguments.String(),
		Result:       result,
		ExtraContent: cloneJSONValue(call.ExtraContent),
	}
	if entry.Name == "" {
		entry.Name = "web_search"
	}
	if strings.TrimSpace(entry.Arguments) == "" {
		entry.Arguments = webSearchArgumentsFromAction(action)
	}

	a.extraMu.Lock()
	defer a.extraMu.Unlock()
	a.webSearchByKey[key] = append(a.webSearchByKey[key], entry)
	a.webSearchOrder = append(a.webSearchOrder, key)
	for len(a.webSearchOrder) > maxWebSearchHistoryEntries {
		oldest := a.webSearchOrder[0]
		a.webSearchOrder = a.webSearchOrder[1:]
		items := a.webSearchByKey[oldest]
		if len(items) <= 1 {
			delete(a.webSearchByKey, oldest)
		} else {
			a.webSearchByKey[oldest] = items[1:]
		}
	}
}

func (a *Adapter) rememberExtraContent(callID string, extra any) {
	if callID == "" || extra == nil {
		return
	}
	a.extraMu.Lock()
	defer a.extraMu.Unlock()
	if _, exists := a.extraByCallID[callID]; !exists {
		a.extraOrder = append(a.extraOrder, callID)
	}
	a.extraByCallID[callID] = cloneJSONValue(extra)
	for len(a.extraOrder) > maxToolExtraContentEntries {
		oldest := a.extraOrder[0]
		a.extraOrder = a.extraOrder[1:]
		delete(a.extraByCallID, oldest)
	}
}

func (a *Adapter) extraContentForCallID(callID string) any {
	if callID == "" {
		return nil
	}
	a.extraMu.Lock()
	defer a.extraMu.Unlock()
	return cloneJSONValue(a.extraByCallID[callID])
}

func (a *Adapter) extraContentForMessage(key string, occurrence int) any {
	if key == "" || occurrence < 0 {
		return nil
	}
	a.extraMu.Lock()
	defer a.extraMu.Unlock()
	items := a.messageExtraByKey[key]
	if occurrence >= len(items) {
		return nil
	}
	return cloneJSONValue(items[occurrence])
}

func (a *Adapter) webSearchHistoryForAction(key string, occurrence int) *webSearchHistoryEntry {
	if key == "" || occurrence < 0 {
		return nil
	}
	a.extraMu.Lock()
	defer a.extraMu.Unlock()
	items := a.webSearchByKey[key]
	if occurrence >= len(items) {
		return nil
	}
	entry := items[occurrence]
	entry.ExtraContent = cloneJSONValue(entry.ExtraContent)
	return &entry
}

func (a *Adapter) buildChatRequest(req map[string]any, stream bool) (map[string]any, *translationContext, error) {
	builder := newRequestBuilder(a.reasoningHistory, a.extraContentForCallID, a.extraContentForMessage, a.webSearchHistoryForAction)
	tools := builder.translateTools(req["tools"])
	messages := builder.translateInput(req)
	if len(builder.extraTools) > 0 {
		tools = append(tools, builder.extraTools...)
	}
	if len(messages) == 0 {
		messages = append(messages, map[string]any{"role": "user", "content": ""})
	}

	chatReq := map[string]any{
		"model":            a.model,
		"reasoning_effort": a.reasoningEffort,
		"messages":         messages,
		"stream":           stream,
	}
	if a.reasoningHistory == reasoningHistoryReasoningContent && modelNeedsPreservedThinking(a.model) {
		chatReq["thinking"] = map[string]any{
			"type": "enabled",
			"keep": "all",
		}
	}
	if stream {
		chatReq["stream_options"] = map[string]any{"include_usage": true}
	}
	if len(tools) > 0 {
		chatReq["tools"] = tools
		if choice, ok := translateToolChoice(req["tool_choice"]); ok {
			chatReq["tool_choice"] = choice
		} else {
			chatReq["tool_choice"] = "auto"
		}
	}
	if v, ok := req["parallel_tool_calls"].(bool); ok {
		chatReq["parallel_tool_calls"] = v
	}
	if responseFormat, ok := translateResponseFormat(req["text"]); ok {
		chatReq["response_format"] = responseFormat
	}

	return chatReq, builder.context(), nil
}

func (a *Adapter) readRequestBody(r *http.Request) ([]byte, error) {
	if enc := r.Header.Get("Content-Encoding"); enc != "" && !strings.EqualFold(enc, "identity") {
		return nil, fmt.Errorf("unsupported content-encoding %q", enc)
	}
	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {
			a.logger.Warn("failed to close response body", zap.Error(err))
		}
	}(r.Body)
	return io.ReadAll(io.LimitReader(r.Body, 128<<20))
}

func writeJSONError(w http.ResponseWriter, status int, code string, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"type":    "server_error",
			"code":    code,
			"message": message,
		},
	})
}

func failedResponse(respID, typ, code, message string) map[string]any {
	return map[string]any{
		"response": map[string]any{
			"id":     respID,
			"object": "response",
			"status": "failed",
			"error": map[string]any{
				"type":    typ,
				"code":    code,
				"message": message,
			},
		},
	}
}

type upstreamErrorInfo struct {
	status         int
	message        string
	rawBody        string
	bodyTruncated  bool
	contentType    string
	requestID      string
	organizationID string
}

func newUpstreamErrorInfo(status int, header http.Header, raw []byte) upstreamErrorInfo {
	body, truncated := truncateString(strings.TrimSpace(string(raw)), maxUpstreamErrorBodyLogBytes)
	message := upstreamErrorMessage(status, raw)
	contentType := header.Get("Content-Type")
	return upstreamErrorInfo{
		status:         status,
		message:        message,
		rawBody:        body,
		bodyTruncated:  truncated,
		contentType:    contentType,
		requestID:      firstHeaderValue(header, "x-request-id", "x-goog-request-id", "request-id"),
		organizationID: firstHeaderValue(header, "openai-organization", "x-organization-id"),
	}
}

func (e upstreamErrorInfo) logFields(responseID, upstreamURL string) []zap.Field {
	fields := []zap.Field{
		zap.String("upstream_url", upstreamURL),
		zap.Int("status", e.status),
		zap.String("upstream_error", e.message),
	}
	if responseID != "" {
		fields = append(fields, zap.String("response_id", responseID))
	}
	if e.contentType != "" {
		fields = append(fields, zap.String("content_type", e.contentType))
	}
	if e.requestID != "" {
		fields = append(fields, zap.String("upstream_request_id", e.requestID))
	}
	if e.organizationID != "" {
		fields = append(fields, zap.String("upstream_organization_id", e.organizationID))
	}
	if e.rawBody != "" {
		fields = append(fields,
			zap.String("upstream_response_body", e.rawBody),
			zap.Bool("upstream_response_body_truncated", e.bodyTruncated),
		)
	}
	return fields
}

func upstreamErrorMessage(status int, raw []byte) string {
	msg := extractUpstreamErrorMessage(raw)
	if msg == "" {
		return fmt.Sprintf("upstream returned HTTP %d", status)
	}
	msg, truncated := truncateString(msg, maxUpstreamErrorMessageBytes)
	if truncated {
		msg += "..."
	}
	return fmt.Sprintf("upstream returned HTTP %d: %s", status, msg)
}

func extractUpstreamErrorMessage(raw []byte) string {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return ""
	}

	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return string(raw)
	}
	msg := messageFromJSON(value)
	if msg != "" {
		return msg
	}
	return string(raw)
}

func messageFromJSON(value any) string {
	switch v := value.(type) {
	case string:
		return strings.TrimSpace(v)
	case []any:
		var parts []string
		for _, item := range v {
			if part := messageFromJSON(item); part != "" {
				parts = append(parts, part)
			}
		}
		return strings.Join(parts, "; ")
	case map[string]any:
		for _, key := range []string{"message", "detail", "error_description"} {
			if msg := messageFromJSON(v[key]); msg != "" {
				return msg
			}
		}
		if errValue, ok := v["error"]; ok {
			if msg := messageFromJSON(errValue); msg != "" {
				return msg
			}
		}
		var parts []string
		for _, key := range []string{"type", "code", "status"} {
			if part := scalarString(v[key]); part != "" {
				parts = append(parts, key+"="+part)
			}
		}
		return strings.Join(parts, " ")
	default:
		return scalarString(v)
	}
}

func scalarString(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(v)
	case float64, bool:
		return fmt.Sprint(v)
	default:
		return ""
	}
}

func firstHeaderValue(header http.Header, names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(header.Get(name)); value != "" {
			return value
		}
	}
	return ""
}

func truncateString(value string, limit int) (string, bool) {
	if limit <= 0 || len(value) <= limit {
		return value, false
	}
	return value[:limit], true
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

func copyForwardHeaders(dst, src http.Header) {
	hopByHop := map[string]bool{
		"connection":          true,
		"keep-alive":          true,
		"proxy-authenticate":  true,
		"proxy-authorization": true,
		"te":                  true,
		"trailer":             true,
		"transfer-encoding":   true,
		"upgrade":             true,
		"host":                true,
		"content-length":      true,
		"content-type":        true,
		"accept":              true,
		"accept-encoding":     true,
		"content-encoding":    true,
	}
	for k, values := range src {
		if hopByHop[strings.ToLower(k)] {
			continue
		}
		for _, v := range values {
			dst.Add(k, v)
		}
	}
}

func isEventStream(contentType string) bool {
	if contentType == "" {
		return false
	}
	mt, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return strings.Contains(strings.ToLower(contentType), "text/event-stream")
	}
	return strings.EqualFold(mt, "text/event-stream")
}

type requestBuilder struct {
	usedNames            map[string]int
	byChat               map[string]toolMapping
	byKey                map[string]toolMapping
	extraTools           []any
	pendingImageMsgs     []map[string]any
	renderedCallIDs      map[string]bool
	renderedCallNames    map[string]string
	reasoningHistory     string
	messageOccurrences   map[string]int
	webSearchOccurrences map[string]int
	lookupExtraContent   func(callID string) any
	lookupMessageExtra   func(key string, occurrence int) any
	lookupWebSearch      func(key string, occurrence int) *webSearchHistoryEntry
}

type translationContext struct {
	byChat map[string]toolMapping
}

type toolMapping struct {
	Kind      string
	ChatName  string
	Name      string
	Namespace string
}

type assistantHistoryMessage struct {
	reasoning    strings.Builder
	content      any
	extraContent any
	hasText      bool
	toolCalls    []any
}

func (m *assistantHistoryMessage) empty() bool {
	return m.reasoning.Len() == 0 && !m.hasText && len(m.toolCalls) == 0
}

func (m *assistantHistoryMessage) flush() []map[string]any {
	if m.empty() {
		return nil
	}
	message := map[string]any{"role": "assistant"}
	if m.hasText {
		message["content"] = m.content
	} else if len(m.toolCalls) > 0 {
		message["content"] = nil
	} else {
		message["content"] = ""
	}
	if m.reasoning.Len() > 0 {
		message["reasoning_content"] = m.reasoning.String()
	}
	if m.extraContent != nil {
		message["extra_content"] = cloneJSONValue(m.extraContent)
	}
	if len(m.toolCalls) > 0 {
		message["tool_calls"] = m.toolCalls
	}
	m.reasoning.Reset()
	m.content = nil
	m.extraContent = nil
	m.hasText = false
	m.toolCalls = nil
	return []map[string]any{message}
}

func newRequestBuilder(reasoningHistory string, lookupExtraContent func(callID string) any, lookupMessageExtra func(key string, occurrence int) any, lookupWebSearch func(key string, occurrence int) *webSearchHistoryEntry) *requestBuilder {
	return &requestBuilder{
		usedNames:            map[string]int{},
		byChat:               map[string]toolMapping{},
		byKey:                map[string]toolMapping{},
		renderedCallIDs:      map[string]bool{},
		renderedCallNames:    map[string]string{},
		reasoningHistory:     reasoningHistory,
		messageOccurrences:   map[string]int{},
		webSearchOccurrences: map[string]int{},
		lookupExtraContent:   lookupExtraContent,
		lookupMessageExtra:   lookupMessageExtra,
		lookupWebSearch:      lookupWebSearch,
	}
}

func (b *requestBuilder) context() *translationContext {
	byChat := make(map[string]toolMapping, len(b.byChat))
	for k, v := range b.byChat {
		byChat[k] = v
	}
	return &translationContext{byChat: byChat}
}

func (b *requestBuilder) translateTools(value any) []any {
	items, ok := value.([]any)
	if !ok {
		return nil
	}
	var out []any
	for _, item := range items {
		tool, ok := item.(map[string]any)
		if !ok {
			continue
		}
		switch stringField(tool, "type") {
		case "function":
			out = append(out, b.functionTool("", tool, "function"))
		case "namespace":
			namespace := stringField(tool, "name")
			children, _ := tool["tools"].([]any)
			for _, child := range children {
				childTool, ok := child.(map[string]any)
				if !ok || stringField(childTool, "type") != "function" {
					continue
				}
				out = append(out, b.functionTool(namespace, childTool, "function"))
			}
		case "custom":
			out = append(out, b.customTool(tool))
		case "tool_search":
			out = append(out, b.hostedFunctionTool("tool_search", "tool_search", stringField(tool, "description"), schemaOrDefault(tool["parameters"])))
		case "web_search":
			out = append(out, b.hostedFunctionTool("web_search", "web_search", webSearchToolDescription, webSearchSchema()))
		case "image_generation":
			out = append(out, b.hostedFunctionTool("image_generation", "image_generation", "Request image generation. If the provider cannot return base64 image data in result, Codex will receive a failed image_generation_call item.", imageGenerationSchema()))
		default:
			if stringField(tool, "name") != "" {
				out = append(out, b.functionTool("", tool, "function"))
			}
		}
	}
	return out
}

func (b *requestBuilder) functionTool(namespace string, tool map[string]any, kind string) map[string]any {
	name := stringField(tool, "name")
	chatName := b.register(kind, namespace, name)
	fn := map[string]any{
		"name":        chatName,
		"description": stringField(tool, "description"),
		"parameters":  schemaOrDefault(tool["parameters"]),
	}
	if strict, ok := tool["strict"].(bool); ok {
		fn["strict"] = strict
	}
	return map[string]any{"type": "function", "function": fn}
}

func (b *requestBuilder) customTool(tool map[string]any) map[string]any {
	name := stringField(tool, "name")
	chatName := b.register("custom", "", name)
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        chatName,
			"description": customToolDescription(tool),
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"input": map[string]any{
						"type":        "string",
						"description": customToolInputDescription(),
					},
				},
				"required":             []any{"input"},
				"additionalProperties": false,
			},
			"strict": true,
		},
	}
}

func customToolDescription(tool map[string]any) string {
	sections := make([]string, 0, 3)
	if description := stringField(tool, "description"); strings.TrimSpace(description) != "" {
		sections = append(sections, description)
	}
	if formatDescription := customToolFormatDescription(tool["format"]); formatDescription != "" {
		sections = append(sections, formatDescription)
	}
	sections = append(sections, "Call it with a JSON object containing exactly one string field named input. The input value must be the complete freeform tool payload.")
	return strings.Join(sections, "\n\n")
}

func customToolFormatDescription(format any) string {
	formatMap, ok := format.(map[string]any)
	if !ok || len(formatMap) == 0 {
		if format == nil {
			return ""
		}
		return "Responses custom tool format:\n" + compactJSONString(format)
	}

	sections := make([]string, 0, 4)
	sections = append(sections, "Responses custom tool format:")
	if formatType := stringField(formatMap, "type"); strings.TrimSpace(formatType) != "" {
		sections = append(sections, "type: "+formatType)
	}
	if syntax := stringField(formatMap, "syntax"); strings.TrimSpace(syntax) != "" {
		sections = append(sections, "syntax: "+syntax)
	}
	if definition := stringField(formatMap, "definition"); strings.TrimSpace(definition) != "" {
		sections = append(sections, "definition:\n"+indentText(definition, "  "))
	}
	return strings.Join(sections, "\n\n")
}

func customToolInputDescription() string {
	return "Complete freeform input for the custom tool. The upstream description preserves the original Responses instructions and format grammar."
}

func indentText(text, prefix string) string {
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = prefix + line
	}
	return strings.Join(lines, "\n")
}

func (b *requestBuilder) hostedFunctionTool(name, kind, description string, parameters any) map[string]any {
	chatName := b.register(kind, "", name)
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        chatName,
			"description": description,
			"parameters":  parameters,
		},
	}
}

func (b *requestBuilder) register(kind, namespace, name string) string {
	if name == "" {
		name = "unnamed_tool"
	}
	flat := flatToolName(namespace, name)
	chatName := b.uniqueChatName(flat)
	mapping := toolMapping{Kind: kind, ChatName: chatName, Name: name, Namespace: namespace}
	b.byChat[chatName] = mapping
	b.byKey[toolKey(kind, namespace, name)] = mapping
	return chatName
}

func (b *requestBuilder) knownChatNameFor(kind, namespace, name string) (string, bool) {
	if mapping, ok := b.byKey[toolKey(kind, namespace, name)]; ok {
		return mapping.ChatName, true
	}
	if kind != "function" {
		if mapping, ok := b.byKey[toolKey("function", namespace, name)]; ok {
			return mapping.ChatName, true
		}
	}
	return "", false
}

func (b *requestBuilder) hasRegisteredTool(kind, namespace, name string) bool {
	if _, ok := b.byKey[toolKey(kind, namespace, name)]; ok {
		return true
	}
	if kind != "function" {
		_, ok := b.byKey[toolKey("function", namespace, name)]
		return ok
	}
	return false
}

func (b *requestBuilder) uniqueChatName(original string) string {
	base := safeToolName(original)
	count := b.usedNames[base]
	b.usedNames[base] = count + 1
	if count == 0 {
		return base
	}
	suffix := "_" + strconv.Itoa(count+1)
	maxBaseLen := 64 - len(suffix)
	if maxBaseLen < 1 {
		maxBaseLen = 1
	}
	if len(base) > maxBaseLen {
		base = strings.TrimRight(base[:maxBaseLen], "_-")
		if base == "" {
			base = "tool"
		}
	}
	return base + suffix
}

func toolKey(kind, namespace, name string) string {
	return kind + "\x00" + namespace + "\x00" + name
}

func flatToolName(namespace, name string) string {
	if namespace == "" {
		return name
	}
	if strings.HasSuffix(namespace, "_") || strings.HasPrefix(name, "_") {
		return namespace + name
	}
	return namespace + "_" + name
}

func safeToolName(name string) string {
	if name == "" {
		return "tool"
	}
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	out := strings.Trim(b.String(), "_-")
	if out == "" {
		out = "tool"
	}
	if reservedMCPToolName(out) {
		out = "tool_" + out
	}
	if len(out) > 64 {
		out = strings.TrimRight(out[:64], "_-")
		if out == "" {
			out = "tool"
		}
	}
	return out
}

func reservedMCPToolName(name string) bool {
	return strings.HasPrefix(strings.ToLower(name), "mcp__")
}

func (b *requestBuilder) translateInput(req map[string]any) []map[string]any {
	var messages []map[string]any
	var pendingAssistant assistantHistoryMessage
	if instructions, ok := req["instructions"].(string); ok && strings.TrimSpace(instructions) != "" {
		messages = append(messages, map[string]any{"role": "system", "content": instructions})
	}

	switch input := req["input"].(type) {
	case string:
		messages = append(messages, map[string]any{"role": "user", "content": input})
	case []any:
		for _, value := range input {
			item, ok := value.(map[string]any)
			if !ok {
				messages = append(messages, b.flushPendingImageMessages()...)
				messages = append(messages, pendingAssistant.flush()...)
				messages = append(messages, markerMessage(value))
				continue
			}
			if !isToolOutputItem(item) {
				messages = append(messages, b.flushPendingImageMessages()...)
			}
			itemMessages := b.itemToMessages(item)
			if b.reasoningHistory == reasoningHistoryReasoningContent {
				if b.mergeReasoningHistoryItem(item, itemMessages, &pendingAssistant, &messages) {
					continue
				}
			} else if b.mergeToolCallHistoryItem(item, itemMessages, &pendingAssistant, &messages) {
				continue
			}
			messages = append(messages, pendingAssistant.flush()...)
			messages = append(messages, itemMessages...)
		}
		messages = append(messages, pendingAssistant.flush()...)
		messages = append(messages, b.flushPendingImageMessages()...)
	case nil:
	default:
		messages = append(messages, b.flushPendingImageMessages()...)
		messages = append(messages, pendingAssistant.flush()...)
		messages = append(messages, markerMessage(input))
	}
	return messages
}

func (b *requestBuilder) mergeReasoningHistoryItem(item map[string]any, itemMessages []map[string]any, pending *assistantHistoryMessage, messages *[]map[string]any) bool {
	switch stringField(item, "type") {
	case "reasoning":
		if content := reasoningItemContent(item); content != "" {
			pending.reasoning.WriteString(content)
		}
		return true
	case "message":
		if normalizeRole(stringField(item, "role")) != "assistant" || len(itemMessages) != 1 {
			return false
		}
		if pending.hasText {
			*messages = append(*messages, pending.flush()...)
		}
		pending.content = itemMessages[0]["content"]
		pending.extraContent = itemMessages[0]["extra_content"]
		pending.hasText = true
		return true
	case "function_call", "custom_tool_call", "tool_search_call":
		return b.mergeToolCallHistoryItem(item, itemMessages, pending, messages)
	case "web_search_call":
		merged := b.mergeToolCallHistoryItem(item, itemMessages, pending, messages)
		if merged && len(itemMessages) == 0 {
			pending.reasoning.Reset()
		}
		return merged
	default:
		return false
	}
}

func (b *requestBuilder) mergeToolCallHistoryItem(item map[string]any, itemMessages []map[string]any, pending *assistantHistoryMessage, messages *[]map[string]any) bool {
	switch stringField(item, "type") {
	case "function_call", "custom_tool_call", "tool_search_call":
		if len(itemMessages) != 1 {
			return false
		}
		calls, ok := itemMessages[0]["tool_calls"].([]any)
		if !ok || len(calls) == 0 {
			return false
		}
		pending.toolCalls = append(pending.toolCalls, calls...)
		return true
	case "web_search_call":
		if len(itemMessages) == 0 {
			return true
		}
		calls, ok := itemMessages[0]["tool_calls"].([]any)
		if !ok || len(calls) == 0 {
			return false
		}
		pending.toolCalls = append(pending.toolCalls, calls...)
		*messages = append(*messages, pending.flush()...)
		*messages = append(*messages, itemMessages[1:]...)
		return true
	default:
		return false
	}
}

func (b *requestBuilder) messageExtraContent(item map[string]any, role string, content any) any {
	key := messageExtraContentKey(role, content)
	occurrence := 0
	if key != "" {
		occurrence = b.messageOccurrences[key]
		b.messageOccurrences[key] = occurrence + 1
	}
	if extra := item["extra_content"]; extra != nil {
		return cloneJSONValue(extra)
	}
	if key == "" || b.lookupMessageExtra == nil {
		return nil
	}
	return b.lookupMessageExtra(key, occurrence)
}

func messageExtraContentKey(role string, content any) string {
	if role != "assistant" || content == nil {
		return ""
	}
	data, err := json.Marshal(content)
	if err != nil {
		data = []byte(fmt.Sprint(content))
	}
	hashInput := make([]byte, 0, len(role)+1+len(data))
	hashInput = append(hashInput, role...)
	hashInput = append(hashInput, 0)
	hashInput = append(hashInput, data...)
	sum := sha256.Sum256(hashInput)
	return hex.EncodeToString(sum[:])
}

func (b *requestBuilder) itemToMessages(item map[string]any) []map[string]any {
	switch stringField(item, "type") {
	case "message":
		role := normalizeRole(stringField(item, "role"))
		message := map[string]any{
			"role":    role,
			"content": chatContentFromResponsesContent(item["content"], role),
		}
		if extra := b.messageExtraContent(item, role, message["content"]); extra != nil {
			message["extra_content"] = extra
		}
		return []map[string]any{message}
	case "reasoning":
		if b.reasoningHistory == reasoningHistoryAssistantContent {
			if content := reasoningItemContent(item); content != "" {
				return []map[string]any{{
					"role":    "assistant",
					"content": content,
				}}
			}
		}
		return nil
	case "function_call":
		name := stringField(item, "name")
		namespace := stringField(item, "namespace")
		callID := stringField(item, "call_id")
		args := stringField(item, "arguments")
		chatName, ok := b.knownChatNameFor("function", namespace, name)
		if !ok {
			return []map[string]any{markerMessage(item)}
		}
		if !validFunctionCallArguments(args) {
			return []map[string]any{markerMessage(item)}
		}
		b.renderedCallIDs[callID] = true
		b.renderedCallNames[callID] = chatName
		return []map[string]any{assistantToolCallMessage(callID, chatName, args, b.extraContentForItem(item, callID))}
	case "custom_tool_call":
		name := stringField(item, "name")
		args, _ := json.Marshal(map[string]string{"input": stringField(item, "input")})
		callID := stringField(item, "call_id")
		chatName, ok := b.knownChatNameFor("custom", "", name)
		if !ok {
			return []map[string]any{markerMessage(item)}
		}
		b.renderedCallIDs[callID] = true
		b.renderedCallNames[callID] = chatName
		return []map[string]any{assistantToolCallMessage(callID, chatName, string(args), b.extraContentForItem(item, callID))}
	case "tool_search_call":
		callID := optionalCallID(item)
		args := compactJSONString(item["arguments"])
		chatName, ok := b.knownChatNameFor("tool_search", "", "tool_search")
		if !ok {
			chatName = "tool_search"
		}
		b.renderedCallIDs[callID] = true
		b.renderedCallNames[callID] = chatName
		return []map[string]any{assistantToolCallMessage(callID, chatName, args, b.extraContentForItem(item, callID))}
	case "function_call_output", "custom_tool_call_output":
		callID := stringField(item, "call_id")
		if callID == "" {
			return []map[string]any{markerMessage(item)}
		}
		if !b.renderedCallIDs[callID] {
			return []map[string]any{markerMessage(item)}
		}
		messages, imageMessages := toolOutputMessages(callID, b.renderedCallNames[callID], item["output"])
		b.pendingImageMsgs = append(b.pendingImageMsgs, imageMessages...)
		return messages
	case "tool_search_output":
		callID := optionalCallID(item)
		b.registerToolSearchOutputTools(item["tools"])
		if callID == "" {
			return []map[string]any{markerMessage(item)}
		}
		if !b.renderedCallIDs[callID] {
			return []map[string]any{markerMessage(item)}
		}
		return []map[string]any{{
			"role":         "tool",
			"tool_call_id": callID,
			"name":         b.renderedCallNames[callID],
			"content":      compactJSONString(item),
		}}
	case "web_search_call":
		return b.webSearchHistoryMessages(item)
	default:
		return []map[string]any{markerMessage(item)}
	}
}

func (b *requestBuilder) webSearchHistoryMessages(item map[string]any) []map[string]any {
	action, ok := item["action"].(map[string]any)
	if !ok {
		return nil
	}
	key := webSearchHistoryKey(action)
	occurrence := 0
	if key != "" {
		occurrence = b.webSearchOccurrences[key]
		b.webSearchOccurrences[key] = occurrence + 1
	}
	if key == "" || b.lookupWebSearch == nil {
		return nil
	}
	entry := b.lookupWebSearch(key, occurrence)
	if entry == nil {
		return nil
	}
	callID := entry.CallID
	if callID == "" {
		callID = webSearchHistoryCallID(key, occurrence)
	}
	name := entry.Name
	if name == "" {
		name = "web_search"
	}
	args := strings.TrimSpace(entry.Arguments)
	if args == "" {
		args = webSearchArgumentsFromAction(action)
	}
	return []map[string]any{
		assistantToolCallMessage(callID, name, args, entry.ExtraContent),
		{
			"role":         "tool",
			"tool_call_id": callID,
			"name":         name,
			"content":      entry.Result,
		},
	}
}

func (b *requestBuilder) flushPendingImageMessages() []map[string]any {
	if len(b.pendingImageMsgs) == 0 {
		return nil
	}
	messages := b.pendingImageMsgs
	b.pendingImageMsgs = nil
	return messages
}

func isToolOutputItem(item map[string]any) bool {
	switch stringField(item, "type") {
	case "function_call_output", "custom_tool_call_output":
		return true
	default:
		return false
	}
}

func reasoningItemContent(item map[string]any) string {
	if content := chatContentFromResponsesContent(item["content"], "assistant"); content != nil {
		if text, ok := content.(string); ok && strings.TrimSpace(text) != "" {
			return text
		}
	}
	if content := chatContentFromResponsesContent(item["summary"], "assistant"); content != nil {
		if text, ok := content.(string); ok && strings.TrimSpace(text) != "" {
			return text
		}
	}
	return ""
}

func (b *requestBuilder) registerToolSearchOutputTools(value any) {
	items, ok := value.([]any)
	if !ok {
		return
	}
	for _, raw := range items {
		tool, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		b.registerDiscoveredTool("", tool)
	}
}

func (b *requestBuilder) registerDiscoveredTool(namespace string, tool map[string]any) {
	toolType := stringField(tool, "type")
	if toolType == "" && tool["tools"] != nil {
		toolType = "namespace"
	}
	if toolType == "" && stringField(tool, "name") != "" {
		toolType = "function"
	}

	switch toolType {
	case "function":
		name := stringField(tool, "name")
		if name == "" || b.hasRegisteredTool("function", namespace, name) {
			return
		}
		b.extraTools = append(b.extraTools, b.functionTool(namespace, tool, "function"))
	case "custom":
		name := stringField(tool, "name")
		if namespace != "" || name == "" || b.hasRegisteredTool("custom", "", name) {
			return
		}
		b.extraTools = append(b.extraTools, b.customTool(tool))
	case "namespace":
		childNamespace := stringField(tool, "name")
		if childNamespace == "" {
			childNamespace = namespace
		}
		children, _ := tool["tools"].([]any)
		for _, rawChild := range children {
			child, ok := rawChild.(map[string]any)
			if !ok {
				continue
			}
			b.registerDiscoveredTool(childNamespace, child)
		}
	}
}

func (b *requestBuilder) extraContentForItem(item map[string]any, callID string) any {
	if extra := item["extra_content"]; extra != nil {
		return cloneJSONValue(extra)
	}
	if b.lookupExtraContent == nil {
		return nil
	}
	return b.lookupExtraContent(callID)
}

func assistantToolCallMessage(callID, name, arguments string, extraContent any) map[string]any {
	if callID == "" {
		callID = newID("call")
	}
	if arguments == "" {
		arguments = "{}"
	}
	toolCall := map[string]any{
		"id":   callID,
		"type": "function",
		"function": map[string]any{
			"name":      name,
			"arguments": arguments,
		},
	}
	if extraContent != nil {
		toolCall["extra_content"] = cloneJSONValue(extraContent)
	}
	return map[string]any{
		"role":       "assistant",
		"content":    nil,
		"tool_calls": []any{toolCall},
	}
}

func assistantToolCallsMessage(calls []*chatToolCall) map[string]any {
	toolCalls := make([]any, 0, len(calls))
	for _, call := range calls {
		callID := call.ID
		if callID == "" {
			callID = newID("call")
			call.ID = callID
		}
		args := call.Arguments.String()
		if args == "" {
			args = "{}"
		}
		toolCall := map[string]any{
			"id":   callID,
			"type": "function",
			"function": map[string]any{
				"name":      call.Name,
				"arguments": args,
			},
		}
		if call.ExtraContent != nil {
			toolCall["extra_content"] = cloneJSONValue(call.ExtraContent)
		}
		toolCalls = append(toolCalls, toolCall)
	}
	return map[string]any{
		"role":       "assistant",
		"content":    nil,
		"tool_calls": toolCalls,
	}
}

func markerMessage(value any) map[string]any {
	return map[string]any{
		"role":    "assistant",
		"content": "[Responses API item]\n" + compactJSONString(value),
	}
}

func normalizeRole(role string) string {
	switch role {
	case "system", "user", "assistant", "tool":
		return role
	case "developer":
		return "system"
	case "":
		return "user"
	default:
		return "user"
	}
}

func chatContentFromResponsesContent(value any, role string) any {
	items, ok := value.([]any)
	if !ok {
		if s, ok := value.(string); ok {
			return s
		}
		return ""
	}
	var text strings.Builder
	var parts []any
	hasImage := false
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		switch stringField(item, "type") {
		case "input_text", "output_text", "text", "reasoning_text", "summary_text":
			s := stringField(item, "text")
			text.WriteString(s)
			parts = append(parts, map[string]any{"type": "text", "text": s})
		case "input_image":
			imageURL := stringField(item, "image_url")
			if imageURL == "" {
				continue
			}
			hasImage = true
			image := map[string]any{"url": imageURL}
			if detail := stringField(item, "detail"); detail != "" {
				image["detail"] = strings.ToLower(detail)
			}
			parts = append(parts, map[string]any{"type": "image_url", "image_url": image})
		default:
			s := compactJSONString(item)
			text.WriteString(s)
			parts = append(parts, map[string]any{"type": "text", "text": s})
		}
	}
	if hasImage && role == "user" {
		return parts
	}
	return text.String()
}

func toolOutputToText(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case []any:
		var parts []string
		for _, raw := range v {
			item, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			switch stringField(item, "type") {
			case "input_text":
				if s := stringField(item, "text"); strings.TrimSpace(s) != "" {
					parts = append(parts, s)
				}
			case "input_image":
				if s := stringField(item, "image_url"); s != "" {
					parts = append(parts, "[image attached]")
				}
			case "encrypted_content":
				parts = append(parts, "[encrypted_content omitted]")
			default:
				parts = append(parts, compactJSONString(item))
			}
		}
		return strings.Join(parts, "\n")
	case nil:
		return ""
	default:
		return compactJSONString(v)
	}
}

func toolOutputMessages(callID, name string, output any) ([]map[string]any, []map[string]any) {
	text := toolOutputToText(output)
	imageContent, hasImage := toolOutputImageContent(output, callID)
	if hasImage && strings.TrimSpace(text) == "" {
		text = "[image attached]"
	}
	messages := []map[string]any{{
		"role":         "tool",
		"tool_call_id": callID,
		"name":         name,
		"content":      text,
	}}
	if hasImage {
		return messages, []map[string]any{{
			"role":    "user",
			"content": imageContent,
		}}
	}
	return messages, nil
}

func toolOutputImageContent(value any, callID string) ([]any, bool) {
	items, ok := value.([]any)
	if !ok {
		return nil, false
	}
	parts := []any{
		map[string]any{
			"type": "text",
			"text": "Tool output image for call " + callID + ".",
		},
	}
	hasImage := false
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		switch stringField(item, "type") {
		case "input_text":
			if text := stringField(item, "text"); strings.TrimSpace(text) != "" {
				parts = append(parts, map[string]any{"type": "text", "text": text})
			}
		case "input_image":
			imageURL := stringField(item, "image_url")
			if imageURL == "" {
				continue
			}
			image := map[string]any{"url": imageURL}
			if detail := stringField(item, "detail"); detail != "" {
				image["detail"] = strings.ToLower(detail)
			}
			parts = append(parts, map[string]any{"type": "image_url", "image_url": image})
			hasImage = true
		}
	}
	if !hasImage {
		return nil, false
	}
	return parts, true
}

func optionalCallID(item map[string]any) string {
	if s := stringField(item, "call_id"); s != "" {
		return s
	}
	return ""
}

func schemaOrDefault(value any) any {
	if value != nil {
		return value
	}
	return map[string]any{
		"type":                 "object",
		"properties":           map[string]any{},
		"additionalProperties": true,
	}
}

func webSearchSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"action":              map[string]any{"type": "string", "enum": []any{"search", "open_page", "find_in_page"}},
			"query":               map[string]any{"type": "string"},
			"queries":             map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			"domains":             map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			"external_web_access": map[string]any{"type": "boolean"},
			"filters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"allowed_domains": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				},
				"additionalProperties": true,
			},
			"search_context_size":  map[string]any{"type": "string", "enum": []any{"low", "medium", "high"}},
			"search_content_types": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			"user_location": map[string]any{
				"type":                 "object",
				"additionalProperties": true,
			},
			"url":     map[string]any{"type": "string"},
			"pattern": map[string]any{"type": "string"},
		},
		"additionalProperties": true,
	}
}

const webSearchToolDescription = "Request a web search action. Use query for one search, or queries for a single hosted web_search action with multiple searches. When explicitly asked for parallel tool calls, issue multiple web_search tool calls in the same assistant turn."

func imageGenerationSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"prompt":         map[string]any{"type": "string"},
			"revised_prompt": map[string]any{"type": "string"},
			"result":         map[string]any{"type": "string", "description": "Base64 encoded PNG image data, if available."},
			"b64_json":       map[string]any{"type": "string", "description": "Base64 encoded image data, if available."},
		},
		"required":             []any{"prompt"},
		"additionalProperties": true,
	}
}

func translateToolChoice(value any) (any, bool) {
	switch v := value.(type) {
	case string:
		switch v {
		case "auto", "none", "required":
			return v, true
		case "":
			return nil, false
		default:
			return v, true
		}
	case map[string]any:
		return v, true
	default:
		return nil, false
	}
}

func translateResponseFormat(value any) (any, bool) {
	text, ok := value.(map[string]any)
	if !ok {
		return nil, false
	}
	format, ok := text["format"].(map[string]any)
	if !ok {
		return nil, false
	}
	if stringField(format, "type") != "json_schema" {
		return nil, false
	}
	name := stringField(format, "name")
	if name == "" {
		name = "response_format"
	}
	jsonSchema := map[string]any{
		"name":   name,
		"schema": format["schema"],
	}
	if strict, ok := format["strict"].(bool); ok {
		jsonSchema["strict"] = strict
	}
	return map[string]any{
		"type":        "json_schema",
		"json_schema": jsonSchema,
	}, true
}

func (a *Adapter) translateChatStream(
	inbound *http.Request,
	body io.Reader,
	upstreamReq map[string]any,
	ctx *translationContext,
	sse *responseSSEWriter,
	respID string,
	webSearchDepth int,
) error {
	gen := newChatGeneration(ctx)
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	var dataLines []string

	process := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		dataLines = nil
		if strings.TrimSpace(data) == "[DONE]" {
			return nil
		}
		a.debug.SaveRawJSON("upstream chat stream chunk", []byte(data))
		var chunk map[string]any
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return err
		}
		if errValue, ok := chunk["error"]; ok && errValue != nil {
			return fmt.Errorf("upstream stream error: %s", compactJSONString(errValue))
		}
		return gen.applyStreamChunk(chunk, sse, respID)
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := process(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	if err := process(); err != nil {
		return err
	}
	return a.handleGeneration(inbound, upstreamReq, gen, ctx, sse, respID, webSearchDepth)
}

type responseItemState struct {
	item any
	done bool
}

type streamedAssistantItem struct {
	id           string
	outputIndex  int
	announced    bool
	done         bool
	ExtraContent any
	text         strings.Builder
}

func (i *streamedAssistantItem) responseItem() map[string]any {
	id := i.id
	if id == "" {
		id = newID("msg")
	}
	item := map[string]any{
		"id":   id,
		"type": "message",
		"role": "assistant",
		"content": []any{
			map[string]any{"type": "output_text", "text": i.text.String()},
		},
	}
	if i.ExtraContent != nil {
		item["extra_content"] = cloneJSONValue(i.ExtraContent)
	}
	return item
}

type streamedReasoningItem struct {
	id          string
	outputIndex int
	announced   bool
	done        bool
	text        strings.Builder
}

func (i *streamedReasoningItem) responseItem() map[string]any {
	id := i.id
	if id == "" {
		id = newID("rs")
	}
	return map[string]any{
		"id":                id,
		"type":              "reasoning",
		"summary":           []any{},
		"encrypted_content": nil,
		"content": []any{
			map[string]any{"type": "reasoning_text", "text": i.text.String()},
		},
	}
}

type chatGeneration struct {
	ctx             *translationContext
	message         *streamedAssistantItem
	reasoning       *streamedReasoningItem
	tools           map[int]*chatToolCall
	usage           map[string]any
	finishReason    string
	activeKind      string
	activeToolIndex int
	nextOutputIndex int
}

type chatToolCall struct {
	Index              int
	ItemID             string
	ID                 string
	Type               string
	Name               string
	ExtraContent       any
	Arguments          strings.Builder
	Mapping            toolMapping
	announced          bool
	done               bool
	outputIndex        int
	customInputEmitted string
	rawCustomInput     bool
}

func newChatGeneration(ctx *translationContext) *chatGeneration {
	return &chatGeneration{ctx: ctx, tools: map[int]*chatToolCall{}, activeToolIndex: -1}
}

func (g *chatGeneration) messageState() *streamedAssistantItem {
	if g.message == nil {
		g.message = &streamedAssistantItem{id: newID("msg"), outputIndex: -1}
	}
	return g.message
}

func (g *chatGeneration) reasoningState() *streamedReasoningItem {
	if g.reasoning == nil {
		g.reasoning = &streamedReasoningItem{id: newID("rs"), outputIndex: -1}
	}
	return g.reasoning
}

func (g *chatGeneration) reasoningContent() string {
	if g.reasoning == nil {
		return ""
	}
	return g.reasoning.text.String()
}

func (g *chatGeneration) applyStreamChunk(chunk map[string]any, sse *responseSSEWriter, respID string) error {
	if usage, ok := chunk["usage"].(map[string]any); ok {
		g.usage = usage
	}
	choices, _ := chunk["choices"].([]any)
	for _, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]any)
		if !ok {
			continue
		}
		if finishReason := stringField(choice, "finish_reason"); finishReason != "" {
			g.finishReason = finishReason
		}
		delta, _ := choice["delta"].(map[string]any)
		if s := textFromAny(delta["content"]); s != "" {
			g.ensureMessageActive(sse, respID)
			g.message.text.WriteString(s)
			_ = sse.Event("response.output_text.delta", map[string]any{
				"response_id":  respID,
				"item_id":      g.message.id,
				"output_index": g.message.outputIndex,
				"delta":        s,
			})
		}
		if extra := delta["extra_content"]; extra != nil {
			g.messageState().ExtraContent = cloneJSONValue(extra)
		}
		if s := firstTextField(delta, "reasoning_content", "reasoning", "reasoning_delta"); s != "" {
			g.ensureReasoningActive(sse, respID)
			g.reasoning.text.WriteString(s)
			_ = sse.Event("response.reasoning_text.delta", map[string]any{
				"response_id":   respID,
				"item_id":       g.reasoning.id,
				"output_index":  g.reasoning.outputIndex,
				"delta":         s,
				"content_index": 0,
			})
		}
		g.applyToolCallDeltas(delta, sse, respID)
	}
	return nil
}

func (g *chatGeneration) applyToolCallDeltas(delta map[string]any, sse *responseSSEWriter, respID string) {
	if calls, ok := delta["tool_calls"].([]any); ok {
		for position, raw := range calls {
			call, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			acc := g.toolForDelta(call, position)
			if s := stringField(call, "id"); s != "" {
				acc.ID = s
			}
			if s := stringField(call, "type"); s != "" {
				acc.Type = s
			}
			if extra := call["extra_content"]; extra != nil {
				acc.ExtraContent = cloneJSONValue(extra)
			}
			fragment := ""
			if fn, ok := call["function"].(map[string]any); ok {
				if s := stringField(fn, "name"); s != "" {
					acc.Name = s
				}
				fragment = stringField(fn, "arguments")
			}
			if custom, ok := call["custom"].(map[string]any); ok {
				acc.Type = "custom"
				acc.rawCustomInput = true
				if s := stringField(custom, "name"); s != "" {
					acc.Name = s
				}
				if s := firstTextField(custom, "input", "arguments"); s != "" {
					fragment = s
				}
			}
			if fragment != "" {
				acc.Arguments.WriteString(fragment)
			}
			g.emitToolCallDelta(acc, fragment, sse, respID)
		}
	}
	if fn, ok := delta["function_call"].(map[string]any); ok {
		acc := g.tool(0)
		acc.Type = "function"
		if s := stringField(fn, "name"); s != "" {
			acc.Name = s
		}
		fragment := stringField(fn, "arguments")
		if fragment != "" {
			acc.Arguments.WriteString(fragment)
		}
		g.emitToolCallDelta(acc, fragment, sse, respID)
	}
}

func (g *chatGeneration) emitToolCallDelta(call *chatToolCall, fragment string, sse *responseSSEWriter, respID string) {
	if call == nil {
		return
	}
	if call.Mapping.Kind == "" && call.Name != "" {
		call.Mapping = g.mappingForTool(call.Name, call.Type)
	}
	if call.Mapping.Kind == "" {
		return
	}

	switch call.Mapping.Kind {
	case "custom":
		g.ensureToolActive(call.Index, sse, respID)
		if !call.announced {
			g.announceCustomTool(call, sse, respID)
		}
		decoded := call.Arguments.String()
		if !call.rawCustomInput {
			decoded = decodeCustomInputPrefix(decoded)
		}
		if delta := strings.TrimPrefix(decoded, call.customInputEmitted); delta != "" {
			call.customInputEmitted = decoded
			_ = sse.Event("response.custom_tool_call_input.delta", map[string]any{
				"response_id":  respID,
				"item_id":      call.ItemID,
				"call_id":      call.ID,
				"output_index": call.outputIndex,
				"delta":        delta,
			})
		}
	case "function":
		g.ensureToolActive(call.Index, sse, respID)
		if !call.announced {
			g.announceFunctionTool(call, sse, respID)
		}
		if fragment != "" {
			_ = sse.Event("response.function_call_arguments.delta", map[string]any{
				"response_id":  respID,
				"item_id":      call.ItemID,
				"call_id":      call.ID,
				"output_index": call.outputIndex,
				"delta":        fragment,
			})
		}
	}
}

func (g *chatGeneration) tool(index int) *chatToolCall {
	if existing := g.tools[index]; existing != nil {
		return existing
	}
	call := &chatToolCall{Index: index, Type: "function", outputIndex: -1}
	g.tools[index] = call
	return call
}

func (g *chatGeneration) toolForDelta(call map[string]any, position int) *chatToolCall {
	if index, ok := intFieldOK(call, "index"); ok {
		return g.tool(index)
	}

	id := stringField(call, "id")
	if id != "" {
		if existing := g.toolByID(id); existing != nil {
			return existing
		}
	}

	fallback := position
	if existing := g.tools[fallback]; existing != nil && existing.started() {
		name, _ := toolDeltaNameAndType(call)
		switch {
		case id != "" && existing.ID != "" && existing.ID != id:
			return g.tool(g.nextToolIndex())
		case name != "" && existing.Name != "" && existing.Name != name:
			return g.tool(g.nextToolIndex())
		}
	}
	return g.tool(fallback)
}

func (g *chatGeneration) toolByID(id string) *chatToolCall {
	for _, call := range g.tools {
		if call != nil && call.ID == id {
			return call
		}
	}
	return nil
}

func (g *chatGeneration) nextToolIndex() int {
	next := 0
	for index := range g.tools {
		if index >= next {
			next = index + 1
		}
	}
	return next
}

func (c *chatToolCall) started() bool {
	return c != nil && (c.ID != "" || c.Name != "" || c.Arguments.Len() > 0 || c.announced)
}

func toolDeltaNameAndType(call map[string]any) (string, string) {
	callType := stringField(call, "type")
	if fn, ok := call["function"].(map[string]any); ok {
		return stringField(fn, "name"), callType
	}
	if custom, ok := call["custom"].(map[string]any); ok {
		return stringField(custom, "name"), "custom"
	}
	return "", callType
}

func (g *chatGeneration) ensureMessageActive(sse *responseSSEWriter, respID string) {
	msg := g.messageState()
	if msg.done {
		g.message = &streamedAssistantItem{id: newID("msg"), outputIndex: -1}
		msg = g.message
	}
	if msg.announced {
		if g.activeKind != "message" {
			g.finishActiveItem(sse, respID)
			g.activeKind = "message"
			g.activeToolIndex = -1
		}
		return
	}
	g.finishActiveItem(sse, respID)
	if msg.outputIndex < 0 {
		msg.outputIndex = g.allocateOutputIndex()
	}
	_ = sse.Event("response.output_item.added", map[string]any{
		"response_id":  respID,
		"output_index": msg.outputIndex,
		"item": map[string]any{
			"id":      msg.id,
			"type":    "message",
			"role":    "assistant",
			"content": []any{},
		},
	})
	msg.announced = true
	g.activeKind = "message"
	g.activeToolIndex = -1
}

func (g *chatGeneration) ensureReasoningActive(sse *responseSSEWriter, respID string) {
	section := g.reasoningState()
	if section.done {
		g.reasoning = &streamedReasoningItem{id: newID("rs"), outputIndex: -1}
		section = g.reasoning
	}
	if section.announced {
		if g.activeKind != "reasoning" {
			g.finishActiveItem(sse, respID)
			g.activeKind = "reasoning"
			g.activeToolIndex = -1
		}
		return
	}
	g.finishActiveItem(sse, respID)
	if section.outputIndex < 0 {
		section.outputIndex = g.allocateOutputIndex()
	}
	_ = sse.Event("response.output_item.added", map[string]any{
		"response_id":  respID,
		"output_index": section.outputIndex,
		"item": map[string]any{
			"id":      section.id,
			"type":    "reasoning",
			"summary": []any{},
		},
	})
	section.announced = true
	g.activeKind = "reasoning"
	g.activeToolIndex = -1
}

func (g *chatGeneration) ensureToolActive(index int, sse *responseSSEWriter, respID string) {
	call := g.tool(index)
	if call.announced {
		if g.activeKind != "tool" {
			g.finishActiveItem(sse, respID)
		}
		g.activeKind = "tool"
		g.activeToolIndex = index
		return
	}
	if g.activeKind != "tool" {
		g.finishActiveItem(sse, respID)
	}
	g.activeKind = "tool"
	g.activeToolIndex = index
}

func (g *chatGeneration) announceFunctionTool(call *chatToolCall, sse *responseSSEWriter, respID string) {
	if call == nil || call.announced {
		return
	}
	mapping := g.mappingForTool(call.Name, call.Type)
	call.Mapping = mapping
	displayName := mapping.Name
	if displayName == "" {
		displayName = call.Name
	}
	if displayName == "" {
		displayName = "unknown_tool"
	}
	if call.ID == "" {
		call.ID = newID("call")
	}
	if call.ItemID == "" {
		call.ItemID = newID("fc")
	}
	if call.outputIndex < 0 {
		call.outputIndex = g.allocateOutputIndex()
	}
	item := map[string]any{
		"id":        call.ItemID,
		"type":      "function_call",
		"call_id":   call.ID,
		"name":      displayName,
		"arguments": "",
		"status":    "in_progress",
	}
	if mapping.Namespace != "" {
		item["namespace"] = mapping.Namespace
	}
	if call.ExtraContent != nil {
		item["extra_content"] = cloneJSONValue(call.ExtraContent)
	}
	_ = sse.Event("response.output_item.added", map[string]any{
		"response_id":  respID,
		"output_index": call.outputIndex,
		"item":         item,
	})
	call.announced = true
}

func (g *chatGeneration) announceCustomTool(call *chatToolCall, sse *responseSSEWriter, respID string) {
	if call == nil || call.announced {
		return
	}
	mapping := g.mappingForTool(call.Name, call.Type)
	call.Mapping = mapping
	displayName := mapping.Name
	if displayName == "" {
		displayName = call.Name
	}
	if displayName == "" {
		displayName = "unknown_tool"
	}
	if call.ID == "" {
		call.ID = newID("call")
	}
	if call.ItemID == "" {
		call.ItemID = newID("ctc")
	}
	if call.outputIndex < 0 {
		call.outputIndex = g.allocateOutputIndex()
	}
	item := map[string]any{
		"id":      call.ItemID,
		"type":    "custom_tool_call",
		"call_id": call.ID,
		"name":    displayName,
		"input":   "",
		"status":  "in_progress",
	}
	if call.ExtraContent != nil {
		item["extra_content"] = cloneJSONValue(call.ExtraContent)
	}
	_ = sse.Event("response.output_item.added", map[string]any{
		"response_id":  respID,
		"output_index": call.outputIndex,
		"item":         item,
	})
	call.announced = true
}

func (g *chatGeneration) mappingForTool(name, callType string) toolMapping {
	if g.ctx == nil {
		return inferToolMapping(name, callType)
	}
	if mapping, ok := g.ctx.byChat[name]; ok {
		return mapping
	}
	return inferToolMapping(name, callType)
}

func (g *chatGeneration) allocateOutputIndex() int {
	index := g.nextOutputIndex
	g.nextOutputIndex++
	return index
}

func (g *chatGeneration) finishActiveItem(sse *responseSSEWriter, respID string) {
	switch g.activeKind {
	case "message":
		if g.message != nil && g.message.announced && !g.message.done {
			_ = sse.Event("response.output_item.done", map[string]any{
				"response_id": respID,
				"item":        g.message.responseItem(),
			})
			g.message.done = true
		}
	case "reasoning":
		if g.reasoning != nil && g.reasoning.announced && !g.reasoning.done {
			_ = sse.Event("response.output_item.done", map[string]any{
				"response_id": respID,
				"item":        g.reasoning.responseItem(),
			})
			g.reasoning.done = true
		}
	case "tool":
		call := g.tools[g.activeToolIndex]
		if call != nil && call.announced && !call.done {
			_ = sse.Event("response.output_item.done", map[string]any{
				"response_id": respID,
				"item":        g.toolCallItem(call),
			})
			call.done = true
		}
	}
	g.activeKind = ""
	g.activeToolIndex = -1
}

func generationFromChatResponse(raw []byte, ctx *translationContext) (*chatGeneration, error) {
	var response map[string]any
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, err
	}
	if errValue, ok := response["error"]; ok && errValue != nil {
		return nil, fmt.Errorf("upstream error: %s", compactJSONString(errValue))
	}
	gen := newChatGeneration(ctx)
	if usage, ok := response["usage"].(map[string]any); ok {
		gen.usage = usage
	}
	choices, _ := response["choices"].([]any)
	if len(choices) == 0 {
		return gen, nil
	}
	choice, _ := choices[0].(map[string]any)
	gen.finishReason = stringField(choice, "finish_reason")
	message, _ := choice["message"].(map[string]any)
	if s := textFromAny(message["content"]); s != "" {
		gen.messageState().text.WriteString(s)
	}
	if extra := message["extra_content"]; extra != nil {
		gen.messageState().ExtraContent = cloneJSONValue(extra)
	}
	if s := firstTextField(message, "reasoning_content", "reasoning"); s != "" {
		gen.reasoningState().text.WriteString(s)
	}
	if calls, ok := message["tool_calls"].([]any); ok {
		for position, raw := range calls {
			call, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			index := intField(call, "index", position)
			acc := gen.tool(index)
			acc.ID = stringField(call, "id")
			acc.Type = stringField(call, "type")
			if acc.Type == "" {
				acc.Type = "function"
			}
			if extra := call["extra_content"]; extra != nil {
				acc.ExtraContent = cloneJSONValue(extra)
			}
			if fn, ok := call["function"].(map[string]any); ok {
				acc.Name = stringField(fn, "name")
				acc.Arguments.WriteString(stringField(fn, "arguments"))
			}
			if custom, ok := call["custom"].(map[string]any); ok {
				acc.Type = "custom"
				acc.Name = stringField(custom, "name")
				acc.Arguments.WriteString(firstTextField(custom, "input", "arguments"))
			}
		}
	}
	if fn, ok := message["function_call"].(map[string]any); ok {
		acc := gen.tool(0)
		acc.Type = "function"
		acc.Name = stringField(fn, "name")
		acc.Arguments.WriteString(stringField(fn, "arguments"))
	}
	return gen, nil
}

func emitGenerationOutputItems(gen *chatGeneration, sse *responseSSEWriter, respID string) {
	if gen.activeKind == "tool" {
		gen.activeKind = ""
		gen.activeToolIndex = -1
	} else {
		gen.finishActiveItem(sse, respID)
	}
	for _, entry := range gen.responseItems() {
		if entry.done {
			continue
		}
		_ = sse.Event("response.output_item.done", map[string]any{"item": entry.item})
	}
}

func emitGenerationCompletion(gen *chatGeneration, sse *responseSSEWriter, respID string) {
	if reason, ok := incompleteReasonFromFinishReason(gen.finishReason); ok {
		_ = sse.Event("response.incomplete", map[string]any{
			"response": map[string]any{
				"id":     respID,
				"object": "response",
				"status": "incomplete",
				"incomplete_details": map[string]any{
					"reason": reason,
				},
				"usage": responsesUsage(gen.usage),
			},
		})
		return
	}
	_ = sse.Event("response.completed", map[string]any{
		"response": map[string]any{
			"id":       respID,
			"object":   "response",
			"status":   "completed",
			"usage":    responsesUsage(gen.usage),
			"end_turn": true,
		},
	})
}

func (g *chatGeneration) responseItems() []responseItemState {
	var items []responseItemState
	if reasoning := g.reasoning; reasoning != nil && reasoning.text.Len() > 0 {
		items = append(items, responseItemState{
			item: reasoning.responseItem(),
			done: reasoning.done,
		})
	}
	if text := g.message; text != nil && text.text.Len() > 0 {
		items = append(items, responseItemState{
			item: text.responseItem(),
			done: text.done,
		})
	}
	indexes := make([]int, 0, len(g.tools))
	for index := range g.tools {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	for _, index := range indexes {
		call := g.tools[index]
		if item := g.toolCallItem(call); item != nil {
			items = append(items, responseItemState{
				item: item,
				done: call.done,
			})
		}
	}
	return items
}

func (g *chatGeneration) outputItems() []any {
	states := g.responseItems()
	items := make([]any, 0, len(states))
	for _, state := range states {
		items = append(items, state.item)
	}
	return items
}

func (g *chatGeneration) toolCallItem(call *chatToolCall) map[string]any {
	if call == nil {
		return nil
	}
	name := call.Name
	if name == "" {
		name = "unknown_tool"
	}
	callID := call.ID
	if callID == "" {
		callID = newID("call")
	}
	itemID := call.ItemID
	args := call.Arguments.String()
	mapping := call.Mapping
	if mapping.Kind == "" {
		mapping = g.mappingForTool(name, call.Type)
	}

	switch mapping.Kind {
	case "custom":
		if itemID == "" {
			itemID = newID("ctc")
		}
		displayName := mapping.Name
		if displayName == "" {
			displayName = name
		}
		item := map[string]any{
			"id":      itemID,
			"type":    "custom_tool_call",
			"call_id": callID,
			"name":    displayName,
			"input":   customInputFromArguments(args),
		}
		addExtraContent(item, call.ExtraContent)
		return item
	case "tool_search":
		item := map[string]any{
			"id":        newID("ts"),
			"type":      "tool_search_call",
			"call_id":   callID,
			"execution": "client",
			"arguments": jsonValueOrObject(args),
		}
		addExtraContent(item, call.ExtraContent)
		return item
	case "web_search":
		item := map[string]any{
			"id":     newID("ws"),
			"type":   "web_search_call",
			"status": "completed",
			"action": webSearchActionFromArguments(args),
		}
		addExtraContent(item, call.ExtraContent)
		return item
	case "image_generation":
		action := jsonObject(args)
		result := firstString(action, "result", "b64_json", "image")
		status := "completed"
		if result == "" {
			status = "failed"
		}
		item := map[string]any{
			"id":             newID("ig"),
			"type":           "image_generation_call",
			"status":         status,
			"revised_prompt": firstString(action, "revised_prompt", "prompt"),
			"result":         result,
		}
		addExtraContent(item, call.ExtraContent)
		return item
	default:
		if itemID == "" {
			itemID = newID("fc")
		}
		displayName := mapping.Name
		if displayName == "" {
			displayName = name
		}
		item := map[string]any{
			"id":        itemID,
			"type":      "function_call",
			"call_id":   callID,
			"name":      displayName,
			"arguments": argsOrEmptyObject(args),
		}
		if mapping.Namespace != "" {
			item["namespace"] = mapping.Namespace
		}
		addExtraContent(item, call.ExtraContent)
		return item
	}
}

func addExtraContent(item map[string]any, extra any) {
	if extra != nil {
		item["extra_content"] = cloneJSONValue(extra)
	}
}

func inferToolMapping(name, callType string) toolMapping {
	kind := "function"
	switch name {
	case "tool_search":
		kind = "tool_search"
	case "web_search":
		kind = "web_search"
	case "image_generation":
		kind = "image_generation"
	default:
		if callType == "custom" {
			kind = "custom"
		}
	}
	return toolMapping{Kind: kind, ChatName: name, Name: name}
}

func incompleteReasonFromFinishReason(reason string) (string, bool) {
	switch reason {
	case "length":
		return "max_output_tokens", true
	case "content_filter":
		return "content_filter", true
	default:
		return "", false
	}
}

func responsesUsage(usage map[string]any) map[string]any {
	input := int64FromAny(usage["prompt_tokens"])
	output := int64FromAny(usage["completion_tokens"])
	total := int64FromAny(usage["total_tokens"])
	if total == 0 {
		total = input + output
	}
	reasoningTokens := int64(0)
	if details, ok := usage["completion_tokens_details"].(map[string]any); ok {
		reasoningTokens = int64FromAny(details["reasoning_tokens"])
	}
	var outputDetails any
	if reasoningTokens > 0 {
		outputDetails = map[string]any{"reasoning_tokens": reasoningTokens}
	}
	return map[string]any{
		"input_tokens":          input,
		"input_tokens_details":  nil,
		"output_tokens":         output,
		"output_tokens_details": outputDetails,
		"total_tokens":          total,
	}
}

func decodeCustomInputPrefix(arguments string) string {
	keyStart := strings.Index(arguments, `"input"`)
	if keyStart < 0 {
		return ""
	}
	rest := arguments[keyStart+len(`"input"`):]
	colon := strings.IndexByte(rest, ':')
	if colon < 0 {
		return ""
	}
	rest = strings.TrimLeft(rest[colon+1:], " \t\r\n")
	if !strings.HasPrefix(rest, `"`) {
		return ""
	}
	rest = rest[1:]
	var out strings.Builder
	escaped := false
	for i := 0; i < len(rest); i++ {
		ch := rest[i]
		if escaped {
			switch ch {
			case '"', '\\', '/':
				out.WriteByte(ch)
			case 'b':
				out.WriteByte('\b')
			case 'f':
				out.WriteByte('\f')
			case 'n':
				out.WriteByte('\n')
			case 'r':
				out.WriteByte('\r')
			case 't':
				out.WriteByte('\t')
			case 'u':
				if i+4 >= len(rest) {
					return out.String()
				}
				codePoint, err := strconv.ParseInt(rest[i+1:i+5], 16, 32)
				if err != nil {
					return out.String()
				}
				out.WriteRune(rune(codePoint))
				i += 4
			default:
				out.WriteByte(ch)
			}
			escaped = false
			continue
		}
		switch ch {
		case '\\':
			escaped = true
		case '"':
			return out.String()
		default:
			out.WriteByte(ch)
		}
	}
	return out.String()
}

func customInputFromArguments(arguments string) string {
	if strings.TrimSpace(arguments) == "" {
		return ""
	}
	var value any
	if err := json.Unmarshal([]byte(arguments), &value); err != nil {
		return arguments
	}
	switch v := value.(type) {
	case string:
		return v
	case map[string]any:
		for _, key := range []string{"input", "patch", "content", "text", "command"} {
			if s, ok := v[key].(string); ok {
				return s
			}
		}
		if len(v) == 1 {
			for _, only := range v {
				if s, ok := only.(string); ok {
					return s
				}
			}
		}
		return compactJSONString(v)
	default:
		return compactJSONString(v)
	}
}

func webSearchActionFromArguments(arguments string) map[string]any {
	obj := jsonObject(arguments)
	action := firstString(obj, "action", "type")
	if action == "" {
		switch {
		case firstString(obj, "url") != "" && firstString(obj, "pattern") != "":
			action = "find_in_page"
		case firstString(obj, "url") != "":
			action = "open_page"
		default:
			action = "search"
		}
	}
	out := map[string]any{"type": action}
	switch action {
	case "open_page":
		if s := firstString(obj, "url"); s != "" {
			out["url"] = s
		}
	case "find_in_page":
		if s := firstString(obj, "url"); s != "" {
			out["url"] = s
		}
		if s := firstString(obj, "pattern"); s != "" {
			out["pattern"] = s
		}
	default:
		if s := firstString(obj, "query"); s != "" {
			out["query"] = s
		}
		if queries, ok := obj["queries"].([]any); ok {
			out["queries"] = queries
		}
	}
	return out
}

func webSearchHistoryKey(action any) string {
	if action == nil {
		return ""
	}
	data, err := json.Marshal(action)
	if err != nil {
		data = []byte(fmt.Sprint(action))
	}
	hashInput := make([]byte, 0, len("web_search")+1+len(data))
	hashInput = append(hashInput, "web_search"...)
	hashInput = append(hashInput, 0)
	hashInput = append(hashInput, data...)
	sum := sha256.Sum256(hashInput)
	return hex.EncodeToString(sum[:])
}

func webSearchHistoryCallID(key string, occurrence int) string {
	if len(key) > 16 {
		key = key[:16]
	}
	return fmt.Sprintf("call_web_search_%s_%d", key, occurrence)
}

func webSearchArgumentsFromAction(action map[string]any) string {
	args := map[string]any{}
	for key, value := range action {
		if key == "type" {
			args["action"] = value
			continue
		}
		args[key] = value
	}
	if _, ok := args["action"]; !ok {
		args["action"] = "search"
	}
	return compactJSONString(args)
}

func jsonValueOrObject(raw string) any {
	var value any
	if err := json.Unmarshal([]byte(raw), &value); err == nil {
		return value
	}
	return map[string]any{"query": raw}
}

func jsonObject(raw string) map[string]any {
	var value map[string]any
	if err := json.Unmarshal([]byte(raw), &value); err != nil {
		return map[string]any{}
	}
	return value
}

func cloneJSONValue(value any) any {
	if value == nil {
		return nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return value
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		return value
	}
	return out
}

func argsOrEmptyObject(args string) string {
	if strings.TrimSpace(args) == "" {
		return "{}"
	}
	return args
}

func validFunctionCallArguments(args string) bool {
	if strings.TrimSpace(args) == "" {
		return true
	}
	var value map[string]any
	if err := json.Unmarshal([]byte(args), &value); err != nil {
		return false
	}
	return value != nil
}

func stringField(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if s, ok := m[key].(string); ok {
		return s
	}
	return ""
}

func firstString(m map[string]any, keys ...string) string {
	for _, key := range keys {
		if s, ok := m[key].(string); ok && s != "" {
			return s
		}
	}
	return ""
}

func firstTextField(m map[string]any, keys ...string) string {
	for _, key := range keys {
		if s := textFromAny(m[key]); s != "" {
			return s
		}
	}
	return ""
}

func textFromAny(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, raw := range v {
			part, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			if s := stringField(part, "text"); s != "" {
				b.WriteString(s)
			}
		}
		return b.String()
	default:
		return ""
	}
}

func intField(m map[string]any, key string, fallback int) int {
	if value, ok := intFieldOK(m, key); ok {
		return value
	}
	return fallback
}

func intFieldOK(m map[string]any, key string) (int, bool) {
	if m == nil {
		return 0, false
	}
	switch v := m[key].(type) {
	case float64:
		return int(v), true
	case int:
		return v, true
	default:
		return 0, false
	}
}

func int64FromAny(value any) int64 {
	switch v := value.(type) {
	case float64:
		return int64(v)
	case int64:
		return v
	case int:
		return int64(v)
	case json.Number:
		n, _ := v.Int64()
		return n
	default:
		return 0
	}
}

func compactJSONString(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(data)
}

func newID(prefix string) string {
	return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
}
