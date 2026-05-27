package adapter

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"go.uber.org/zap"
)

const (
	maxWebSearchFollowUps = 50
	maxWebSearchPageBytes = 2 << 24

	webSearchLowResultLimit       = 5
	webSearchMediumResultLimit    = 10
	webSearchHighResultLimit      = 20
	webSearchMediumExcerptResults = 5
	webSearchHighExcerptResults   = 8
	webSearchMediumExcerptBytes   = 12 << 10
	webSearchHighExcerptBytes     = 24 << 10
	webSearchLowOpenPageBytes     = 32 << 10
	webSearchMediumOpenPageBytes  = 128 << 10
	webSearchHighOpenPageBytes    = 256 << 10
	searchResultPageFetchTimeout  = 10 * time.Second
)

type searchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

type querySearchResults struct {
	Query   string
	Results []searchResult
}

func (g *chatGeneration) webSearchCalls() []*chatToolCall {
	calls, allWebSearch := g.webSearchCallsForTurn()
	if !allWebSearch {
		return nil
	}
	return calls
}

func (g *chatGeneration) webSearchCallsForTurn() ([]*chatToolCall, bool) {
	if len(g.tools) == 0 {
		return nil, false
	}
	indexes := make([]int, 0, len(g.tools))
	for index := range g.tools {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	calls := make([]*chatToolCall, 0, len(indexes))
	allWebSearch := true
	for _, index := range indexes {
		call := g.tools[index]
		if call == nil {
			allWebSearch = false
			continue
		}
		mapping := call.Mapping
		if mapping.Kind == "" {
			mapping = g.mappingForTool(call.Name, call.Type)
		}
		if mapping.Kind != "web_search" {
			allWebSearch = false
			continue
		}
		calls = append(calls, call)
	}
	return calls, allWebSearch && len(calls) > 0
}

func (a *Adapter) executeAndRememberWebSearchCalls(
	inbound *http.Request,
	calls []*chatToolCall,
	ctx *translationContext,
) ([]string, error) {
	webSearchResults := make([]string, 0, len(calls))
	for _, call := range calls {
		webSearchText, err := a.executeWebSearch(inbound.Context(), call, ctx)
		if err != nil {
			return nil, err
		}
		a.rememberWebSearchHistory(call, webSearchText)
		webSearchResults = append(webSearchResults, webSearchText)
	}
	return webSearchResults, nil
}

func (a *Adapter) handleGeneration(
	inbound *http.Request,
	upstreamReq map[string]any,
	gen *chatGeneration,
	ctx *translationContext,
	sse *responseSSEWriter,
	respID string,
	webSearchDepth int,
) error {
	a.rememberToolExtraContent(gen)
	a.rememberMessageExtraContent(gen)
	emitGenerationOutputItems(gen, sse, respID)

	calls, allWebSearch := gen.webSearchCallsForTurn()
	if len(calls) == 0 {
		emitGenerationCompletion(gen, sse, respID)
		return nil
	}
	if !allWebSearch {
		if _, err := a.executeAndRememberWebSearchCalls(inbound, calls, ctx); err != nil {
			return fmt.Errorf("web search execution failed: %w", err)
		}
		emitGenerationCompletion(gen, sse, respID)
		return nil
	}
	if webSearchDepth >= maxWebSearchFollowUps {
		a.logger.Warn("web search follow-up depth exceeded", zap.String("response_id", respID))
		emitGenerationCompletion(gen, sse, respID)
		return nil
	}

	webSearchResults, err := a.executeAndRememberWebSearchCalls(inbound, calls, ctx)
	if err != nil {
		return fmt.Errorf("web search follow-up failed: %w", err)
	}
	reasoningContent := ""
	if a.reasoningHistory == reasoningHistoryReasoningContent {
		reasoningContent = gen.reasoningContent()
	}
	followUpReq, err := buildWebSearchFollowUpRequest(upstreamReq, calls, webSearchResults, reasoningContent)
	if err != nil {
		return fmt.Errorf("failed to build web search follow-up request: %w", err)
	}
	a.debug.SaveJSON("upstream chat web search follow-up request", followUpReq)

	upstream, err := a.postChat(inbound, followUpReq)
	if err != nil {
		return fmt.Errorf("failed to send web search follow-up request: %w", err)
	}
	defer func(body io.ReadCloser) {
		if err := body.Close(); err != nil {
			a.logger.Warn("failed to close web search follow-up response body", zap.Error(err))
		}
	}(upstream.Body)

	if upstream.StatusCode < 200 || upstream.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(upstream.Body, 4<<20))
		a.debug.SaveRawJSON("upstream chat web search follow-up error", raw)
		errInfo := newUpstreamErrorInfo(upstream.StatusCode, upstream.Header, raw)
		a.logger.Warn("upstream web search follow-up failed",
			errInfo.logFields(respID, a.chatURL)...,
		)
		return fmt.Errorf("upstream web search follow-up returned HTTP %d: %s", upstream.StatusCode, errInfo.message)
	}

	if isEventStream(upstream.Header.Get("Content-Type")) {
		return a.translateChatStream(inbound, upstream.Body, followUpReq, ctx, sse, respID, webSearchDepth+1)
	}

	raw, err := io.ReadAll(upstream.Body)
	if err != nil {
		return fmt.Errorf("failed to read web search follow-up response: %w", err)
	}
	a.debug.SaveRawJSON("upstream chat web search follow-up response", raw)

	gen2, err := generationFromChatResponse(raw, ctx)
	if err != nil {
		return fmt.Errorf("failed to translate web search follow-up response: %w", err)
	}
	a.logUpstreamUsage("web_search_follow_up", respID, gen2.usage)
	return a.handleGeneration(inbound, followUpReq, gen2, ctx, sse, respID, webSearchDepth+1)
}

func (a *Adapter) executeWebSearch(ctx context.Context, call *chatToolCall, tx *translationContext) (string, error) {
	action := webSearchActionFromArguments(call.Arguments.String())
	var defaults map[string]any
	if tx != nil {
		defaults = tx.webSearchDefaults
	}
	executionAction := webSearchExecutionActionFromArguments(call.Arguments.String(), defaults)
	search := a.search
	if search == nil {
		search = newGenericWebSearcher(a.client, a.logger, false)
	}
	a.debug.SaveJSON("web search request", map[string]any{
		"call_id":          call.ID,
		"name":             call.Name,
		"action":           action,
		"execution_action": executionAction,
		"raw_args":         call.Arguments.String(),
		"backend":          search.Name(),
	})

	result, err := a.executeWebSearchAction(ctx, executionAction)
	if err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		msg := limitWebSearchText(err.Error(), 1200)
		result = formatWebSearchActionFailure(executionAction, msg)
		a.debug.SaveJSON("web search action error", map[string]any{
			"call_id":          call.ID,
			"name":             call.Name,
			"action":           action,
			"execution_action": executionAction,
			"error":            msg,
			"backend":          search.Name(),
		})
		a.logger.Warn("web search action failed",
			zap.String("backend", search.Name()),
			zap.String("call_id", call.ID),
			zap.String("error", msg),
		)
	}
	a.debug.SaveJSON("web search response", map[string]any{
		"call_id":          call.ID,
		"name":             call.Name,
		"action":           action,
		"execution_action": executionAction,
		"result":           result,
		"backend":          search.Name(),
	})
	return result, nil
}

func (a *Adapter) executeWebSearchAction(ctx context.Context, action map[string]any) (string, error) {
	switch strings.TrimSpace(stringField(action, "type")) {
	case "open_page":
		return a.executeOpenPage(ctx, stringField(action, "url"), webSearchOpenPageByteLimit(action))
	case "find_in_page":
		return a.executeFindInPage(ctx, stringField(action, "url"), stringField(action, "pattern"), webSearchOpenPageByteLimit(action))
	default:
		return a.executeSearch(ctx, action)
	}
}

func (a *Adapter) executeSearch(ctx context.Context, action map[string]any) (string, error) {
	queries := webSearchQueries(action)
	if len(queries) == 0 {
		return "", fmt.Errorf("web search query is empty")
	}
	queries = expandWebSearchQueries(queries, action)
	limit := webSearchResultLimit(action)
	domains := webSearchAllowedDomains(action)
	search := a.search
	if search == nil {
		search = newGenericWebSearcher(a.client, a.logger, false)
	}

	var groups []querySearchResults
	resultCount := 0
	var searchErrors []string
	seen := map[string]struct{}{}
	for _, query := range queries {
		queryResults, err := search.Search(ctx, query, limit)
		if err != nil {
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			msg := limitWebSearchText(err.Error(), 1200)
			searchErrors = append(searchErrors, fmt.Sprintf("%q: %s", query, msg))
			a.debug.SaveJSON("web search backend error", map[string]any{
				"query":   query,
				"backend": search.Name(),
				"error":   msg,
			})
			a.logger.Warn("web search backend failed",
				zap.String("backend", search.Name()),
				zap.String("query", query),
				zap.Error(err),
			)
			continue
		}
		group := querySearchResults{Query: query}
		for _, result := range queryResults {
			if len(domains) > 0 && !searchResultMatchesAllowedDomains(result, domains) {
				continue
			}
			key := result.URL
			if key == "" {
				key = result.Title
			}
			if key == "" {
				continue
			}
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			group.Results = append(group.Results, result)
			resultCount++
			if len(group.Results) >= limit {
				break
			}
		}
		if len(group.Results) > 0 {
			groups = append(groups, group)
		}
	}
	if searcherAllowsPageExcerptEnrichment(search) {
		groups = a.enrichSearchResultGroups(ctx, groups, webSearchPageExcerptResultLimit(action), webSearchPageExcerptByteLimit(action))
	}

	if resultCount == 0 && len(searchErrors) > 0 {
		return formatWebSearchFailure(queries, searchErrors), nil
	}
	return formatWebSearchResults(queries, groups), nil
}

func (a *Adapter) executeOpenPage(ctx context.Context, rawURL string, limit int) (string, error) {
	page, err := a.fetchPageText(ctx, rawURL, limit)
	if err != nil {
		return "", err
	}
	return page, nil
}

func (a *Adapter) executeFindInPage(ctx context.Context, rawURL, pattern string, limit int) (string, error) {
	page, err := a.fetchPageText(ctx, rawURL, limit)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(pattern) == "" {
		return page, nil
	}
	excerpts := excerptsAround(page, pattern, 160, 3)
	if len(excerpts) == 0 {
		return fmt.Sprintf("No matches found for %q on %s\n\n%s", pattern, rawURL, page), nil
	}
	return fmt.Sprintf("Matches for %q on %s:\n\n%s", pattern, rawURL, strings.Join(excerpts, "\n\n---\n\n")), nil
}

func (a *Adapter) fetchPageText(ctx context.Context, rawURL string, limit int) (string, error) {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return "", fmt.Errorf("web search page url is empty")
	}
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", err
	}
	if u.Scheme == "" {
		u.Scheme = "https"
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("unsupported page URL scheme %q", u.Scheme)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", webSearchUserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, err := a.client.Do(req)
	if err != nil {
		return "", err
	}
	defer func(body io.ReadCloser) {
		if err := body.Close(); err != nil {
			a.logger.Warn("failed to close page response body", zap.Error(err))
		}
	}(resp.Body)

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxWebSearchPageBytes))
	if err != nil {
		return "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("page fetch returned HTTP %d: %s", resp.StatusCode, collapseSearchWhitespace(string(body)))
	}
	title, text, err := extractPageText(body, resp.Header.Get("Content-Type"), u.String(), limit)
	if err != nil {
		return "", err
	}
	if title == "" {
		title = u.String()
	}
	if text == "" {
		return fmt.Sprintf("Title: %s\nURL: %s\nExcerpt: [no text extracted]", title, u.String()), nil
	}
	return fmt.Sprintf("Title: %s\nURL: %s\nExcerpt:\n%s", title, u.String(), text), nil
}

func buildWebSearchFollowUpRequest(original map[string]any, calls []*chatToolCall, resultTexts []string, reasoningContent string) (map[string]any, error) {
	followUp := cloneJSONValue(original)
	followUpReq, ok := followUp.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected upstream request shape")
	}
	if len(calls) == 0 || len(calls) != len(resultTexts) {
		return nil, fmt.Errorf("web search follow-up call/result mismatch")
	}
	assistantMessage := assistantToolCallsMessage(calls)
	if strings.TrimSpace(reasoningContent) != "" {
		assistantMessage["reasoning_content"] = reasoningContent
	}
	messages := []map[string]any{assistantMessage}
	for i, call := range calls {
		messages = append(messages, map[string]any{
			"role":         "tool",
			"tool_call_id": call.ID,
			"name":         call.Name,
			"content":      resultTexts[i],
		})
	}
	followUpReq["messages"] = appendChatMessages(followUpReq["messages"], messages...)
	return followUpReq, nil
}

func appendChatMessages(value any, messages ...map[string]any) []any {
	var out []any
	switch v := value.(type) {
	case []any:
		out = append(out, v...)
	case []map[string]any:
		for _, message := range v {
			out = append(out, message)
		}
	}
	for _, message := range messages {
		out = append(out, message)
	}
	return out
}

func webSearchQueries(action map[string]any) []string {
	var queries []string
	if query := strings.TrimSpace(stringField(action, "query")); query != "" {
		queries = append(queries, query)
	}
	if rawQueries, ok := action["queries"].([]any); ok {
		for _, raw := range rawQueries {
			if query, ok := raw.(string); ok && strings.TrimSpace(query) != "" {
				queries = append(queries, strings.TrimSpace(query))
			}
		}
	}
	return dedupeStrings(queries)
}

func expandWebSearchQueries(queries []string, action map[string]any) []string {
	domains := webSearchAllowedDomains(action)
	if len(domains) == 0 {
		return queries
	}
	var out []string
	for _, query := range queries {
		out = append(out, query)
		for _, domain := range domains {
			out = append(out, fmt.Sprintf("site:%s %s", domain, query))
		}
	}
	return dedupeStrings(out)
}

func webSearchAllowedDomains(action map[string]any) []string {
	var domains []string
	if rawDomains, ok := action["domains"].([]any); ok {
		for _, raw := range rawDomains {
			if domain, ok := raw.(string); ok && strings.TrimSpace(domain) != "" {
				domains = append(domains, strings.TrimSpace(domain))
			}
		}
	}
	if filters, ok := action["filters"].(map[string]any); ok {
		if rawDomains, ok := filters["allowed_domains"].([]any); ok {
			for _, raw := range rawDomains {
				if domain, ok := raw.(string); ok && strings.TrimSpace(domain) != "" {
					domains = append(domains, strings.TrimSpace(domain))
				}
			}
		}
	}
	return dedupeStrings(domains)
}

func searchResultMatchesAllowedDomains(result searchResult, domains []string) bool {
	if len(domains) == 0 {
		return true
	}
	parsed, err := url.Parse(result.URL)
	if err != nil {
		return false
	}
	host := strings.ToLower(strings.TrimSpace(parsed.Hostname()))
	if host == "" {
		return false
	}
	for _, domain := range domains {
		domain = strings.ToLower(strings.TrimSpace(domain))
		if domain == "" {
			continue
		}
		if host == domain || strings.HasSuffix(host, "."+domain) {
			return true
		}
	}
	return false
}

func searcherAllowsPageExcerptEnrichment(search WebSearcher) bool {
	if search == nil {
		return false
	}
	_, ok := search.(searchResultPageEnrichingSearcher)
	return ok
}

func webSearchPageExcerptResultLimit(action map[string]any) int {
	switch webSearchContextSize(action) {
	case "low":
		return 0
	case "high":
		return webSearchHighExcerptResults
	default:
		return webSearchMediumExcerptResults
	}
}

func webSearchPageExcerptByteLimit(action map[string]any) int {
	switch webSearchContextSize(action) {
	case "low":
		return 0
	case "high":
		return webSearchHighExcerptBytes
	default:
		return webSearchMediumExcerptBytes
	}
}

func (a *Adapter) enrichSearchResultGroups(ctx context.Context, groups []querySearchResults, perQueryLimit, excerptLimit int) []querySearchResults {
	if perQueryLimit <= 0 || excerptLimit <= 0 || len(groups) == 0 {
		return groups
	}
	cache := map[string]string{}
	for groupIndex := range groups {
		enriched := 0
		for resultIndex := range groups[groupIndex].Results {
			if enriched >= perQueryLimit {
				break
			}
			result := &groups[groupIndex].Results[resultIndex]
			if !shouldFetchSearchResultPage(result.URL) {
				continue
			}
			key := strings.ToLower(strings.TrimSpace(result.URL))
			excerpt, ok := cache[key]
			if !ok {
				var err error
				excerpt, err = a.fetchSearchResultPageExcerpt(ctx, result.URL, excerptLimit)
				if err != nil {
					if a.logger != nil {
						a.logger.Debug("failed to enrich search result page",
							zap.String("url", result.URL),
							zap.Error(err),
						)
					}
					excerpt = ""
				}
				cache[key] = excerpt
			}
			if excerpt == "" {
				continue
			}
			result.Snippet = mergeSearchResultSnippet(result.Snippet, excerpt)
			enriched++
		}
	}
	return groups
}

func shouldFetchSearchResultPage(rawURL string) bool {
	if rawURL == "" || isBlockedSearchResultURL(rawURL) {
		return false
	}
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return false
	}
	path := strings.ToLower(u.EscapedPath())
	for _, suffix := range []string{
		".7z", ".avi", ".bmp", ".bz2", ".dmg", ".doc", ".docx", ".exe", ".gif", ".gz",
		".ico", ".jpeg", ".jpg", ".mov", ".mp3", ".mp4", ".pdf", ".png", ".ppt", ".pptx",
		".rar", ".svg", ".tar", ".tgz", ".webm", ".webp", ".xls", ".xlsx", ".zip",
	} {
		if strings.HasSuffix(path, suffix) {
			return false
		}
	}
	return true
}

func (a *Adapter) fetchSearchResultPageExcerpt(ctx context.Context, rawURL string, limit int) (string, error) {
	if limit <= 0 {
		return "", nil
	}
	u, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return "", err
	}
	if u.Scheme == "" {
		u.Scheme = "https"
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("unsupported page URL scheme %q", u.Scheme)
	}
	pageCtx, cancel := context.WithTimeout(ctx, searchResultPageFetchTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(pageCtx, http.MethodGet, u.String(), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", webSearchUserAgent)
	req.Header.Set("Accept", "text/html,text/plain;q=0.9,application/xhtml+xml;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	client := a.client
	if client == nil {
		client = http.DefaultClient
	}
	resp, body, err := doRequestWithRetry(pageCtx, client, req, a.logger, "search result page")
	if err != nil {
		return "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", searchHTTPError("search result page", resp, body)
	}
	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if contentType != "" && !strings.Contains(contentType, "html") && !strings.Contains(contentType, "xml") && !strings.Contains(contentType, "text/plain") {
		return "", fmt.Errorf("unsupported page content type %q", resp.Header.Get("Content-Type"))
	}
	_, text, err := extractPageText(body, resp.Header.Get("Content-Type"), u.String(), limit)
	if err != nil {
		return "", err
	}
	return limitWebSearchText(text, limit), nil
}

func mergeSearchResultSnippet(snippet, excerpt string) string {
	snippet = collapseSearchWhitespace(snippet)
	excerpt = collapseSearchWhitespace(excerpt)
	if excerpt == "" {
		return snippet
	}
	if snippet == "" {
		return "Page excerpt: " + excerpt
	}
	snippetKey := strings.ToLower(snippet)
	excerptKey := strings.ToLower(excerpt)
	if strings.Contains(snippetKey, excerptKey) {
		return snippet
	}
	if len(excerptKey) > 160 && strings.Contains(snippetKey, excerptKey[:160]) {
		return snippet
	}
	return snippet + "\nPage excerpt: " + excerpt
}

func webSearchResultLimit(action map[string]any) int {
	if limit := positiveIntField(action, "limit", "max_results"); limit > 0 {
		if limit > webSearchHighResultLimit {
			return webSearchHighResultLimit
		}
		return limit
	}
	switch webSearchContextSize(action) {
	case "low":
		return webSearchLowResultLimit
	case "high":
		return webSearchHighResultLimit
	default:
		return webSearchMediumResultLimit
	}
}

func webSearchOpenPageByteLimit(action map[string]any) int {
	switch webSearchContextSize(action) {
	case "low":
		return webSearchLowOpenPageBytes
	case "high":
		return webSearchHighOpenPageBytes
	default:
		return webSearchMediumOpenPageBytes
	}
}

func webSearchContextSize(action map[string]any) string {
	size := strings.ToLower(strings.TrimSpace(stringField(action, "search_context_size")))
	switch size {
	case "low", "medium", "high":
		return size
	default:
		return "high"
	}
}

func positiveIntField(obj map[string]any, keys ...string) int {
	for _, key := range keys {
		switch v := obj[key].(type) {
		case int:
			if v > 0 {
				return v
			}
		case int64:
			if v > 0 {
				return int(v)
			}
		case float64:
			if v > 0 {
				return int(v)
			}
		}
	}
	return 0
}

func dedupeStrings(values []string) []string {
	seen := map[string]struct{}{}
	var out []string
	for _, value := range values {
		key := strings.TrimSpace(strings.ToLower(value))
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, value)
	}
	return out
}

func formatWebSearchResults(queries []string, groups []querySearchResults) string {
	queryLabel := strings.Join(queries, ", ")
	if queryLabel == "" {
		queryLabel = "web search"
	}
	var b strings.Builder
	if len(groups) == 0 {
		fmt.Fprintf(&b, "Search results for %q:\n", queryLabel)
		b.WriteString("No results found.")
		return b.String()
	}
	for groupIndex, group := range groups {
		if groupIndex > 0 {
			b.WriteString("\n\n")
		}
		label := strings.TrimSpace(group.Query)
		if label == "" {
			label = queryLabel
		}
		fmt.Fprintf(&b, "Search results for %q:\n", label)
		for i, result := range group.Results {
			if i > 0 {
				b.WriteString("\n\n")
			}
			fmt.Fprintf(&b, "%d. %s\n", i+1, strings.TrimSpace(result.Title))
			if result.URL != "" {
				fmt.Fprintf(&b, "URL: %s\n", result.URL)
			}
			if result.Snippet != "" {
				b.WriteString(result.Snippet)
			}
		}
		if len(group.Results) == 0 {
			b.WriteString("No results found.")
		}
	}
	return strings.TrimSpace(b.String())
}

func formatWebSearchFailure(queries []string, errs []string) string {
	var b strings.Builder
	b.WriteString(formatWebSearchResults(queries, nil))
	if len(errs) == 0 {
		return b.String()
	}
	b.WriteString("\n\nSearch backend errors:")
	for _, errText := range errs {
		b.WriteString("\n- ")
		b.WriteString(limitWebSearchText(errText, 1200))
	}
	return strings.TrimSpace(b.String())
}

func formatWebSearchActionFailure(action map[string]any, errText string) string {
	actionType := strings.TrimSpace(stringField(action, "type"))
	if actionType == "" {
		actionType = "search"
	}
	target := ""
	switch actionType {
	case "open_page", "find_in_page":
		target = strings.TrimSpace(stringField(action, "url"))
	default:
		target = strings.TrimSpace(stringField(action, "query"))
	}
	var b strings.Builder
	b.WriteString("Web search action failed.\n")
	b.WriteString("Action: ")
	b.WriteString(actionType)
	b.WriteByte('\n')
	if target != "" {
		b.WriteString("Target: ")
		b.WriteString(target)
		b.WriteByte('\n')
	}
	if errText != "" {
		b.WriteString("Error: ")
		b.WriteString(errText)
		b.WriteByte('\n')
	}
	b.WriteString("Use the available search results or try another source if needed.")
	return strings.TrimSpace(b.String())
}

func collapseSearchWhitespace(value string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
}

func extractPageText(body []byte, contentType, fallbackTitle string, limit int) (string, string, error) {
	if limit <= 0 {
		limit = webSearchMediumOpenPageBytes
	}
	if !looksLikeHTML(contentType, body) {
		text := limitWebSearchText(string(body), limit)
		return fallbackTitle, text, nil
	}

	doc, err := goquery.NewDocumentFromReader(bytes.NewReader(body))
	if err != nil {
		return "", "", err
	}
	doc.Find("script,style,noscript").Remove()

	title := collapseSearchWhitespace(doc.Find("title").First().Text())
	bodySelection := doc.Find("main,article,body").First()
	text := ""
	if bodySelection.Length() > 0 {
		text = collapseSearchWhitespace(bodySelection.Text())
	}
	if text == "" {
		text = collapseSearchWhitespace(doc.Text())
	}
	text = limitWebSearchText(text, limit)
	return title, text, nil
}

func looksLikeHTML(contentType string, body []byte) bool {
	contentType = strings.ToLower(contentType)
	if strings.Contains(contentType, "html") || strings.Contains(contentType, "xml") {
		return true
	}
	trimmed := bytes.TrimSpace(body)
	return len(trimmed) > 0 && len(trimmed) < maxWebSearchPageBytes && trimmed[0] == '<'
}

func limitWebSearchText(text string, limit int) string {
	text = collapseSearchWhitespace(text)
	if len(text) <= limit {
		return text
	}
	if limit <= 0 {
		return ""
	}
	trimmed, _ := truncateString(text, limit)
	return trimmed
}

func excerptsAround(text, pattern string, radius, limit int) []string {
	needle := strings.ToLower(strings.TrimSpace(pattern))
	if needle == "" || text == "" {
		return nil
	}
	source := strings.ToLower(text)
	var excerpts []string
	start := 0
	for len(excerpts) < limit {
		idx := strings.Index(source[start:], needle)
		if idx < 0 {
			break
		}
		idx += start
		begin := idx - radius
		if begin < 0 {
			begin = 0
		}
		end := idx + len(needle) + radius
		if end > len(text) {
			end = len(text)
		}
		excerpt := strings.TrimSpace(text[begin:end])
		if excerpt != "" {
			excerpts = append(excerpts, excerpt)
		}
		start = idx + len(needle)
	}
	return excerpts
}

var webSearchUserAgent = strings.Join([]string{
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
	"AppleWebKit/537.36 (KHTML, like Gecko)",
	"Chrome/124.0.0.0",
	"Safari/537.36",
}, " ")
