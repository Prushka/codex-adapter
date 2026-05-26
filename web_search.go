package adapter

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/PuerkitoBio/goquery"
	"go.uber.org/zap"
)

const (
	maxWebSearchFollowUps    = 50
	maxWebSearchResults      = 30
	maxWebSearchPageBytes    = 2 << 24
	maxWebSearchExcerptBytes = 6 << 16
)

type searchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

func (g *chatGeneration) singleWebSearchCall() *chatToolCall {
	if len(g.tools) != 1 {
		return nil
	}
	for _, call := range g.tools {
		if call == nil {
			return nil
		}
		mapping := call.Mapping
		if mapping.Kind == "" {
			mapping = g.mappingForTool(call.Name, call.Type)
		}
		if mapping.Kind == "web_search" {
			return call
		}
		return nil
	}
	return nil
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

	call := gen.singleWebSearchCall()
	if call == nil {
		emitGenerationCompletion(gen, sse, respID)
		return nil
	}
	if webSearchDepth >= maxWebSearchFollowUps {
		a.logger.Warn("web search follow-up depth exceeded", zap.String("response_id", respID))
		emitGenerationCompletion(gen, sse, respID)
		return nil
	}

	webSearchText, err := a.executeWebSearch(inbound.Context(), call)
	if err != nil {
		return fmt.Errorf("web search follow-up failed: %w", err)
	}
	a.rememberWebSearchHistory(call, webSearchText)
	reasoningContent := ""
	if a.reasoningHistory == reasoningHistoryReasoningContent {
		reasoningContent = gen.reasoningContent()
	}
	followUpReq, err := buildWebSearchFollowUpRequest(upstreamReq, call, webSearchText, reasoningContent)
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
	return a.handleGeneration(inbound, followUpReq, gen2, ctx, sse, respID, webSearchDepth+1)
}

func (a *Adapter) executeWebSearch(ctx context.Context, call *chatToolCall) (string, error) {
	action := webSearchActionFromArguments(call.Arguments.String())
	search := a.search
	if search == nil {
		search = newGenericWebSearcher(a.client, a.logger, false)
	}
	a.debug.SaveJSON("web search request", map[string]any{
		"call_id":  call.ID,
		"name":     call.Name,
		"action":   action,
		"raw_args": call.Arguments.String(),
		"backend":  search.Name(),
	})

	result, err := a.executeWebSearchAction(ctx, action)
	if err != nil {
		return "", err
	}
	a.debug.SaveJSON("web search response", map[string]any{
		"call_id": call.ID,
		"name":    call.Name,
		"action":  action,
		"result":  result,
		"backend": search.Name(),
	})
	return result, nil
}

func (a *Adapter) executeWebSearchAction(ctx context.Context, action map[string]any) (string, error) {
	switch strings.TrimSpace(stringField(action, "type")) {
	case "open_page":
		return a.executeOpenPage(ctx, stringField(action, "url"))
	case "find_in_page":
		return a.executeFindInPage(ctx, stringField(action, "url"), stringField(action, "pattern"))
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

	var results []searchResult
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
			results = append(results, result)
			if len(results) >= limit {
				break
			}
		}
		if len(results) >= limit {
			break
		}
	}

	if len(results) == 0 && len(searchErrors) > 0 {
		return formatWebSearchFailure(queries, searchErrors), nil
	}
	return formatWebSearchResults(queries, results), nil
}

func (a *Adapter) executeOpenPage(ctx context.Context, rawURL string) (string, error) {
	page, err := a.fetchPageText(ctx, rawURL)
	if err != nil {
		return "", err
	}
	return page, nil
}

func (a *Adapter) executeFindInPage(ctx context.Context, rawURL, pattern string) (string, error) {
	page, err := a.fetchPageText(ctx, rawURL)
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

func (a *Adapter) fetchPageText(ctx context.Context, rawURL string) (string, error) {
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
	title, text, err := extractPageText(body, resp.Header.Get("Content-Type"), u.String())
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

func buildWebSearchFollowUpRequest(original map[string]any, call *chatToolCall, resultText, reasoningContent string) (map[string]any, error) {
	followUp := cloneJSONValue(original)
	followUpReq, ok := followUp.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected upstream request shape")
	}
	assistantMessage := assistantToolCallMessage(call.ID, call.Name, call.Arguments.String(), call.ExtraContent)
	if strings.TrimSpace(reasoningContent) != "" {
		assistantMessage["reasoning_content"] = reasoningContent
	}
	toolMessage := map[string]any{
		"role":         "tool",
		"tool_call_id": call.ID,
		"content":      resultText,
	}
	followUpReq["messages"] = appendChatMessages(followUpReq["messages"], assistantMessage, toolMessage)
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

func webSearchResultLimit(action map[string]any) int {
	switch strings.ToLower(strings.TrimSpace(stringField(action, "search_context_size"))) {
	case "low":
		return 3
	case "high":
		return 8
	default:
		return maxWebSearchResults
	}
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

func formatWebSearchResults(queries []string, results []searchResult) string {
	queryLabel := strings.Join(queries, ", ")
	if queryLabel == "" {
		queryLabel = "web search"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Search results for %q:\n", queryLabel)
	if len(results) == 0 {
		b.WriteString("No results found.")
		return b.String()
	}
	for i, result := range results {
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

func collapseSearchWhitespace(value string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
}

func extractPageText(body []byte, contentType, fallbackTitle string) (string, string, error) {
	if !looksLikeHTML(contentType, body) {
		text := limitWebSearchText(string(body), maxWebSearchExcerptBytes)
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
	text = limitWebSearchText(text, maxWebSearchExcerptBytes)
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
