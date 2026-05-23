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
	maxWebSearchFollowUps    = 3
	maxWebSearchResults      = 5
	maxWebSearchPageBytes    = 2 << 20
	maxWebSearchExcerptBytes = 6 << 10
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
	if g.message != nil && g.message.text.Len() > 0 {
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
	followUpReq, err := buildWebSearchFollowUpRequest(upstreamReq, call, webSearchText)
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
	a.debug.SaveJSON("web search request", map[string]any{
		"call_id":  call.ID,
		"name":     call.Name,
		"action":   action,
		"raw_args": call.Arguments.String(),
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

	var results []searchResult
	seen := map[string]struct{}{}
	for _, query := range queries {
		queryResults, err := a.duckDuckGoSearch(ctx, query, maxWebSearchResults)
		if err != nil {
			return "", err
		}
		for _, result := range queryResults {
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
			if len(results) >= maxWebSearchResults {
				break
			}
		}
		if len(results) >= maxWebSearchResults {
			break
		}
	}

	return formatWebSearchResults(queries, results), nil
}

func (a *Adapter) duckDuckGoSearch(ctx context.Context, query string, limit int) ([]searchResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	searchURL := "https://html.duckduckgo.com/html/?q=" + url.QueryEscape(query)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, searchURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", webSearchUserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func(body io.ReadCloser) {
		if err := body.Close(); err != nil {
			a.logger.Warn("failed to close search response body", zap.Error(err))
		}
	}(resp.Body)

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxWebSearchPageBytes))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("web search returned HTTP %d", resp.StatusCode)
	}

	doc, err := goquery.NewDocumentFromReader(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	var results []searchResult
	doc.Find(".result").EachWithBreak(func(_ int, s *goquery.Selection) bool {
		if len(results) >= limit {
			return false
		}
		link := s.Find("a.result__a").First()
		if link.Length() == 0 {
			link = s.Find(".result__title a").First()
		}
		href, _ := link.Attr("href")
		title := collapseSearchWhitespace(link.Text())
		snippet := collapseSearchWhitespace(s.Find(".result__snippet").First().Text())
		if title == "" && snippet == "" && href == "" {
			return true
		}
		results = append(results, searchResult{
			Title:   title,
			URL:     normalizeSearchResultURL(href),
			Snippet: snippet,
		})
		return true
	})

	if len(results) == 0 {
		doc.Find("a.result__a").EachWithBreak(func(_ int, s *goquery.Selection) bool {
			if len(results) >= limit {
				return false
			}
			href, _ := s.Attr("href")
			title := collapseSearchWhitespace(s.Text())
			if title == "" && href == "" {
				return true
			}
			results = append(results, searchResult{
				Title: title,
				URL:   normalizeSearchResultURL(href),
			})
			return true
		})
	}

	return results, nil
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
		return "", fmt.Errorf("page fetch returned HTTP %d", resp.StatusCode)
	}

	doc, err := goquery.NewDocumentFromReader(bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	doc.Find("script,style,noscript").Remove()

	title := collapseSearchWhitespace(doc.Find("title").First().Text())
	text := collapseSearchWhitespace(doc.Find("body").Text())
	text = limitWebSearchText(text, maxWebSearchExcerptBytes)

	if title == "" {
		title = u.String()
	}
	if text == "" {
		return fmt.Sprintf("Title: %s\nURL: %s\nExcerpt: [no text extracted]", title, u.String()), nil
	}
	return fmt.Sprintf("Title: %s\nURL: %s\nExcerpt:\n%s", title, u.String(), text), nil
}

func buildWebSearchFollowUpRequest(original map[string]any, call *chatToolCall, resultText string) (map[string]any, error) {
	followUp := cloneJSONValue(original)
	followUpReq, ok := followUp.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected upstream request shape")
	}
	assistantMessage := assistantToolCallMessage(call.ID, call.Name, call.Arguments.String(), call.ExtraContent)
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

func collapseSearchWhitespace(value string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
}

func normalizeSearchResultURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if strings.HasPrefix(raw, "//") {
		raw = "https:" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	if q := u.Query().Get("uddg"); q != "" {
		if decoded, err := url.QueryUnescape(q); err == nil && decoded != "" {
			return decoded
		}
		return q
	}
	if u.Scheme == "" && u.Host == "" {
		return raw
	}
	return u.String()
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
