package adapter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"go.uber.org/zap"
)

type WebSearcher interface {
	Name() string
	Search(ctx context.Context, query string, limit int) ([]searchResult, error)
}

type WebSearchConfig struct {
	Provider string
	Endpoint string
}

func NewWebSearcher(cfg WebSearchConfig, client *http.Client, logger *zap.Logger) (WebSearcher, error) {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	if logger == nil {
		logger = zap.NewNop()
	}

	switch strings.ToLower(strings.TrimSpace(cfg.Provider)) {
	case "", "duckduckgo":
		return &duckDuckGoSearcher{client: client, logger: logger}, nil
	case "duckduckgo-lite":
		return &duckDuckGoSearcher{client: client, logger: logger, liteOnly: true}, nil
	case "searxng":
		endpoint := strings.TrimSpace(cfg.Endpoint)
		if endpoint == "" {
			return nil, fmt.Errorf("search-url is required when search-provider is searxng")
		}
		return &searxngSearcher{endpoint: endpoint, client: client, logger: logger}, nil
	default:
		return nil, fmt.Errorf("unsupported search provider %q", cfg.Provider)
	}
}

type duckDuckGoSearcher struct {
	client    *http.Client
	logger    *zap.Logger
	liteOnly  bool
	endpoints []string
}

func newDuckDuckGoSearcher(client *http.Client, logger *zap.Logger) WebSearcher {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	if logger == nil {
		logger = zap.NewNop()
	}
	return &duckDuckGoSearcher{
		client: client,
		logger: logger,
		endpoints: []string{
			"https://html.duckduckgo.com/html/?q=",
			"https://duckduckgo.com/html/?q=",
			"https://lite.duckduckgo.com/lite/?q=",
		},
	}
}

func (s *duckDuckGoSearcher) Name() string {
	return "duckduckgo"
}

func (s *duckDuckGoSearcher) Search(ctx context.Context, query string, limit int) ([]searchResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	endpoints := s.endpoints
	if s.liteOnly {
		endpoints = endpoints[len(endpoints)-1:]
	}
	var errs []string
	for _, endpoint := range endpoints {
		results, err := s.searchEndpoint(ctx, endpoint, query, limit)
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		if len(results) > 0 {
			return results, nil
		}
	}
	if len(errs) > 0 {
		return nil, fmt.Errorf("duckduckgo search failed: %s", strings.Join(errs, "; "))
	}
	return nil, nil
}

func (s *duckDuckGoSearcher) searchEndpoint(ctx context.Context, endpoint, query string, limit int) ([]searchResult, error) {
	searchURL := endpoint + url.QueryEscape(query)
	body, contentType, err := s.fetch(ctx, searchURL)
	if err != nil {
		return nil, err
	}
	results, err := parseHTMLSearchResults(body, contentType, limit)
	if err != nil {
		return nil, err
	}
	return results, nil
}

func (s *duckDuckGoSearcher) fetch(ctx context.Context, rawURL string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("User-Agent", webSearchUserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, body, err := doRequestWithRetry(ctx, s.client, req, s.logger, "duckduckgo search")
	if err != nil {
		return nil, "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", fmt.Errorf("duckduckgo search returned HTTP %d: %s", resp.StatusCode, collapseSearchWhitespace(string(body)))
	}
	return body, resp.Header.Get("Content-Type"), nil
}

type searxngSearcher struct {
	endpoint string
	client   *http.Client
	logger   *zap.Logger
}

func (s *searxngSearcher) Name() string {
	return "searxng"
}

func (s *searxngSearcher) Search(ctx context.Context, query string, limit int) ([]searchResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	u, err := url.Parse(s.endpoint)
	if err != nil {
		return nil, err
	}
	q := u.Query()
	q.Set("q", query)
	q.Set("format", "json")
	q.Set("language", "en")
	q.Set("safesearch", "0")
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", webSearchUserAgent)
	req.Header.Set("Accept", "application/json, text/plain;q=0.9, */*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, body, err := doRequestWithRetry(ctx, s.client, req, s.logger, "searxng search")
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("searxng search returned HTTP %d: %s", resp.StatusCode, collapseSearchWhitespace(string(body)))
	}

	var parsed struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
			Snippet string `json:"snippet"`
		} `json:"results"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse searxng response: %w", err)
	}

	results := make([]searchResult, 0, min(limit, len(parsed.Results)))
	for _, result := range parsed.Results {
		title := collapseSearchWhitespace(result.Title)
		if title == "" {
			title = collapseSearchWhitespace(result.URL)
		}
		snippet := collapseSearchWhitespace(result.Snippet)
		if snippet == "" {
			snippet = collapseSearchWhitespace(result.Content)
		}
		if title == "" && snippet == "" && strings.TrimSpace(result.URL) == "" {
			continue
		}
		results = append(results, searchResult{
			Title:   title,
			URL:     normalizeSearchResultURL(result.URL),
			Snippet: snippet,
		})
		if len(results) >= limit {
			break
		}
	}
	return results, nil
}

func doRequestWithRetry(ctx context.Context, client *http.Client, req *http.Request, logger *zap.Logger, label string) (*http.Response, []byte, error) {
	if client == nil {
		client = http.DefaultClient
	}
	attempts := 3
	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		resp, err := client.Do(req.Clone(ctx))
		if err != nil {
			lastErr = err
			if attempt == attempts || ctx.Err() != nil {
				return nil, nil, err
			}
			sleepBackoff(ctx, attempt)
			continue
		}
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, maxWebSearchPageBytes))
		closeErr := resp.Body.Close()
		if readErr != nil {
			if attempt == attempts {
				return nil, nil, readErr
			}
			lastErr = readErr
			sleepBackoff(ctx, attempt)
			continue
		}
		if shouldRetrySearchResponse(resp.StatusCode) && attempt < attempts {
			if logger != nil {
				logger.Debug("retrying search request", zap.String("label", label), zap.Int("attempt", attempt), zap.Int("status", resp.StatusCode))
			}
			if closeErr != nil && logger != nil {
				logger.Warn("failed to close search response body", zap.String("label", label), zap.Error(closeErr))
			}
			sleepBackoff(ctx, attempt)
			lastErr = fmt.Errorf("%s returned HTTP %d", label, resp.StatusCode)
			continue
		}
		if closeErr != nil && logger != nil {
			logger.Warn("failed to close search response body", zap.String("label", label), zap.Error(closeErr))
		}
		return resp, body, nil
	}
	if lastErr != nil {
		return nil, nil, lastErr
	}
	return nil, nil, fmt.Errorf("%s failed", label)
}

func shouldRetrySearchResponse(status int) bool {
	switch status {
	case http.StatusTooManyRequests, http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func sleepBackoff(ctx context.Context, attempt int) {
	delay := time.Duration(attempt) * 200 * time.Millisecond
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}

func parseHTMLSearchResults(body []byte, contentType string, limit int) ([]searchResult, error) {
	if !looksLikeHTML(contentType, body) {
		return nil, fmt.Errorf("unexpected search response content type %q", contentType)
	}
	doc, err := goquery.NewDocumentFromReader(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	return extractHTMLSearchResults(doc, limit), nil
}

func extractHTMLSearchResults(doc *goquery.Document, limit int) []searchResult {
	type pattern struct {
		root    string
		link    []string
		snippet []string
	}
	patterns := []pattern{
		{
			root:    ".result",
			link:    []string{"a.result__a", "a[data-testid='result-title-a']", "a.result-link"},
			snippet: []string{".result__snippet", "[data-testid='result-snippet']", ".result-snippet"},
		},
		{
			root:    "div[data-testid='result']",
			link:    []string{"a[data-testid='result-title-a']", "a.result-link", "a"},
			snippet: []string{"[data-testid='result-snippet']", ".result-snippet"},
		},
		{
			root:    "li.result",
			link:    []string{"a.result__a", "a.result-link", "a"},
			snippet: []string{".result__snippet", ".result-snippet"},
		},
	}

	var results []searchResult
	seen := map[string]struct{}{}
	appendResult := func(title, href, snippet string) {
		title = collapseSearchWhitespace(title)
		href = normalizeSearchResultURL(href)
		snippet = collapseSearchWhitespace(snippet)
		key := strings.ToLower(strings.TrimSpace(href))
		if key == "" {
			key = strings.ToLower(strings.TrimSpace(title))
		}
		if key == "" {
			return
		}
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		results = append(results, searchResult{Title: title, URL: href, Snippet: snippet})
	}

	for _, pattern := range patterns {
		if len(results) >= limit {
			break
		}
		doc.Find(pattern.root).EachWithBreak(func(_ int, selection *goquery.Selection) bool {
			if len(results) >= limit {
				return false
			}
			link := firstMatchingSelection(selection, pattern.link...)
			if link == nil {
				return true
			}
			href, _ := link.Attr("href")
			title := link.Text()
			snippet := firstMatchingText(selection, pattern.snippet...)
			if snippet == "" {
				snippet = collapseSearchWhitespace(selection.Text())
			}
			appendResult(title, href, snippet)
			return true
		})
	}

	if len(results) == 0 {
		doc.Find("a[href]").EachWithBreak(func(_ int, selection *goquery.Selection) bool {
			if len(results) >= limit {
				return false
			}
			href, _ := selection.Attr("href")
			title := collapseSearchWhitespace(selection.Text())
			if title == "" && href == "" {
				return true
			}
			appendResult(title, href, "")
			return true
		})
	}

	if len(results) > limit {
		results = results[:limit]
	}
	return results
}

func firstMatchingSelection(selection *goquery.Selection, selectors ...string) *goquery.Selection {
	for _, selector := range selectors {
		if matched := selection.Find(selector).First(); matched.Length() > 0 {
			return matched
		}
	}
	return nil
}

func firstMatchingText(selection *goquery.Selection, selectors ...string) string {
	for _, selector := range selectors {
		if matched := selection.Find(selector).First(); matched.Length() > 0 {
			text := collapseSearchWhitespace(matched.Text())
			if text != "" {
				return text
			}
		}
	}
	return ""
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
