package adapter

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"go.uber.org/zap"
)

type searchResultPattern struct {
	root    string
	link    []string
	snippet []string
}

type fallbackSearcher struct {
	name      string
	searchers []WebSearcher
	logger    *zap.Logger
}

func newGenericWebSearcher(client *http.Client, logger *zap.Logger, liteOnly bool) WebSearcher {
	searchers := []WebSearcher{
		newDuckDuckGoSearcher(client, logger, liteOnly),
		newBingSearcher(client, logger),
		newYahooSearcher(client, logger),
	}
	return newFallbackSearcher(logger, searchers...)
}

func newFallbackSearcher(logger *zap.Logger, searchers ...WebSearcher) WebSearcher {
	names := make([]string, 0, len(searchers))
	filtered := make([]WebSearcher, 0, len(searchers))
	for _, searcher := range searchers {
		if searcher == nil {
			continue
		}
		names = append(names, searcher.Name())
		filtered = append(filtered, searcher)
	}
	if logger == nil {
		logger = zap.NewNop()
	}
	return &fallbackSearcher{
		name:      strings.Join(names, "+"),
		searchers: filtered,
		logger:    logger,
	}
}

func (s *fallbackSearcher) Name() string {
	if s == nil || s.name == "" {
		return "search"
	}
	return s.name
}

func (s *fallbackSearcher) allowSearchResultPageEnrichment() {}

func (s *fallbackSearcher) Search(ctx context.Context, query string, limit int) ([]searchResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	var errs []string
	for _, searcher := range s.searchers {
		results, err := searcher.Search(ctx, query, limit)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", searcher.Name(), err))
			if s.logger != nil {
				s.logger.Debug("search backend failed",
					zap.String("backend", searcher.Name()),
					zap.String("query", query),
					zap.Error(err),
				)
			}
			continue
		}
		if len(results) == 0 {
			if s.logger != nil {
				s.logger.Debug("search backend returned no results",
					zap.String("backend", searcher.Name()),
					zap.String("query", query),
				)
			}
			continue
		}
		if s.logger != nil {
			s.logger.Debug("search backend returned results",
				zap.String("backend", searcher.Name()),
				zap.String("query", query),
				zap.Int("results", len(results)),
			)
		}
		return results, nil
	}
	if len(errs) > 0 {
		return nil, fmt.Errorf("%s", strings.Join(errs, "; "))
	}
	return nil, nil
}

type bingSearcher struct {
	client *http.Client
	logger *zap.Logger
}

func newBingSearcher(client *http.Client, logger *zap.Logger) WebSearcher {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	if logger == nil {
		logger = zap.NewNop()
	}
	return &bingSearcher{client: client, logger: logger}
}

func (s *bingSearcher) Name() string {
	return "bing"
}

func (s *bingSearcher) allowSearchResultPageEnrichment() {}

func (s *bingSearcher) Search(ctx context.Context, query string, limit int) ([]searchResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	searchURL := "https://www.bing.com/search?q=" + url.QueryEscape(query) + "&setlang=en-US"
	body, contentType, err := s.fetch(ctx, searchURL)
	if err != nil {
		return nil, err
	}
	results, err := parseHTMLSearchResultsWithPatterns(body, contentType, limit, bingSearchPatterns())
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		if challenge := searchChallengeError("bing", body); challenge != nil {
			return nil, challenge
		}
	}
	return results, nil
}

func (s *bingSearcher) fetch(ctx context.Context, rawURL string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("User-Agent", webSearchUserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, body, err := doRequestWithRetry(ctx, s.client, req, s.logger, "bing search")
	if err != nil {
		return nil, "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", searchHTTPError("bing", resp, body)
	}
	return body, resp.Header.Get("Content-Type"), nil
}

type yahooSearcher struct {
	client *http.Client
	logger *zap.Logger
}

func newYahooSearcher(client *http.Client, logger *zap.Logger) WebSearcher {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	if logger == nil {
		logger = zap.NewNop()
	}
	return &yahooSearcher{client: client, logger: logger}
}

func (s *yahooSearcher) Name() string {
	return "yahoo"
}

func (s *yahooSearcher) allowSearchResultPageEnrichment() {}

func (s *yahooSearcher) Search(ctx context.Context, query string, limit int) ([]searchResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	searchURL := "https://search.yahoo.com/search?p=" + url.QueryEscape(query) + "&ei=UTF-8"
	body, contentType, err := s.fetch(ctx, searchURL)
	if err != nil {
		return nil, err
	}
	results, err := parseHTMLSearchResultsWithPatterns(body, contentType, limit, yahooSearchPatterns())
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		if challenge := searchChallengeError("yahoo", body); challenge != nil {
			return nil, challenge
		}
	}
	return results, nil
}

func (s *yahooSearcher) fetch(ctx context.Context, rawURL string) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("User-Agent", webSearchUserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	resp, body, err := doRequestWithRetry(ctx, s.client, req, s.logger, "yahoo search")
	if err != nil {
		return nil, "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", searchHTTPError("yahoo", resp, body)
	}
	return body, resp.Header.Get("Content-Type"), nil
}

func parseHTMLSearchResultsWithPatterns(body []byte, contentType string, limit int, patterns []searchResultPattern) ([]searchResult, error) {
	if !looksLikeHTML(contentType, body) {
		return nil, fmt.Errorf("unexpected search response content type %q", contentType)
	}
	doc, err := goquery.NewDocumentFromReader(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	return extractSearchResults(doc, limit, patterns), nil
}

func extractSearchResults(doc *goquery.Document, limit int, patterns []searchResultPattern) []searchResult {
	var results []searchResult
	if limit <= 0 {
		return results
	}
	seen := map[string]struct{}{}
	appendResult := func(title, href, snippet string) {
		title = collapseSearchWhitespace(title)
		if isBlockedSearchResultURL(href) {
			return
		}
		href = normalizeSearchResultURL(href)
		if isBlockedSearchResultURL(href) {
			return
		}
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

func duckDuckGoSearchPatterns() []searchResultPattern {
	return []searchResultPattern{
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
}

func bingSearchPatterns() []searchResultPattern {
	return []searchResultPattern{
		{
			root:    "li.b_algo",
			link:    []string{"h2 a", ".b_tpcn a", "a"},
			snippet: []string{".b_caption p", ".b_caption .b_lineclamp2", ".b_snippet", ".b_paractl"},
		},
		{
			root:    "div.b_algo",
			link:    []string{"h2 a", ".b_tpcn a", "a"},
			snippet: []string{".b_caption p", ".b_caption .b_lineclamp2", ".b_snippet"},
		},
	}
}

func yahooSearchPatterns() []searchResultPattern {
	return []searchResultPattern{
		{
			root:    "#web .algo-sr",
			link:    []string{".compTitle a", "h3 a", "a"},
			snippet: []string{".compText p", ".compText", ".compSnippet", ".compText .fc-dustygray"},
		},
		{
			root:    "#web .algo",
			link:    []string{".compTitle a", "h3 a", "a"},
			snippet: []string{".compText p", ".compText", ".compSnippet"},
		},
		{
			root:    "#web .dd",
			link:    []string{".compTitle a", "h3 a", "a"},
			snippet: []string{".compText p", ".compText", ".compSnippet"},
		},
	}
}

func searchHTTPError(provider string, resp *http.Response, body []byte) error {
	return fmt.Errorf("%s search returned HTTP %d (%s): %s", provider, resp.StatusCode, resp.Header.Get("Content-Type"), searchBodySnippet(body, 240))
}

func searchChallengeError(provider string, body []byte) error {
	text := strings.ToLower(collapseSearchWhitespace(string(body)))
	markers := []string{
		"captcha",
		"verify you are human",
		"verify you are a human",
		"unusual traffic",
		"security check",
		"press and hold",
		"robot",
		"enable javascript",
		"access denied",
		"blocked",
	}
	for _, marker := range markers {
		if strings.Contains(text, marker) {
			return fmt.Errorf("%s search returned a challenge page (%s): %s", provider, marker, searchBodySnippet(body, 240))
		}
	}
	return nil
}

func searchBodySnippet(body []byte, limit int) string {
	return limitWebSearchText(collapseSearchWhitespace(string(body)), limit)
}

func normalizeSearchResultURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if isBlockedSearchResultURL(raw) {
		return ""
	}
	if decoded := normalizeDuckDuckGoSearchResultURL(raw); decoded != "" {
		return decoded
	}
	if decoded := normalizeBingSearchResultURL(raw); decoded != "" {
		return decoded
	}
	if decoded := normalizeYahooSearchResultURL(raw); decoded != "" {
		return decoded
	}
	if strings.HasPrefix(raw, "//") {
		raw = "https:" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	if u.Scheme == "" && u.Host == "" {
		return raw
	}
	if isBlockedSearchResultURL(u.String()) {
		return ""
	}
	return u.String()
}

func isBlockedSearchResultURL(raw string) bool {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return false
	}
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	host := strings.ToLower(strings.TrimSpace(u.Hostname()))
	path := strings.ToLower(strings.TrimSpace(u.EscapedPath()))
	query := u.Query()
	if query.Get("ad_domain") != "" || query.Get("ad_provider") != "" || query.Get("ad_type") != "" {
		return true
	}
	if isHostOrSubdomain(host, "duckduckgo.com") && path == "/y.js" {
		return true
	}
	if isHostOrSubdomain(host, "bing.com") && strings.Contains(path, "/aclick") {
		return true
	}
	switch host {
	case "googleadservices.com", "www.googleadservices.com", "doubleclick.net", "www.doubleclick.net", "clickserve.dartsearch.net":
		return true
	default:
		return false
	}
}

func normalizeDuckDuckGoSearchResultURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	if q := u.Query().Get("uddg"); q != "" {
		if decoded, err := url.QueryUnescape(q); err == nil && decoded != "" {
			return decoded
		}
		return q
	}
	return ""
}

func normalizeBingSearchResultURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	host := strings.ToLower(strings.TrimSpace(u.Hostname()))
	if !isHostOrSubdomain(host, "bing.com") {
		return ""
	}
	redirect := strings.TrimSpace(u.Query().Get("u"))
	if redirect == "" {
		return ""
	}
	if decoded := decodeBingRedirectTarget(redirect); decoded != "" {
		return decoded
	}
	return redirect
}

func decodeBingRedirectTarget(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if decoded, err := url.QueryUnescape(raw); err == nil && decoded != "" {
		raw = decoded
	}
	candidates := []string{raw}
	if len(raw) > 2 && raw[0] == 'a' && raw[1] >= '0' && raw[1] <= '9' {
		candidates = append([]string{raw[2:]}, candidates...)
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		for _, enc := range []*base64.Encoding{
			base64.RawStdEncoding,
			base64.RawURLEncoding,
			base64.StdEncoding,
			base64.URLEncoding,
		} {
			decoded, err := enc.DecodeString(candidate)
			if err != nil {
				continue
			}
			value := strings.TrimSpace(string(decoded))
			if value == "" {
				continue
			}
			if unescaped, err := url.QueryUnescape(value); err == nil && unescaped != "" {
				value = unescaped
			}
			if strings.HasPrefix(value, "http://") || strings.HasPrefix(value, "https://") {
				return value
			}
		}
	}
	return ""
}

func normalizeYahooSearchResultURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	host := strings.ToLower(strings.TrimSpace(u.Hostname()))
	if !isHostOrSubdomain(host, "search.yahoo.com") {
		return ""
	}
	if q := u.Query().Get("RU"); q != "" {
		if decoded, err := url.QueryUnescape(q); err == nil && decoded != "" {
			return decoded
		}
		if decoded, err := url.PathUnescape(q); err == nil && decoded != "" {
			return decoded
		}
		return q
	}
	if idx := strings.Index(raw, "RU="); idx >= 0 {
		target := raw[idx+3:]
		for _, sep := range []string{"/RK=", "&RK=", "?RK=", "#"} {
			if cut := strings.Index(target, sep); cut >= 0 {
				target = target[:cut]
			}
		}
		if decoded, err := url.QueryUnescape(target); err == nil && decoded != "" {
			return decoded
		}
		if decoded, err := url.PathUnescape(target); err == nil && decoded != "" {
			return decoded
		}
		if strings.HasPrefix(target, "http://") || strings.HasPrefix(target, "https://") {
			return target
		}
	}
	return ""
}

func isHostOrSubdomain(host, domain string) bool {
	host = strings.ToLower(strings.TrimSpace(host))
	domain = strings.ToLower(strings.TrimSpace(domain))
	return host == domain || strings.HasSuffix(host, "."+domain)
}
