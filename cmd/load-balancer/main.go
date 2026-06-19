package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

const (
	maxRequestBodyBytes         = 128 << 20
	defaultAttempts             = 5
	defaultAttemptDelay         = time.Minute
	defaultTimeout              = 30 * time.Minute
	defaultModelRefreshInterval = 5 * time.Minute
	defaultModelRefreshTimeout  = 30 * time.Second
	defaultProviderCooldown     = time.Minute
	defaultCooldownFailures     = 3
	maxRecentProviderFailures   = 10
)

type cliConfig struct {
	listenAddr       string
	apiKey           string
	providers        providerList
	timeout          time.Duration
	attempts         int
	delay            time.Duration
	refresh          time.Duration
	refreshTimeout   time.Duration
	cooldown         time.Duration
	cooldownFailures int
}

func main() {
	logger, err := newLogger()
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = logger.Sync() }()

	cfg := parseFlags()
	if err := cfg.validate(); err != nil {
		exitWithError(logger, err.Error())
	}

	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DisableCompression = true
	client := &http.Client{
		Timeout:   cfg.timeout,
		Transport: transport,
	}

	handler, err := newProxy(proxyConfig{
		Providers:            cfg.providers,
		Client:               client,
		Logger:               logger,
		APIKey:               cfg.apiKey,
		Attempts:             cfg.attempts,
		Delay:                cfg.delay,
		ModelRefreshInterval: cfg.refresh,
		ModelRefreshTimeout:  cfg.refreshTimeout,
		ProviderCooldown:     cfg.cooldown,
		CooldownFailures:     cfg.cooldownFailures,
	})
	if err != nil {
		exitWithError(logger, "failed to create load balancer", zap.Error(err))
	}
	defer handler.Close()

	logger.Info("OpenAI-compatible load balancer listening",
		zap.String("listen", cfg.listenAddr),
		zap.Int("providers", handler.pool.providerCount()),
		zap.Int("models", handler.pool.modelCount()),
		zap.Int("attempts", handler.attempts),
		zap.Duration("delay", handler.delay),
		zap.Duration("model_refresh_interval", cfg.refresh),
		zap.Duration("model_refresh_timeout", cfg.refreshTimeout),
		zap.Duration("provider_cooldown", cfg.cooldown),
		zap.Int("cooldown_failures", cfg.cooldownFailures),
	)

	server := &http.Server{
		Addr:              cfg.listenAddr,
		Handler:           handler,
		ReadHeaderTimeout: 30 * time.Second,
	}
	if err := server.ListenAndServe(); err != nil {
		exitWithError(logger, "server stopped", zap.Error(err))
	}
}

func parseFlags() cliConfig {
	var cfg cliConfig
	flag.StringVar(&cfg.listenAddr, "listen", "127.0.0.1:18081", "local listening address for Chat Completions and Responses requests")
	flag.StringVar(&cfg.apiKey, "api-key", "", "load balancer API key required from downstream requests")
	flag.Var(&cfg.providers, "provider", "upstream provider tuple id,url,key; may be repeated")
	flag.DurationVar(&cfg.timeout, "timeout", defaultTimeout, "upstream request timeout")
	flag.IntVar(&cfg.attempts, "attempts", defaultAttempts, "number of full provider-pool passes before returning failure")
	flag.DurationVar(&cfg.delay, "delay", defaultAttemptDelay, "delay between full provider-pool attempts; first attempt has no delay")
	flag.DurationVar(&cfg.refresh, "model-refresh-interval", defaultModelRefreshInterval, "interval for refreshing provider /models lists; set to 0 to disable")
	flag.DurationVar(&cfg.refreshTimeout, "model-refresh-timeout", defaultModelRefreshTimeout, "timeout for each provider model refresh request; set to 0 to use the HTTP client timeout")
	flag.DurationVar(&cfg.cooldown, "provider-cooldown", defaultProviderCooldown, "duration to skip a provider after repeated request failures; set to 0 to disable")
	flag.IntVar(&cfg.cooldownFailures, "provider-cooldown-failures", defaultCooldownFailures, "consecutive request failures before provider cooldown; set to 0 to disable")
	flag.Parse()
	return cfg
}

func (c cliConfig) validate() error {
	switch {
	case strings.TrimSpace(c.apiKey) == "":
		return errors.New("missing required flag: -api-key")
	case len(c.providers) == 0:
		return errors.New("missing required flag: -provider")
	case c.attempts < 1:
		return errors.New("attempts must be at least 1")
	case c.delay < 0:
		return errors.New("delay must not be negative")
	case c.refresh < 0:
		return errors.New("model refresh interval must not be negative")
	case c.refreshTimeout < 0:
		return errors.New("model refresh timeout must not be negative")
	case c.cooldown < 0:
		return errors.New("provider cooldown must not be negative")
	case c.cooldownFailures < 0:
		return errors.New("provider cooldown failures must not be negative")
	default:
		seen := map[string]struct{}{}
		for _, provider := range c.providers {
			if _, ok := seen[provider.ID]; ok {
				return fmt.Errorf("duplicate provider id %q", provider.ID)
			}
			seen[provider.ID] = struct{}{}
		}
		return nil
	}
}

type providerList []providerConfig

func (p *providerList) String() string {
	if p == nil {
		return ""
	}
	parts := make([]string, 0, len(*p))
	for _, provider := range *p {
		parts = append(parts, provider.ID+","+provider.ProviderURL+",<redacted>")
	}
	return strings.Join(parts, ";")
}

func (p *providerList) Set(value string) error {
	provider, err := parseProviderConfig(value)
	if err != nil {
		return err
	}
	*p = append(*p, provider)
	return nil
}

func parseProviderConfig(value string) (providerConfig, error) {
	parts := strings.SplitN(value, ",", 3)
	if len(parts) != 3 {
		return providerConfig{}, fmt.Errorf("invalid provider %q: expected id,url,key", value)
	}
	cfg := providerConfig{
		ID:          strings.TrimSpace(parts[0]),
		ProviderURL: strings.TrimSpace(parts[1]),
		APIKey:      strings.TrimSpace(parts[2]),
	}
	switch {
	case cfg.ID == "":
		return providerConfig{}, fmt.Errorf("invalid provider %q: id is required", value)
	case strings.ContainsAny(cfg.ID, " \t\r\n,"):
		return providerConfig{}, fmt.Errorf("invalid provider id %q: id must not contain whitespace or commas", cfg.ID)
	case cfg.ProviderURL == "":
		return providerConfig{}, fmt.Errorf("invalid provider %q: url is required", value)
	case cfg.APIKey == "":
		return providerConfig{}, fmt.Errorf("invalid provider %q: key is required", value)
	default:
		return cfg, nil
	}
}

type proxyConfig struct {
	Providers            []providerConfig
	Client               *http.Client
	Logger               *zap.Logger
	APIKey               string
	Attempts             int
	Delay                time.Duration
	ModelRefreshInterval time.Duration
	ModelRefreshTimeout  time.Duration
	ProviderCooldown     time.Duration
	CooldownFailures     int
}

type proxy struct {
	pool          *providerPool
	client        *http.Client
	logger        *zap.Logger
	apiKey        string
	attempts      int
	delay         time.Duration
	closeOnce     sync.Once
	refreshCancel context.CancelFunc
	refreshDone   <-chan struct{}
}

func newProxy(cfg proxyConfig) (*proxy, error) {
	client := cfg.Client
	if client == nil {
		return nil, errors.New("http client is required")
	}
	apiKey := strings.TrimSpace(cfg.APIKey)
	if apiKey == "" {
		return nil, errors.New("load balancer API key is required")
	}
	logger := cfg.Logger
	if logger == nil {
		logger = zap.NewNop()
	}
	attempts := cfg.Attempts
	if attempts < 1 {
		attempts = defaultAttempts
	}
	pool, err := newProviderPool(context.Background(), cfg.Providers, client, logger)
	if err != nil {
		return nil, err
	}
	pool.setCooldownPolicy(cfg.CooldownFailures, cfg.ProviderCooldown)
	pr := &proxy{
		pool:     pool,
		client:   client,
		logger:   logger,
		apiKey:   apiKey,
		attempts: attempts,
		delay:    cfg.Delay,
	}
	if cfg.ModelRefreshInterval > 0 {
		refreshTimeout := cfg.ModelRefreshTimeout
		if refreshTimeout < 0 {
			refreshTimeout = defaultModelRefreshTimeout
		}
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		pr.refreshCancel = cancel
		pr.refreshDone = done
		go pr.refreshModelsLoop(ctx, cfg.ModelRefreshInterval, refreshTimeout, done)
	}
	return pr, nil
}

func (p *proxy) Close() {
	p.closeOnce.Do(func() {
		if p.refreshCancel != nil {
			p.refreshCancel()
		}
		if p.refreshDone != nil {
			<-p.refreshDone
		}
	})
}

func (p *proxy) refreshModelsLoop(ctx context.Context, interval, timeout time.Duration, done chan<- struct{}) {
	defer close(done)
	timer := time.NewTimer(interval)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			p.pool.refreshModels(ctx, p.client, p.logger, timeout)
			timer.Reset(interval)
		}
	}
}

type providerConfig struct {
	ID          string
	ProviderURL string
	APIKey      string
}

type provider struct {
	id                      string
	chatURL                 string
	responsesURL            string
	modelsURL               string
	apiKey                  string
	models                  map[string]struct{}
	busy                    int
	recentFailures          []providerFailure
	consecutiveFailures     int
	lastSuccess             time.Time
	lastFailure             time.Time
	cooldownUntil           time.Time
	lastModelRefresh        time.Time
	lastModelRefreshFailure time.Time
	lastModelRefreshError   string
}

type providerPool struct {
	mu               sync.Mutex
	providers        []provider
	modelProviders   map[string][]int
	candidateOrder   func([]int) []int
	cooldownFailures int
	cooldown         time.Duration
}

type providerFailure struct {
	at         time.Time
	endpoint   string
	statusCode int
	err        string
}

type providerModelRefreshTarget struct {
	index     int
	id        string
	modelsURL string
	apiKey    string
}

type modelListResponse struct {
	Object string          `json:"object"`
	Data   []modelListItem `json:"data"`
}

type providersStatusResponse struct {
	Object string           `json:"object"`
	Data   []providerStatus `json:"data"`
}

type providerStatus struct {
	ID                  string                     `json:"id"`
	Models              []string                   `json:"models"`
	BusyCount           int                        `json:"busy_count"`
	RecentFailures      []providerFailureStatus    `json:"recent_failures"`
	RecentFailureCount  int                        `json:"recent_failure_count"`
	ConsecutiveFailures int                        `json:"consecutive_failures"`
	LastSuccessAt       *time.Time                 `json:"last_success_at"`
	LastFailureAt       *time.Time                 `json:"last_failure_at"`
	Cooldown            providerCooldownStatus     `json:"cooldown"`
	ModelRefresh        providerModelRefreshStatus `json:"model_refresh"`
}

type providerFailureStatus struct {
	At         time.Time `json:"at"`
	Endpoint   string    `json:"endpoint"`
	StatusCode int       `json:"status_code,omitempty"`
	Error      string    `json:"error,omitempty"`
}

type providerCooldownStatus struct {
	Active          bool       `json:"active"`
	Until           *time.Time `json:"until"`
	RemainingMillis int64      `json:"remaining_millis"`
}

type providerModelRefreshStatus struct {
	LastSuccessAt *time.Time `json:"last_success_at"`
	LastFailureAt *time.Time `json:"last_failure_at"`
	LastError     string     `json:"last_error,omitempty"`
}

type modelListItem struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func newProviderPool(ctx context.Context, cfgs []providerConfig, client *http.Client, logger *zap.Logger) (*providerPool, error) {
	if len(cfgs) == 0 {
		return nil, errors.New("at least one upstream provider is required")
	}
	if logger == nil {
		logger = zap.NewNop()
	}

	providers := make([]provider, 0, len(cfgs))
	modelProviders := map[string][]int{}
	for i, cfg := range cfgs {
		chatURL, err := normalizeChatCompletionsURL(cfg.ProviderURL)
		if err != nil {
			return nil, fmt.Errorf("invalid provider %q URL %q: %w", cfg.ID, cfg.ProviderURL, err)
		}
		responsesURL, err := normalizeResponsesURL(cfg.ProviderURL)
		if err != nil {
			return nil, fmt.Errorf("invalid provider %q responses URL %q: %w", cfg.ID, cfg.ProviderURL, err)
		}
		modelsURL, err := normalizeModelsURL(cfg.ProviderURL)
		if err != nil {
			return nil, fmt.Errorf("invalid provider %q models URL %q: %w", cfg.ID, cfg.ProviderURL, err)
		}
		now := time.Now()
		models, err := fetchProviderModels(ctx, client, modelsURL, cfg.APIKey)
		if err != nil {
			models = map[string]struct{}{}
		}
		upstreamProvider := provider{
			id:               cfg.ID,
			chatURL:          chatURL,
			responsesURL:     responsesURL,
			modelsURL:        modelsURL,
			apiKey:           cfg.APIKey,
			models:           models,
			lastModelRefresh: now,
		}
		if err != nil {
			upstreamProvider.lastModelRefresh = time.Time{}
			upstreamProvider.lastModelRefreshFailure = now
			upstreamProvider.lastModelRefreshError = err.Error()
			upstreamProvider.lastFailure = now
			upstreamProvider.recentFailures = []providerFailure{{
				at:       now,
				endpoint: "models",
				err:      err.Error(),
			}}
			logger.Warn("failed to load provider models; provider will be skipped until refresh succeeds",
				zap.Int("provider_index", i),
				zap.String("provider_id", cfg.ID),
				zap.String("models_url", modelsURL),
				zap.Error(err),
			)
		}
		providers = append(providers, upstreamProvider)
		for model := range models {
			modelProviders[model] = append(modelProviders[model], i)
		}
		modelIDs := sortedModelIDs(models)
		logger.Info("loaded provider models",
			zap.Int("provider_index", i),
			zap.String("provider_id", cfg.ID),
			zap.String("models_url", modelsURL),
			zap.Int("models", len(models)),
			zap.Strings("model_ids", modelIDs),
		)
	}

	return &providerPool{
		providers:      providers,
		modelProviders: modelProviders,
		candidateOrder: randomCandidateOrder,
	}, nil
}

func (p *providerPool) providerCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.providers)
}

func (p *providerPool) setCooldownPolicy(failures int, cooldown time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if failures < 0 {
		failures = 0
	}
	if cooldown < 0 {
		cooldown = 0
	}
	p.cooldownFailures = failures
	p.cooldown = cooldown
}

func (p *providerPool) modelCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.modelProviders)
}

func (p *providerPool) provider(index int) provider {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.providers[index]
}

func (p *providerPool) candidatesForModel(model string) ([]int, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return nil, errors.New("request must include a model")
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	candidates := append([]int(nil), p.modelProviders[model]...)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("model %q is not available on any configured provider", model)
	}
	return candidates, nil
}

func (p *providerPool) allModels() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	models := make([]string, 0, len(p.modelProviders))
	for model := range p.modelProviders {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}

func (p *providerPool) providerModelMap() map[string][]string {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make(map[string][]string, len(p.providers))
	for _, provider := range p.providers {
		out[provider.id] = sortedModelIDs(provider.models)
	}
	return out
}

func (p *providerPool) providerStatuses(now time.Time) []providerStatus {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.clearExpiredCooldownsLocked(now)
	out := make([]providerStatus, 0, len(p.providers))
	for _, provider := range p.providers {
		failures := make([]providerFailureStatus, 0, len(provider.recentFailures))
		for _, failure := range provider.recentFailures {
			failures = append(failures, providerFailureStatus{
				At:         failure.at.UTC(),
				Endpoint:   failure.endpoint,
				StatusCode: failure.statusCode,
				Error:      failure.err,
			})
		}

		var cooldownUntil *time.Time
		var remainingMillis int64
		cooldownActive := !provider.cooldownUntil.IsZero() && now.Before(provider.cooldownUntil)
		if cooldownActive {
			cooldownUntil = timePtr(provider.cooldownUntil)
			remainingMillis = provider.cooldownUntil.Sub(now).Milliseconds()
		}

		out = append(out, providerStatus{
			ID:                  provider.id,
			Models:              sortedModelIDs(provider.models),
			BusyCount:           provider.busy,
			RecentFailures:      failures,
			RecentFailureCount:  len(failures),
			ConsecutiveFailures: provider.consecutiveFailures,
			LastSuccessAt:       timePtr(provider.lastSuccess),
			LastFailureAt:       timePtr(provider.lastFailure),
			Cooldown: providerCooldownStatus{
				Active:          cooldownActive,
				Until:           cooldownUntil,
				RemainingMillis: remainingMillis,
			},
			ModelRefresh: providerModelRefreshStatus{
				LastSuccessAt: timePtr(provider.lastModelRefresh),
				LastFailureAt: timePtr(provider.lastModelRefreshFailure),
				LastError:     provider.lastModelRefreshError,
			},
		})
	}
	return out
}

func timePtr(t time.Time) *time.Time {
	if t.IsZero() {
		return nil
	}
	utc := t.UTC()
	return &utc
}

func (p *providerPool) refreshModels(ctx context.Context, client *http.Client, logger *zap.Logger, timeout time.Duration) {
	if client == nil {
		return
	}
	if logger == nil {
		logger = zap.NewNop()
	}
	targets := p.modelRefreshTargets()
	for _, target := range targets {
		if ctx.Err() != nil {
			return
		}
		fetchCtx := ctx
		cancel := func() {}
		if timeout > 0 {
			fetchCtx, cancel = context.WithTimeout(ctx, timeout)
		}
		models, err := fetchProviderModels(fetchCtx, client, target.modelsURL, target.apiKey)
		cancel()
		now := time.Now()
		if err != nil {
			oldModels, changed, ok := p.markProviderModelsUnavailable(target.index, err, now)
			logger.Warn("failed to refresh provider models",
				zap.Int("provider_index", target.index),
				zap.String("provider_id", target.id),
				zap.String("models_url", target.modelsURL),
				zap.Bool("models_cleared", ok && changed),
				zap.Strings("previous_model_ids", oldModels),
				zap.Error(err),
			)
			continue
		}

		oldModels, newModels, changed, ok := p.replaceProviderModels(target.index, models, now)
		if !ok {
			continue
		}
		fields := []zap.Field{
			zap.Int("provider_index", target.index),
			zap.String("provider_id", target.id),
			zap.String("models_url", target.modelsURL),
			zap.Int("models", len(newModels)),
			zap.Strings("model_ids", newModels),
		}
		if changed {
			fields = append(fields, zap.Strings("previous_model_ids", oldModels))
			logger.Info("refreshed provider models", fields...)
		} else {
			logger.Debug("provider models unchanged", fields...)
		}
	}
}

func (p *providerPool) markProviderModelsUnavailable(index int, err error, failedAt time.Time) ([]string, bool, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if index < 0 || index >= len(p.providers) {
		return nil, false, false
	}
	message := ""
	if err != nil {
		message = err.Error()
	}
	oldModels := sortedModelIDs(p.providers[index].models)
	provider := &p.providers[index]
	provider.models = map[string]struct{}{}
	provider.lastFailure = failedAt
	provider.lastModelRefreshFailure = failedAt
	provider.lastModelRefreshError = message
	provider.recentFailures = append(provider.recentFailures, providerFailure{
		at:       failedAt,
		endpoint: "models",
		err:      message,
	})
	if len(provider.recentFailures) > maxRecentProviderFailures {
		copy(provider.recentFailures, provider.recentFailures[len(provider.recentFailures)-maxRecentProviderFailures:])
		provider.recentFailures = provider.recentFailures[:maxRecentProviderFailures]
	}
	p.rebuildModelProvidersLocked()
	return oldModels, len(oldModels) > 0, true
}

func (p *providerPool) modelRefreshTargets() []providerModelRefreshTarget {
	p.mu.Lock()
	defer p.mu.Unlock()
	targets := make([]providerModelRefreshTarget, 0, len(p.providers))
	for i, provider := range p.providers {
		targets = append(targets, providerModelRefreshTarget{
			index:     i,
			id:        provider.id,
			modelsURL: provider.modelsURL,
			apiKey:    provider.apiKey,
		})
	}
	return targets
}

func (p *providerPool) replaceProviderModels(index int, models map[string]struct{}, refreshedAt time.Time) ([]string, []string, bool, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if index < 0 || index >= len(p.providers) {
		return nil, nil, false, false
	}
	oldModels := sortedModelIDs(p.providers[index].models)
	newModels := sortedModelIDs(models)
	p.providers[index].models = copyModelSet(models)
	p.providers[index].lastModelRefresh = refreshedAt
	p.providers[index].lastModelRefreshFailure = time.Time{}
	p.providers[index].lastModelRefreshError = ""
	p.rebuildModelProvidersLocked()
	return oldModels, newModels, !equalStringSlices(oldModels, newModels), true
}

func (p *providerPool) rebuildModelProvidersLocked() {
	modelProviders := map[string][]int{}
	for i, provider := range p.providers {
		for model := range provider.models {
			modelProviders[model] = append(modelProviders[model], i)
		}
	}
	p.modelProviders = modelProviders
}

func copyModelSet(models map[string]struct{}) map[string]struct{} {
	out := make(map[string]struct{}, len(models))
	for model := range models {
		out[model] = struct{}{}
	}
	return out
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (p *providerPool) recordProviderSuccess(index int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if index < 0 || index >= len(p.providers) {
		return
	}
	provider := &p.providers[index]
	provider.lastSuccess = time.Now()
	provider.consecutiveFailures = 0
	provider.cooldownUntil = time.Time{}
}

func (p *providerPool) recordProviderFailure(index int, endpoint string, statusCode int, err error) {
	p.recordProviderFailureAt(index, endpoint, statusCode, err, time.Now(), true)
}

func (p *providerPool) recordProviderFailureAt(index int, endpoint string, statusCode int, err error, at time.Time, requestFailure bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if index < 0 || index >= len(p.providers) {
		return
	}
	message := ""
	if err != nil {
		message = err.Error()
	} else if statusCode != 0 {
		message = fmt.Sprintf("upstream returned HTTP %d", statusCode)
	}
	failure := providerFailure{
		at:         at,
		endpoint:   endpoint,
		statusCode: statusCode,
		err:        message,
	}
	provider := &p.providers[index]
	provider.lastFailure = at
	if requestFailure {
		provider.consecutiveFailures++
		if p.cooldownFailures > 0 && p.cooldown > 0 && provider.consecutiveFailures >= p.cooldownFailures {
			provider.cooldownUntil = at.Add(p.cooldown)
		}
	}
	provider.recentFailures = append(provider.recentFailures, failure)
	if len(provider.recentFailures) > maxRecentProviderFailures {
		copy(provider.recentFailures, provider.recentFailures[len(provider.recentFailures)-maxRecentProviderFailures:])
		provider.recentFailures = provider.recentFailures[:maxRecentProviderFailures]
	}
	if endpoint == "models" {
		provider.lastModelRefreshFailure = at
		provider.lastModelRefreshError = message
	}
}

func sortedModelIDs(models map[string]struct{}) []string {
	modelIDs := make([]string, 0, len(models))
	for model := range models {
		modelIDs = append(modelIDs, model)
	}
	sort.Strings(modelIDs)
	return modelIDs
}

func (p *providerPool) acquire(orderedCandidates []int, tried map[int]struct{}) (int, bool, func(), error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(orderedCandidates) == 0 {
		return 0, false, nil, errors.New("no provider candidates")
	}
	now := time.Now()
	p.clearExpiredCooldownsLocked(now)

	for _, providerIndex := range orderedCandidates {
		if _, ok := tried[providerIndex]; ok {
			continue
		}
		if p.providerCoolingDownLocked(providerIndex, now) {
			continue
		}
		if p.providers[providerIndex].busy == 0 {
			p.providers[providerIndex].busy++
			return providerIndex, false, p.releaseFunc(providerIndex), nil
		}
	}

	for _, providerIndex := range orderedCandidates {
		if _, ok := tried[providerIndex]; ok {
			continue
		}
		if p.providerCoolingDownLocked(providerIndex, now) {
			continue
		}
		p.providers[providerIndex].busy++
		return providerIndex, true, p.releaseFunc(providerIndex), nil
	}

	return 0, false, nil, errors.New("no untried provider candidates outside cooldown")
}

func (p *providerPool) clearExpiredCooldownsLocked(now time.Time) {
	for i := range p.providers {
		if !p.providers[i].cooldownUntil.IsZero() && !now.Before(p.providers[i].cooldownUntil) {
			p.providers[i].cooldownUntil = time.Time{}
		}
	}
}

func (p *providerPool) providerCoolingDownLocked(index int, now time.Time) bool {
	if index < 0 || index >= len(p.providers) {
		return true
	}
	cooldownUntil := p.providers[index].cooldownUntil
	return !cooldownUntil.IsZero() && now.Before(cooldownUntil)
}

func (p *providerPool) orderedCandidates(candidates []int) []int {
	if p.candidateOrder == nil {
		return append([]int(nil), candidates...)
	}
	return p.candidateOrder(candidates)
}

func randomCandidateOrder(candidates []int) []int {
	ordered := append([]int(nil), candidates...)
	if len(ordered) > 1 {
		rand.Shuffle(len(ordered), func(i, j int) {
			ordered[i], ordered[j] = ordered[j], ordered[i]
		})
	}
	return ordered
}

func (p *providerPool) releaseFunc(index int) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			p.mu.Lock()
			defer p.mu.Unlock()
			if index >= 0 && index < len(p.providers) && p.providers[index].busy > 0 {
				p.providers[index].busy--
			}
		})
	}
}

func requestedModel(body []byte) (string, error) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return "", fmt.Errorf("invalid JSON request body: %w", err)
	}
	model := strings.TrimSpace(req.Model)
	if model == "" {
		return "", errors.New("chat completions request must include a model")
	}
	return model, nil
}

type modelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

func fetchProviderModels(ctx context.Context, client *http.Client, modelsURL, apiKey string) (map[string]struct{}, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", authorizationHeader(apiKey))

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer closeBody(zap.NewNop(), resp.Body)

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxRequestBodyBytes))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("models endpoint returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var payload modelsResponse
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	models := map[string]struct{}{}
	for _, model := range payload.Data {
		id := strings.TrimSpace(model.ID)
		if id != "" {
			models[id] = struct{}{}
		}
	}
	return models, nil
}

func (p *proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimRight(r.URL.Path, "/")
	if path == "" {
		path = "/"
	}
	switch path {
	case "/healthz":
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	case "/models", "/v1/models":
		p.handleModels(w, r)
	case "/models/map", "/v1/models/map":
		p.handleModelMap(w, r)
	case "/providers", "/v1/providers", "/providers/status", "/v1/providers/status":
		p.handleProviderStatus(w, r)
	case "/chat/completions", "/v1/chat/completions":
		p.handleAPIRequest(w, r, endpointChatCompletions)
	case "/responses", "/v1/responses":
		p.handleAPIRequest(w, r, endpointResponses)
	default:
		http.NotFound(w, r)
	}
}

func (p *proxy) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !p.authorizeDownstream(w, r, "models") {
		return
	}

	models := p.pool.allModels()
	data := make([]modelListItem, 0, len(models))
	now := time.Now().Unix()
	for _, model := range models {
		data = append(data, modelListItem{
			ID:      model,
			Object:  "model",
			Created: now,
			OwnedBy: "load-balancer",
		})
	}
	writeJSON(w, modelListResponse{
		Object: "list",
		Data:   data,
	})
}

func (p *proxy) handleModelMap(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !p.authorizeDownstream(w, r, "models map") {
		return
	}
	writeJSON(w, p.pool.providerModelMap())
}

func (p *proxy) handleProviderStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !p.authorizeDownstream(w, r, "providers status") {
		return
	}
	writeJSON(w, providersStatusResponse{
		Object: "list",
		Data:   p.pool.providerStatuses(time.Now()),
	})
}

func (p *proxy) authorizeDownstream(w http.ResponseWriter, r *http.Request, api string) bool {
	if validAuthorization(r.Header.Get("Authorization"), p.apiKey) {
		return true
	}
	p.logger.Warn("unauthorized downstream request",
		zap.String("api", api),
		zap.String("path", r.URL.Path),
	)
	w.Header().Set("WWW-Authenticate", "Bearer")
	writeJSONError(w, http.StatusUnauthorized, "invalid_api_key", "invalid or missing API key")
	return false
}

type upstreamEndpoint string

const (
	endpointChatCompletions upstreamEndpoint = "chat completions"
	endpointResponses       upstreamEndpoint = "responses"
)

func (e upstreamEndpoint) url(provider provider) string {
	switch e {
	case endpointResponses:
		return provider.responsesURL
	default:
		return provider.chatURL
	}
}

func (p *proxy) handleAPIRequest(w http.ResponseWriter, r *http.Request, endpoint upstreamEndpoint) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !p.authorizeDownstream(w, r, string(endpoint)) {
		return
	}
	requestStarted := time.Now()
	body, err := readRequestBody(r, maxRequestBodyBytes)
	if err != nil {
		p.logger.Warn("failed to read downstream request",
			zap.String("api", string(endpoint)),
			zap.String("path", r.URL.Path),
			zap.Error(err),
		)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	model, err := requestedModel(body)
	if err != nil {
		p.logger.Warn("invalid downstream request",
			zap.String("api", string(endpoint)),
			zap.String("path", r.URL.Path),
			zap.Int("request_bytes", len(body)),
			zap.Error(err),
		)
		writeJSONError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	candidates, err := p.pool.candidatesForModel(model)
	if err != nil {
		p.logger.Warn("requested model is not available",
			zap.String("api", string(endpoint)),
			zap.String("path", r.URL.Path),
			zap.String("model", model),
			zap.Int("request_bytes", len(body)),
			zap.Error(err),
		)
		writeJSONError(w, http.StatusBadRequest, "model_not_available", err.Error())
		return
	}
	requestFields := []zap.Field{
		zap.String("api", string(endpoint)),
		zap.String("path", r.URL.Path),
		zap.String("model", model),
		zap.Int("request_bytes", len(body)),
		zap.Int("candidate_providers", len(candidates)),
	}

	var lastHTTP *bufferedResponse
	var lastErr error
	attempt := 0
	attemptPasses := p.attempts
	if attemptPasses < 1 {
		attemptPasses = defaultAttempts
	}
	for pass := 0; pass < attemptPasses; pass++ {
		if pass > 0 && p.delay > 0 {
			if err := sleepWithContext(r.Context(), p.delay); err != nil {
				p.logger.Info("downstream canceled request during retry delay",
					appendRequestFields(requestFields,
						zap.Int("attempt_pass", pass+1),
						zap.Duration("elapsed", time.Since(requestStarted)),
						zap.Error(err),
					)...,
				)
				return
			}
		}
		orderedCandidates := p.pool.orderedCandidates(candidates)
		tried := make(map[int]struct{}, len(orderedCandidates))
		for passAttempt := 0; passAttempt < len(orderedCandidates); passAttempt++ {
			providerIndex, busyFallback, release, err := p.pool.acquire(orderedCandidates, tried)
			if err != nil {
				lastErr = err
				break
			}
			tried[providerIndex] = struct{}{}
			attempt++

			attemptStarted := time.Now()
			resp, err := p.postUpstream(r, body, providerIndex, endpoint)
			if err != nil {
				if r.Context().Err() != nil {
					p.logger.Info("downstream canceled request",
						appendRequestFields(requestFields,
							zap.Int("attempt", attempt),
							zap.Int("attempt_pass", pass+1),
							zap.Int("provider_index", providerIndex),
							zap.Bool("busy_fallback", busyFallback),
							zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
							zap.Duration("elapsed", time.Since(requestStarted)),
							zap.Error(err),
						)...,
					)
					release()
					return
				}
				provider := p.pool.provider(providerIndex)
				lastErr = err
				p.pool.recordProviderFailure(providerIndex, string(endpoint), 0, err)
				release()
				p.logger.Warn("upstream request failed",
					appendRequestFields(requestFields,
						zap.Int("attempt", attempt),
						zap.Int("attempt_pass", pass+1),
						zap.Int("provider_index", providerIndex),
						zap.String("provider_id", provider.id),
						zap.Bool("busy_fallback", busyFallback),
						zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
						zap.Error(err),
					)...,
				)
				continue
			}

			provider := p.pool.provider(providerIndex)
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				defer release()
				defer closeBody(p.logger, resp.Body)
				copyResponseHeaders(w.Header(), resp.Header)
				w.WriteHeader(resp.StatusCode)
				responseBytes, err := copyResponseBody(w, resp.Body)
				if err != nil {
					if r.Context().Err() == nil {
						p.pool.recordProviderFailure(providerIndex, string(endpoint), resp.StatusCode, err)
					}
					p.logger.Warn("failed to copy upstream response",
						appendRequestFields(requestFields,
							zap.Int("attempt", attempt),
							zap.Int("attempt_pass", pass+1),
							zap.Int("provider_index", providerIndex),
							zap.String("provider_id", provider.id),
							zap.Bool("busy_fallback", busyFallback),
							zap.Int("status", resp.StatusCode),
							zap.Int64("response_bytes", responseBytes),
							zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
							zap.Duration("elapsed", time.Since(requestStarted)),
							zap.Error(err),
						)...,
					)
					return
				}
				p.pool.recordProviderSuccess(providerIndex)
				p.logger.Info("upstream request succeeded",
					appendRequestFields(requestFields,
						zap.Int("attempt", attempt),
						zap.Int("attempt_pass", pass+1),
						zap.Int("provider_index", providerIndex),
						zap.String("provider_id", provider.id),
						zap.Bool("busy_fallback", busyFallback),
						zap.Int("status", resp.StatusCode),
						zap.Int64("response_bytes", responseBytes),
						zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
						zap.Duration("elapsed", time.Since(requestStarted)),
					)...,
				)
				return
			}

			failure, readErr := bufferResponse(resp)
			if readErr != nil {
				p.pool.recordProviderFailure(providerIndex, string(endpoint), resp.StatusCode, readErr)
			} else {
				p.pool.recordProviderFailure(providerIndex, string(endpoint), resp.StatusCode, nil)
			}
			release()
			if readErr != nil {
				lastErr = readErr
				p.logger.Warn("failed to read upstream error response",
					appendRequestFields(requestFields,
						zap.Int("attempt", attempt),
						zap.Int("attempt_pass", pass+1),
						zap.Int("provider_index", providerIndex),
						zap.String("provider_id", provider.id),
						zap.Bool("busy_fallback", busyFallback),
						zap.Int("status", resp.StatusCode),
						zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
						zap.Error(readErr),
					)...,
				)
			} else {
				lastHTTP = failure
				lastErr = nil
				p.logger.Warn("upstream returned failure",
					appendRequestFields(requestFields,
						zap.Int("attempt", attempt),
						zap.Int("attempt_pass", pass+1),
						zap.Int("provider_index", providerIndex),
						zap.String("provider_id", provider.id),
						zap.Bool("busy_fallback", busyFallback),
						zap.Int("status", resp.StatusCode),
						zap.Int("response_bytes", len(failure.body)),
						zap.Duration("attempt_elapsed", time.Since(attemptStarted)),
					)...,
				)
			}
		}
	}

	if lastHTTP != nil {
		p.logger.Warn("returning last upstream failure",
			appendRequestFields(requestFields,
				zap.Int("attempts", attempt),
				zap.Int("status", lastHTTP.statusCode),
				zap.Int("response_bytes", len(lastHTTP.body)),
				zap.Duration("elapsed", time.Since(requestStarted)),
			)...,
		)
		copyResponseHeaders(w.Header(), lastHTTP.header)
		w.WriteHeader(lastHTTP.statusCode)
		_, _ = w.Write(lastHTTP.body)
		return
	}
	msg := "all matching upstream providers failed"
	if lastErr != nil {
		msg = lastErr.Error()
	}
	p.logger.Warn("all matching upstream providers failed",
		appendRequestFields(requestFields,
			zap.Int("attempts", attempt),
			zap.Duration("elapsed", time.Since(requestStarted)),
			zap.Error(lastErr),
		)...,
	)
	writeJSONError(w, http.StatusBadGateway, "upstream_request_error", msg)
}

func appendRequestFields(base []zap.Field, fields ...zap.Field) []zap.Field {
	out := make([]zap.Field, 0, len(base)+len(fields))
	out = append(out, base...)
	out = append(out, fields...)
	return out
}

func sleepWithContext(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *proxy) postUpstream(inbound *http.Request, body []byte, providerIndex int, endpoint upstreamEndpoint) (*http.Response, error) {
	provider := p.pool.provider(providerIndex)
	targetURL := urlWithRawQuery(endpoint.url(provider), inbound.URL.RawQuery)
	req, err := http.NewRequestWithContext(inbound.Context(), http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	copyForwardHeaders(req.Header, inbound.Header)
	req.Header.Set("Authorization", authorizationHeader(provider.apiKey))
	return p.client.Do(req)
}

type bufferedResponse struct {
	statusCode int
	header     http.Header
	body       []byte
}

func bufferResponse(resp *http.Response) (*bufferedResponse, error) {
	defer closeBody(zap.NewNop(), resp.Body)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return &bufferedResponse{
		statusCode: resp.StatusCode,
		header:     resp.Header.Clone(),
		body:       body,
	}, nil
}

func readRequestBody(r *http.Request, limit int64) ([]byte, error) {
	defer closeBody(zap.NewNop(), r.Body)
	limited := &io.LimitedReader{R: r.Body, N: limit + 1}
	body, err := io.ReadAll(limited)
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, fmt.Errorf("request body exceeds %d bytes", limit)
	}
	return body, nil
}

func writeJSONError(w http.ResponseWriter, status int, code string, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = fmt.Fprintf(w, `{"error":{"type":"server_error","code":%q,"message":%q}}`, code, message)
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func normalizeChatCompletionsURL(raw string) (string, error) {
	return normalizeProviderEndpointURL(raw, "/chat/completions")
}

func normalizeResponsesURL(raw string) (string, error) {
	return normalizeProviderEndpointURL(raw, "/responses")
}

func normalizeProviderEndpointURL(raw string, endpointPath string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("provider URL must include scheme and host: %s", raw)
	}
	path := strings.TrimRight(u.Path, "/")
	if basePath, ok := stripKnownProviderEndpoint(path); ok {
		u.Path = joinEndpointPath(basePath, endpointPath)
		return u.String(), nil
	}
	switch {
	case path == "":
		u.Path = "/v1" + endpointPath
	case strings.HasSuffix(path, "/v1"):
		u.Path = path + endpointPath
	default:
		u.Path = path + endpointPath
	}
	return u.String(), nil
}

func normalizeModelsURL(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("provider URL must include scheme and host: %s", raw)
	}
	path := strings.TrimRight(u.Path, "/")
	if basePath, ok := stripKnownProviderEndpoint(path); ok {
		u.Path = joinEndpointPath(basePath, "/models")
		u.RawQuery = ""
		return u.String(), nil
	}
	switch {
	case path == "":
		u.Path = "/v1/models"
	case strings.HasSuffix(path, "/v1"):
		u.Path = path + "/models"
	default:
		u.Path = path + "/models"
	}
	u.RawQuery = ""
	return u.String(), nil
}

func stripKnownProviderEndpoint(path string) (string, bool) {
	for _, suffix := range []string{"/chat/completions", "/responses", "/models"} {
		if strings.HasSuffix(path, suffix) {
			return strings.TrimRight(strings.TrimSuffix(path, suffix), "/"), true
		}
	}
	return "", false
}

func joinEndpointPath(basePath string, endpointPath string) string {
	if basePath == "" {
		return endpointPath
	}
	return basePath + endpointPath
}

func urlWithRawQuery(rawURL, rawQuery string) string {
	if rawQuery == "" {
		return rawURL
	}
	u, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}
	if u.RawQuery == "" {
		u.RawQuery = rawQuery
	} else {
		u.RawQuery += "&" + rawQuery
	}
	return u.String()
}

func authorizationHeader(apiKey string) string {
	apiKey = strings.TrimSpace(apiKey)
	if strings.Contains(apiKey, " ") {
		return apiKey
	}
	return "Bearer " + apiKey
}

func validAuthorization(got string, apiKey string) bool {
	want := authorizationHeader(apiKey)
	return subtle.ConstantTimeCompare([]byte(strings.TrimSpace(got)), []byte(want)) == 1
}

func copyForwardHeaders(dst, src http.Header) {
	for k, values := range src {
		lower := strings.ToLower(k)
		if isHopByHopHeader(lower) || lower == "authorization" || lower == "content-length" {
			continue
		}
		for _, v := range values {
			dst.Add(k, v)
		}
	}
}

func copyResponseHeaders(dst, src http.Header) {
	for k, values := range src {
		if isHopByHopHeader(strings.ToLower(k)) {
			continue
		}
		for _, v := range values {
			dst.Add(k, v)
		}
	}
}

func isHopByHopHeader(lower string) bool {
	switch lower {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
		"te", "trailer", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}

func copyResponseBody(w http.ResponseWriter, body io.Reader) (int64, error) {
	if flusher, ok := w.(http.Flusher); ok {
		buf := make([]byte, 32<<10)
		var copied int64
		for {
			n, readErr := body.Read(buf)
			if n > 0 {
				written, writeErr := w.Write(buf[:n])
				copied += int64(written)
				if writeErr != nil {
					return copied, writeErr
				}
				flusher.Flush()
			}
			if readErr != nil {
				if errors.Is(readErr, io.EOF) {
					return copied, nil
				}
				return copied, readErr
			}
		}
	}
	return io.Copy(w, body)
}

func closeBody(logger *zap.Logger, body io.Closer) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		logger.Warn("failed to close body", zap.Error(err))
	}
}

func newLogger() (*zap.Logger, error) {
	cfg := zap.NewProductionConfig()
	cfg.DisableStacktrace = true
	return cfg.Build()
}

func exitWithError(logger *zap.Logger, msg string, fields ...zap.Field) {
	logger.Error(msg, fields...)
	_ = logger.Sync()
	os.Exit(1)
}
