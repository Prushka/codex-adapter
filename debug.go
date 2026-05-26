package adapter

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type DebugRecorder struct {
	dir string
	mu  sync.Mutex
	seq int
}

func NewDebugRecorder(dir string) (*DebugRecorder, error) {
	if dir == "" {
		dir = "debug"
	}
	return &DebugRecorder{dir: dir}, nil
}

func (r *DebugRecorder) SaveJSON(label string, value any) {
	if r == nil {
		return
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		data, _ = json.MarshalIndent(map[string]any{
			"debug_error": err.Error(),
			"value":       fmt.Sprint(value),
		}, "", "  ")
	}
	r.write(label, data)
}

func (r *DebugRecorder) SaveRawJSON(label string, raw []byte) {
	if r == nil {
		return
	}
	var value any
	if err := json.Unmarshal(raw, &value); err == nil {
		r.SaveJSON(label, value)
		return
	}
	r.SaveJSON(label, map[string]string{"raw": string(raw)})
}

func (r *DebugRecorder) write(label string, data []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.seq++
	name := fmt.Sprintf("%06d-%s.json", r.seq, sanitizeDebugLabel(label))
	path := filepath.Join(r.dir, name)
	if err := os.MkdirAll(r.dir, 0o755); err != nil {
		return
	}
	_ = os.WriteFile(path, append(data, '\n'), 0o644)
}

func sanitizeDebugLabel(label string) string {
	label = strings.ToLower(label)
	var b strings.Builder
	lastDash := false
	for _, r := range label {
		ok := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if ok {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "debug"
	}
	return out
}
