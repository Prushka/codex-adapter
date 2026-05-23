package main

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type responseSSEWriter struct {
	w     http.ResponseWriter
	flush http.Flusher
	debug *DebugRecorder
}

func newResponseSSEWriter(w http.ResponseWriter, debug *DebugRecorder) *responseSSEWriter {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	return &responseSSEWriter{w: w, flush: flusher, debug: debug}
}

func (s *responseSSEWriter) Event(kind string, payload map[string]any) error {
	if payload == nil {
		payload = map[string]any{}
	}
	payload["type"] = kind
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	s.debug.SaveJSON("outbound responses event", payload)
	if _, err := fmt.Fprintf(s.w, "event: %s\n", kind); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", data); err != nil {
		return err
	}
	if s.flush != nil {
		s.flush.Flush()
	}
	return nil
}
