package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type aliceReq struct {
	Request struct {
		Command string `json:"command"`
	} `json:"request"`
	Session struct {
		SessionID   string `json:"session_id"`
		MessageID   int    `json:"message_id"`
		UserID      string `json:"user_id"`
		Application struct {
			ApplicationID string `json:"application_id"`
		} `json:"application"`
	} `json:"session"`
}

type respSession struct {
	SessionID string `json:"session_id"`
	MessageID int    `json:"message_id"`
	UserID    string `json:"user_id"`
}

type aliceResp struct {
	Response struct {
		Text       string `json:"text"`
		EndSession bool   `json:"end_session"`
	} `json:"response"`
	Session respSession `json:"session"`
	Version string      `json:"version"`
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

var (
	baseURL   = strings.TrimRight(getenv("LLM_BASE_URL", "http://localhost:11434/v1"), "/")
	apiKey    = os.Getenv("LLM_API_KEY")
	model     = getenv("LLM_MODEL", "qwen3:1.7b")
	system    = getenv("LLM_SYSTEM", "Отвечай по-русски, максимум в двух предложениях, без markdown и списков.")
	reasoning = os.Getenv("LLM_REASONING_EFFORT")
	maxTokens = getenvInt("LLM_MAX_TOKENS", 200)
	timeout   = time.Duration(getenvInt("LLM_TIMEOUT_MS", 2700)) * time.Millisecond
	historyN  = getenvInt("HISTORY_TURNS", 5)
	listen    = getenv("LISTEN_ADDR", ":8080")
	path      = getenv("WEBHOOK_PATH", "/alice")

	client = &http.Client{Transport: &http.Transport{
		MaxIdleConnsPerHost: 32,
		IdleConnTimeout:     90 * time.Second,
	}}
	mdCleaner = strings.NewReplacer("**", "", "__", "", "`", "", "#", "")

	lim  = newLimiter(getenvInt("RATE_PER_MIN", 10), time.Minute)
	hist = newHistory(10 * time.Minute)
)

func getenv(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func getenvInt(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return d
}

type limiter struct {
	mu     sync.Mutex
	hits   map[string][]time.Time
	limit  int
	window time.Duration
}

func newLimiter(limit int, window time.Duration) *limiter {
	l := &limiter{hits: map[string][]time.Time{}, limit: limit, window: window}
	go func() {
		for range time.Tick(window) {
			l.mu.Lock()
			cutoff := time.Now().Add(-window)
			for k, v := range l.hits {
				if len(v) == 0 || v[len(v)-1].Before(cutoff) {
					delete(l.hits, k)
				}
			}
			l.mu.Unlock()
		}
	}()
	return l
}

func (l *limiter) allow(key string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	cutoff := now.Add(-l.window)
	recent := l.hits[key][:0]
	for _, t := range l.hits[key] {
		if t.After(cutoff) {
			recent = append(recent, t)
		}
	}
	if len(recent) >= l.limit {
		l.hits[key] = recent
		return false
	}
	l.hits[key] = append(recent, now)
	return true
}

type dialog struct {
	msgs []chatMsg
	seen time.Time
}

type history struct {
	mu   sync.Mutex
	data map[string]*dialog
}

func newHistory(ttl time.Duration) *history {
	h := &history{data: map[string]*dialog{}}
	go func() {
		for range time.Tick(time.Minute) {
			h.mu.Lock()
			for k, d := range h.data {
				if time.Since(d.seen) > ttl {
					delete(h.data, k)
				}
			}
			h.mu.Unlock()
		}
	}()
	return h
}

func (h *history) get(id string) []chatMsg {
	h.mu.Lock()
	defer h.mu.Unlock()
	if d, ok := h.data[id]; ok {
		return slices.Clone(d.msgs)
	}
	return nil
}

func (h *history) add(id string, msgs ...chatMsg) {
	h.mu.Lock()
	defer h.mu.Unlock()
	d, ok := h.data[id]
	if !ok {
		d = &dialog{}
		h.data[id] = d
	}
	d.msgs = append(d.msgs, msgs...)
	if max := historyN * 2; len(d.msgs) > max {
		d.msgs = slices.Clone(d.msgs[len(d.msgs)-max:])
	}
	d.seen = time.Now()
}

func (h *history) drop(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.data, id)
}

func askLLM(ctx context.Context, past []chatMsg, prompt string) (string, error) {
	msgs := make([]chatMsg, 0, len(past)+2)
	msgs = append(msgs, chatMsg{Role: "system", Content: system})
	msgs = append(msgs, past...)
	msgs = append(msgs, chatMsg{Role: "user", Content: prompt})

	payload := map[string]any{
		"model":      model,
		"messages":   msgs,
		"max_tokens": maxTokens,
		"stream":     false,
	}
	if reasoning != "" {
		payload["reasoning_effort"] = reasoning
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	res, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return "", fmt.Errorf("llm status %d: %s", res.StatusCode, msg)
	}

	var out struct {
		Choices []struct {
			Message chatMsg `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", errors.New("empty response")
	}
	return strings.TrimSpace(out.Choices[0].Message.Content), nil
}

func clean(s string) string {
	s = strings.TrimSpace(mdCleaner.Replace(s))
	if s == "" {
		return "Нет ответа, попробуй переформулировать."
	}
	if rs := []rune(s); len(rs) > 1000 {
		s = string(rs[:999]) + "…"
	}
	return s
}

func reply(ctx context.Context, in *aliceReq) (string, bool) {
	sid := in.Session.SessionID
	cmd := strings.TrimSpace(in.Request.Command)

	switch cmd {
	case "ping":
		return "pong", false
	case "":
		return "Слушаю. Задай вопрос.", false
	case "хватит", "выйти", "стоп":
		hist.drop(sid)
		return "Пока.", true
	case "забудь", "начни сначала", "новый диалог":
		hist.drop(sid)
		return "Начинаем заново.", false
	}

	uid := in.Session.UserID
	if uid == "" {
		uid = in.Session.Application.ApplicationID
	}
	if !lim.allow(uid) {
		return "Слишком много запросов, подожди минуту.", false
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	start := time.Now()
	ans, err := askLLM(ctx, hist.get(sid), cmd)
	if err != nil {
		log.Printf("llm error after %v: %v", time.Since(start), err)
		return "Не успела ответить, попробуй ещё раз.", false
	}
	log.Printf("llm ok in %v", time.Since(start))
	hist.add(sid, chatMsg{Role: "user", Content: cmd}, chatMsg{Role: "assistant", Content: ans})
	return clean(ans), false
}

func handler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var in aliceReq
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&in); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	out := aliceResp{
		Version: "1.0",
		Session: respSession{
			SessionID: in.Session.SessionID,
			MessageID: in.Session.MessageID,
			UserID:    in.Session.UserID,
		},
	}
	out.Response.Text, out.Response.EndSession = reply(r.Context(), &in)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(out); err != nil {
		log.Println("encode:", err)
	}
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc(path, handler)
	srv := &http.Server{
		Addr:              listen,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Printf("listening on %s%s, llm: %s model=%s", listen, path, baseURL, model)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}()

	<-ctx.Done()
	shCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shCtx); err != nil {
		log.Println("shutdown:", err)
	}
}
