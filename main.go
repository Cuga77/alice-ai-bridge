package main

import (
	"bufio"
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
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	_ "time/tzdata" // embedded time zones for minimal images without system tzdata
	"unicode"
	"unicode/utf8"
)

type aliceReq struct {
	Meta struct {
		Timezone string `json:"timezone"`
	} `json:"meta"`
	Request struct {
		Command           string `json:"command"`
		OriginalUtterance string `json:"original_utterance"`
		Markup            struct {
			DangerousContext bool `json:"dangerous_context"`
		} `json:"markup"`
	} `json:"request"`
	Session struct {
		SessionID   string `json:"session_id"`
		SkillID     string `json:"skill_id"`
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
	dotenv = loadDotenv(os.Getenv("ENV_FILE"))

	baseURL   = strings.TrimRight(getenv("LLM_BASE_URL", "http://localhost:11434/v1"), "/")
	apiKey    = getenv("LLM_API_KEY", "")
	model     = getenv("LLM_MODEL", "gemma3:1b")
	system    = getenv("LLM_SYSTEM", "Отвечай по-русски, кратко и по делу. Ответ зачитает голосовой помощник, поэтому пиши так, как говорят вслух: без markdown, списков, эмодзи, ссылок и сокращений.")
	reasoning = getenv("LLM_REASONING_EFFORT", "")
	maxTokens = getenvInt("LLM_MAX_TOKENS", 300)
	timeout   = time.Duration(getenvInt("LLM_TIMEOUT_MS", 2500)) * time.Millisecond
	genTime   = time.Duration(getenvInt("LLM_GEN_TIMEOUT_MS", 60000)) * time.Millisecond
	chunkLen  = getenvInt("CHUNK_RUNES", 600)
	historyN  = getenvInt("HISTORY_TURNS", 5)
	listen    = getenv("LISTEN_ADDR", ":8080")
	path      = getenv("WEBHOOK_PATH", "/alice")
	tzName    = getenv("TIMEZONE", "")
	danger    = getenv("DANGER_REPLY", "Я не могу помочь с этим. Если тебе тяжело или кто-то в опасности, поговори с близкими или позвони в экстренные службы по номеру 112.")
	skillIDs  = strings.FieldsFunc(getenv("ALLOWED_SKILL_ID", ""), func(r rune) bool { return r == ',' || unicode.IsSpace(r) })

	client = &http.Client{Transport: &http.Transport{
		MaxIdleConnsPerHost: 32,
		IdleConnTimeout:     90 * time.Second,
	}}
	mdCleaner = strings.NewReplacer("**", "", "__", "", "`", "", "#", "")

	urlRe     = regexp.MustCompile(`https?://\S+|www\.\S+`)
	mdLinkRe  = regexp.MustCompile(`\[([^\]]+)\]\([^)]*\)`)
	spacePunc = regexp.MustCompile(`\s+([.,!?;:])`)
	abbrevFix = strings.NewReplacer(
		"т.е.", "то есть", "т. е.", "то есть",
		"т.к.", "так как", "т. к.", "так как",
		"и т.д.,", "и так далее,", "и т. д.,", "и так далее,",
		"и т.п.,", "и тому подобное,", "и т. п.,", "и тому подобное,",
		"и т.д.", "и так далее.", "и т. д.", "и так далее.",
		"и т.п.", "и тому подобное.", "и т. п.", "и тому подобное.",
		"напр.", "например",
	)

	// Background generations: cancelled and awaited on shutdown.
	genCtx, stopGen = context.WithCancel(context.Background())
	gens            sync.WaitGroup

	lim  = newLimiter(getenvInt("RATE_PER_MIN", 10), time.Minute)
	hist = newHistory(10 * time.Minute)
)

// loadDotenv reads KEY=VALUE lines from a .env file; a missing file is not an error.
func loadDotenv(name string) map[string]string {
	if name == "" {
		name = ".env"
	}
	vars := map[string]string{}
	data, err := os.ReadFile(name)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			log.Printf("env file %s: %v", name, err)
		}
		return vars
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(strings.TrimPrefix(line, "export "), "=")
		if !ok {
			continue
		}
		v = strings.TrimSpace(v)
		if len(v) >= 2 && (v[0] == '"' || v[0] == '\'') && v[len(v)-1] == v[0] {
			v = v[1 : len(v)-1]
		}
		vars[strings.TrimSpace(k)] = v
	}
	return vars
}

// getenv prefers the process environment over the .env file.
func getenv(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	if v := dotenv[k]; v != "" {
		return v
	}
	return d
}

func getenvInt(k string, d int) int {
	if v, err := strconv.Atoi(getenv(k, "")); err == nil {
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
	pend *answer
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
					if d.pend != nil {
						d.pend.cancel()
					}
					delete(h.data, k)
				}
			}
			h.mu.Unlock()
		}
	}()
	return h
}

func (h *history) dialogLocked(id string) *dialog {
	d, ok := h.data[id]
	if !ok {
		d = &dialog{}
		h.data[id] = d
	}
	d.seen = time.Now()
	return d
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
	d := h.dialogLocked(id)
	d.msgs = append(d.msgs, msgs...)
	if max := historyN * 2; len(d.msgs) > max {
		d.msgs = slices.Clone(d.msgs[len(d.msgs)-max:])
	}
}

func (h *history) pending(id string) *answer {
	h.mu.Lock()
	defer h.mu.Unlock()
	if d, ok := h.data[id]; ok {
		return d.pend
	}
	return nil
}

func (h *history) setPending(id string, a *answer) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.dialogLocked(id).pend = a
}

func (h *history) stopPending(id string) {
	h.mu.Lock()
	var a *answer
	if d, ok := h.data[id]; ok {
		a, d.pend = d.pend, nil
	}
	h.mu.Unlock()
	if a != nil {
		a.cancel()
		<-a.finished
	}
}

func (h *history) drop(id string) {
	h.stopPending(id)
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.data, id)
}

type answer struct {
	mu       sync.Mutex
	text     string
	sent     int
	done     bool
	err      error
	changed  chan struct{}
	cancel   context.CancelFunc
	finished chan struct{}
}

func (a *answer) notifyLocked() {
	close(a.changed)
	a.changed = make(chan struct{})
}

func (a *answer) append(s string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.text += s
	a.notifyLocked()
}

func (a *answer) finish(err error) string {
	a.mu.Lock()
	defer a.mu.Unlock()
	// Drop an unfinished sentence left by the token limit or a timeout.
	if i := strings.LastIndexAny(a.text, ".!?…\n"); i >= a.sent {
		_, n := utf8.DecodeRuneInString(a.text[i:])
		if strings.TrimSpace(a.text[i+n:]) != "" {
			a.text = a.text[:i+n]
		}
	}
	a.done, a.err = true, err
	a.notifyLocked()
	return a.text
}

func (a *answer) next(ctx context.Context, budget time.Duration) (chunk string, more bool, err error) {
	timer := time.NewTimer(budget)
	defer timer.Stop()
	for {
		a.mu.Lock()
		if a.done || utf8.RuneCountInString(a.text[a.sent:]) >= chunkLen {
			break
		}
		ch := a.changed
		a.mu.Unlock()
		select {
		case <-ch:
			continue
		case <-timer.C:
		case <-ctx.Done():
		}
		a.mu.Lock()
		break
	}
	defer a.mu.Unlock()
	rest := abbrevFix.Replace(a.text[a.sent:])
	a.text = a.text[:a.sent] + rest
	n := cut(rest, a.done)
	a.sent += n
	return rest[:n], !a.done || a.sent < len(a.text), a.err
}

func cut(s string, done bool) int {
	limit := len(s)
	if i := runeOffset(s, chunkLen); i >= 0 {
		limit = i
	} else if done {
		return len(s)
	}
	best, prev := -1, rune(0)
	for i, r := range s[:limit] {
		if r == '\n' && i > 0 {
			best = i
		} else if strings.ContainsRune(".!?…", r) && !(r == '.' && unicode.IsDigit(prev)) {
			j := i + utf8.RuneLen(r)
			if next, _ := utf8.DecodeRuneInString(s[j:]); unicode.IsSpace(next) || (j == len(s) && done) {
				best = j
			}
		}
		prev = r
	}
	if best > 0 {
		return best
	}
	if i := strings.LastIndexFunc(s[:limit], unicode.IsSpace); i > 0 {
		return i
	}
	return 0
}

func runeOffset(s string, n int) int {
	for i := range s {
		if n == 0 {
			return i
		}
		n--
	}
	return -1
}

func startAnswer(sid, sys, prompt string) *answer {
	past := hist.get(sid)
	ctx, cancel := context.WithTimeout(genCtx, genTime)
	a := &answer{changed: make(chan struct{}), cancel: cancel, finished: make(chan struct{})}
	gens.Add(1)
	go func() {
		defer gens.Done()
		defer close(a.finished)
		defer cancel()
		start := time.Now()
		err := streamLLM(ctx, past, sys, prompt, a.append)
		full := strings.TrimSpace(a.finish(err))
		switch {
		case errors.Is(err, context.Canceled):
			log.Printf("%s llm cancelled after %v (%d chars)", short(sid), time.Since(start).Round(time.Millisecond), len([]rune(full)))
		case err != nil:
			log.Printf("%s llm error after %v (%d chars): %v", short(sid), time.Since(start).Round(time.Millisecond), len([]rune(full)), err)
		default:
			log.Printf("%s llm done in %v (%d chars)", short(sid), time.Since(start).Round(time.Millisecond), len([]rune(full)))
		}
		if full != "" {
			hist.add(sid, chatMsg{Role: "user", Content: prompt}, chatMsg{Role: "assistant", Content: full})
		}
	}()
	return a
}

func streamLLM(ctx context.Context, past []chatMsg, sys, prompt string, onDelta func(string)) error {
	msgs := make([]chatMsg, 0, len(past)+2)
	msgs = append(msgs, chatMsg{Role: "system", Content: sys})
	msgs = append(msgs, past...)
	msgs = append(msgs, chatMsg{Role: "user", Content: prompt})

	payload := map[string]any{
		"model":      model,
		"messages":   msgs,
		"max_tokens": maxTokens,
		"stream":     true,
	}
	if reasoning != "" {
		payload["reasoning_effort"] = reasoning
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	res, err := client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("llm status %d: %s", res.StatusCode, msg)
	}

	sc := bufio.NewScanner(res.Body)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			return nil
		}
		var ev struct {
			Choices []struct {
				Delta chatMsg `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return err
		}
		if len(ev.Choices) > 0 && ev.Choices[0].Delta.Content != "" {
			onDelta(ev.Choices[0].Delta.Content)
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return ctx.Err()
}

var (
	weekdays = [...]string{"воскресенье", "понедельник", "вторник", "среда", "четверг", "пятница", "суббота"}
	months   = [...]string{"января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"}
)

// systemPrompt adds the current date and time: the model has no clock of its own.
func systemPrompt(tz string) string {
	if tzName != "" {
		tz = tzName
	}
	loc, err := time.LoadLocation(tz)
	if err != nil || tz == "" {
		loc = time.UTC
	}
	now := time.Now().In(loc)
	return fmt.Sprintf("%s\nСейчас %s, %d %s %d года, %02d:%02d (часовой пояс %s).",
		system, weekdays[now.Weekday()], now.Day(), months[now.Month()-1], now.Year(), now.Hour(), now.Minute(), loc)
}

// isEmoji reports pictographs and emoji modifiers; ordinary symbols like ° and № are kept.
func isEmoji(r rune) bool {
	return r >= 0x1F000 || (r >= 0x2600 && r <= 0x27BF) || (r >= 0x2300 && r <= 0x23FF) ||
		r == 0x2B50 || r == 0x2B55 || r == 0xFE0F || r == 0x200D
}

// clean turns model output into plain speech: no markdown, links or emoji; list items become sentences.
func clean(s string) string {
	s = mdLinkRe.ReplaceAllString(s, "$1")
	s = urlRe.ReplaceAllString(s, "")
	s = strings.Map(func(r rune) rune {
		if isEmoji(r) {
			return -1
		}
		return r
	}, s)
	var parts []string
	for _, line := range strings.Split(mdCleaner.Replace(s), "\n") {
		line = strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(line), "*-•·"))
		if line == "" {
			continue
		}
		if r, _ := utf8.DecodeLastRuneInString(line); !strings.ContainsRune(".!?…:;,", r) {
			line += "."
		}
		parts = append(parts, line)
	}
	return spacePunc.ReplaceAllString(strings.Join(strings.Fields(strings.Join(parts, " ")), " "), "$1")
}

func short(sid string) string {
	if len(sid) > 8 {
		return sid[:8]
	}
	return sid
}

func speak(ctx context.Context, sid, kind string, a *answer) string {
	start := time.Now()
	chunk, more, err := a.next(ctx, timeout)
	chunk = clean(chunk)
	state := "last"
	switch {
	case chunk == "" && more:
		state = "still thinking"
	case chunk == "":
		state = "empty"
	case more:
		state = "more"
	}
	log.Printf("%s %s: %d chars in %v (%s)", short(sid), kind, utf8.RuneCountInString(chunk), time.Since(start).Round(time.Millisecond), state)
	switch {
	case chunk != "" && more:
		return chunk + " Продолжить?"
	case chunk != "":
		hist.setPending(sid, nil)
		return chunk
	case more:
		return "Ещё думаю. Скажи «дальше»."
	case err != nil:
		hist.setPending(sid, nil)
		return "Не получилось ответить, попробуй ещё раз."
	default:
		hist.setPending(sid, nil)
		return "Это всё."
	}
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
	case "дальше", "продолжи", "продолжай", "да", "ещё", "еще", "давай":
		if a := hist.pending(sid); a != nil {
			return speak(ctx, sid, "continue", a), false
		}
		if cmd != "да" && cmd != "давай" {
			return "Продолжать нечего. Задай вопрос.", false
		}
	}

	// command is lowercased and stripped of punctuation; prefer the original phrase
	// unless it also contains the activation part ("запусти навык …").
	prompt := cmd
	if orig := strings.TrimSpace(in.Request.OriginalUtterance); orig != "" && normalize(orig) == normalize(cmd) {
		prompt = orig
	}

	uid := in.Session.UserID
	if uid == "" {
		uid = in.Session.Application.ApplicationID
	}
	if !lim.allow(uid) {
		return "Слишком много запросов, подожди минуту.", false
	}

	hist.stopPending(sid)
	if in.Request.Markup.DangerousContext {
		log.Printf("%s dangerous_context: fixed reply", short(sid))
		return danger, false
	}
	a := startAnswer(sid, systemPrompt(in.Meta.Timezone), prompt)
	hist.setPending(sid, a)
	return speak(ctx, sid, "answer", a), false
}

func normalize(s string) string {
	s = strings.ReplaceAll(strings.ToLower(s), "ё", "е")
	return strings.Join(strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}), " ")
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
	if len(skillIDs) > 0 && !slices.Contains(skillIDs, in.Session.SkillID) {
		log.Printf("rejected request from skill %q", in.Session.SkillID)
		http.Error(w, "forbidden", http.StatusForbidden)
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
		if len(skillIDs) == 0 {
			log.Print("warning: ALLOWED_SKILL_ID is not set, requests from any skill are accepted")
		}
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
	stopGen()
	done := make(chan struct{})
	go func() { gens.Wait(); close(done) }()
	select {
	case <-done:
	case <-shCtx.Done():
		log.Println("shutdown: background generations did not finish in time")
	}
	log.Println("stopped")
}
