package langid

import "testing"

// The trigram model has to tell the Latin-script languages apart on realistic query
// and sentence text, the case the script vote cannot settle. Each case is a phrase in
// one language carrying that language's characteristic short words and affixes.
func TestDetectLatinLanguages(t *testing.T) {
	d := New()
	cases := []struct {
		text string
		want string
	}{
		{"the best search results for the query the user typed", English},
		{"el mejor motor de búsqueda para encontrar la información correcta", Spanish},
		{"le meilleur moteur de recherche pour trouver la bonne information", French},
		{"die beste Suchmaschine um die richtigen Informationen zu finden", German},
		{"il miglior motore di ricerca per trovare le informazioni giuste", Italian},
		{"o melhor motor de busca para encontrar a informação certa", Portuguese},
		{"de beste zoekmachine om de juiste informatie te vinden", Dutch},
	}
	for _, c := range cases {
		got := d.DetectResult(c.text)
		if got.Lang != c.want {
			t.Errorf("Detect(%q) = %q (confident=%v), want %q", c.text, got.Lang, got.Confident, c.want)
		}
		if got.Script != ScriptLatin {
			t.Errorf("Detect(%q) script = %q, want latin", c.text, got.Script)
		}
		if !got.Confident {
			t.Errorf("Detect(%q) not confident, want confident on a full sentence", c.text)
		}
	}
}

// The script layer must resolve the non-Latin writing systems without the trigram
// model: Han is Chinese, Han mixed with kana is Japanese, Hangul is Korean, and the
// rest route by their script. This is the path that lets a CJK query reach the
// segmenting analyzer.
func TestDetectByScript(t *testing.T) {
	d := New()
	cases := []struct {
		text       string
		wantLang   string
		wantScript Script
	}{
		{"搜索引擎返回最佳结果", Chinese, ScriptHan},
		{"検索エンジンは最高の結果を返します", Japanese, ScriptHiragana},
		{"검색 엔진은 최상의 결과를 반환합니다", Korean, ScriptHangul},
		{"поисковая система возвращает лучшие результаты", Russian, ScriptCyrillic},
		{"محرك البحث يعيد أفضل النتائج", Arabic, ScriptArabic},
		{"מנוע החיפוש מחזיר את התוצאות הטובות", Hebrew, ScriptHebrew},
		{"खोज इंजन सबसे अच्छे परिणाम देता है", Hindi, ScriptDevanagari},
		{"เครื่องมือค้นหาคืนผลลัพธ์ที่ดีที่สุด", Thai, ScriptThai},
		{"η μηχανή αναζήτησης επιστρέφει τα καλύτερα αποτελέσματα", Greek, ScriptGreek},
	}
	for _, c := range cases {
		got := d.DetectResult(c.text)
		if got.Lang != c.wantLang {
			t.Errorf("Detect(%q) = %q, want %q", c.text, got.Lang, c.wantLang)
		}
		if got.Script != c.wantScript {
			t.Errorf("Detect(%q) script = %q, want %q", c.text, got.Script, c.wantScript)
		}
		if !got.Confident {
			t.Errorf("Detect(%q) not confident, a clear single-script run should be", c.text)
		}
	}
}

// A pure-kanji sentence with no kana is Chinese; the same sentence with a kana
// particle is Japanese. This single rule separates the two CJK languages a plain Han
// count would merge, so it gets its own check.
func TestJapaneseKanaOverridesHan(t *testing.T) {
	d := New()
	if got := d.DetectResult("東京都"); got.Lang != Chinese {
		t.Errorf("pure kanji = %q, want %q", got.Lang, Chinese)
	}
	if got := d.DetectResult("東京に行く"); got.Lang != Japanese {
		t.Errorf("kanji with kana = %q, want %q", got.Lang, Japanese)
	}
}

// The empty and the symbols-only inputs carry no script signal, so Detect must report
// unknown rather than guess, the precondition for the caller's script-default fallback.
func TestDetectUnknownOnNoSignal(t *testing.T) {
	d := New()
	for _, text := range []string{"", "   ", "123 456", "!!! ??? ###"} {
		got := d.DetectResult(text)
		if got.Lang != Unknown || got.Confident {
			t.Errorf("Detect(%q) = %q confident=%v, want unknown and not confident", text, got.Lang, got.Confident)
		}
	}
}

// A one or two word Latin fragment is too little to score confidently, the spec's
// low-confidence case. Detect still returns a best guess but marks it not confident so
// the caller falls back to a script-based default rather than commit to it.
func TestDetectLowConfidenceShortLatin(t *testing.T) {
	d := New()
	got := d.DetectResult("ok")
	if got.Script != ScriptLatin {
		t.Errorf("short Latin script = %q, want latin", got.Script)
	}
	if got.Confident {
		t.Errorf("Detect(%q) confident, want not confident on a two-letter fragment", "ok")
	}
}

// A nil detector still classifies non-Latin scripts, the degraded mode for a
// deployment that ships no trigram profiles: the script vote needs no profiles. Latin
// text it cannot separate, so it reports unknown and not confident.
func TestNilDetector(t *testing.T) {
	var d *Detector
	if got := d.DetectResult("搜索引擎返回最佳结果"); got.Lang != Chinese {
		t.Errorf("nil detector on Han = %q, want %q", got.Lang, Chinese)
	}
	if got := d.DetectResult("the best search results"); got.Confident {
		t.Errorf("nil detector confident on Latin, want not confident")
	}
}

// Detection is deterministic: the same query must resolve to the same language every
// call, because the analyzer it picks must not flap between requests. Map iteration
// order is the obvious way this could break, so the ranking sorts ties.
func TestDetectDeterministic(t *testing.T) {
	d := New()
	text := "le meilleur moteur de recherche pour trouver la bonne information"
	first := d.DetectResult(text)
	for i := 0; i < 50; i++ {
		if got := d.DetectResult(text); got != first {
			t.Fatalf("Detect not deterministic: %+v vs %+v", got, first)
		}
	}
}

// A string that is exactly half Latin and half a non-Latin script is a script tie, and
// the tie must resolve the same way every call: a Chinese product page with Latin model
// codes must not flap between the script default and the trigram model, or a document
// would index under a different analyzer than the query that looks for it. This is the
// exact case the real-data routing gate caught, pinned here at the unit level. The
// non-Latin script wins the tie, the more specific route.
func TestDetectMixedScriptDeterministic(t *testing.T) {
	d := New()
	mixed := "MPVC/FRP聚強襯塑復合管B型"
	first := d.DetectResult(mixed)
	if first.Script != ScriptHan {
		t.Errorf("mixed Latin/Han script = %q, want han to win the tie", first.Script)
	}
	for i := 0; i < 200; i++ {
		if got := d.DetectResult(mixed); got != first {
			t.Fatalf("mixed-script detection not deterministic: %+v vs %+v", got, first)
		}
	}
}

func BenchmarkDetectLatin(b *testing.B) {
	d := New()
	text := "the best search results for the query the user typed today"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = d.DetectResult(text)
	}
}

func BenchmarkDetectCJK(b *testing.B) {
	d := New()
	text := "検索エンジンは最高の結果を返します"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = d.DetectResult(text)
	}
}
