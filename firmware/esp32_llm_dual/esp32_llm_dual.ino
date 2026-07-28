// PLE TinyLM inference on the ESP32-S3 (Dual-language: English + Chinese).
// Language selectable at boot — switch between EN/ZH prompt menus.
// Only the model partition needs reflashing when changing language.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../common/llm.h"
#include "vocab.h"         // English: VOCAB_N, VOCAB_BLOB, VOCAB_OFF
#include "../zh/vocab_zh.h" // Chinese: VOCAB_N_ZH, VOCAB_BLOB_ZH, VOCAB_OFF_ZH

// Alias English macros so both use the same naming convention
#define VOCAB_N_EN VOCAB_N
#define VOCAB_BLOB_EN VOCAB_BLOB
#define VOCAB_OFF_EN VOCAB_OFF

#define USE_DISPLAY 1
#if USE_DISPLAY
#define DISPLAY_KIND DISPLAY_CORES3
#include "display.h"
#endif

#define N_GENERATE 200

// ---- language selection ---------------------------------------------------
typedef enum { LANG_EN, LANG_ZH } Lang;
static Lang lang = LANG_EN;
static int vn;
static const unsigned char *vb;
static const int *vo;

static void select_lang(Lang l) {
  lang = l;
  if (l == LANG_EN)  { vn = VOCAB_N_EN;  vb = VOCAB_BLOB_EN;  vo = VOCAB_OFF_EN;  }
  else               { vn = VOCAB_N_ZH;  vb = VOCAB_BLOB_ZH;  vo = VOCAB_OFF_ZH;  }
}

// ---- prompts (language-specific) ------------------------------------------
#define MAX_PROMPTS 10
typedef struct { const char *text; int len; const int *ids; } Prompt;
static const Prompt PROMPTS_EN[] = {
  { "Once upon a time",          4, (int[]){433, 447, 259, 405} },
  { "There was a little",        4, (int[]){2614, 282, 259, 403} },
  { "A long time ago",           4, (int[]){33, 796, 405, 5725} },
  { "In a faraway land",         4, (int[]){2322, 259, 5961, 1675} },
  { "Timmy was a",               3, (int[]){1318, 282, 259} },
  { "The best day of",           4, (int[]){382, 833, 358, 348} },
  { "A wise old owl",            4, (int[]){33, 1968, 704, 1920} },
  { "Deep in the forest",        4, (int[]){17117, 317, 263, 1078} },
  { "Lily and her dog",          4, (int[]){342, 265, 309, 635} },
  { "Once there lived a",        4, (int[]){433, 406, 907, 259} },
};
static const Prompt PROMPTS_ZH[] = {
  { "从前有一只小兔子",  2, (int[]){312, 4141} },
  { "从前有只小猫",      3, (int[]){312, 13998, 1846} },
  { "莉莉和本是朋友",    1, (int[]){14651} },
  { "有一天",            1, (int[]){324} },
  { "在一个美丽的森林里", 2, (int[]){22887, 1106} },
  { "小熊和小兔",        3, (int[]){4559, 2923, 717} },
  { "小鸭子找妈妈",      2, (int[]){4371, 17048} },
  { "春天来了",          2, (int[]){7457, 1614} },
  { "小红帽和大灰狼",    5, (int[]){24006, 8583, 4441, 2168, 2496} },
  { "月亮上的兔子",      3, (int[]){3690, 1598, 720} },
};
#define N_PROMPTS 10

static const char *lang_menu_title() {
  return lang == LANG_EN ? "===== PLE TinyLM =====" : "===== PLE TinyLM (中文) =====";
}
static const char *lang_prompt_sel() {
  return lang == LANG_EN ? "Select a prompt (0-9) or:" : "选择故事开头 (0-9):";
}
static const char *lang_prompt_cust() {
  return lang == LANG_EN ? "  c: Enter custom token IDs" : "  c: 输入自定义 token ID";
}
static const char *lang_prompt_tip1() {
  return lang == LANG_EN ? "Tip: use Python to tokenize text:" : "提示: 用 Python 转换文字:";
}
static const char *lang_prompt_tip2() {
  return lang == LANG_EN ? "  >>> from tokenizers import Tokenizer" : "  >>> from tokenizers import Tokenizer";
}
static const char *lang_prompt_tip3() {
  return lang == LANG_EN ? "  >>> tok.encode('your text').ids" : "  >>> tok.encode('你的文字').ids";
}
static const char *lang_invalid() {
  return lang == LANG_EN ? "(no valid IDs)" : "(无效)";
}
static const char *lang_prompt_custom() {
  return lang == LANG_EN ? "Enter token IDs (space/comma separated):" : "输入 token ID (空格分隔):";
}
static const char *lang_choose() {
  return lang == LANG_EN ? "  l: Switch language" : "  l: 切换语言";
}
static const char *lang_temp() {
  return lang == LANG_EN ? "  t: Randomness" : "  t: 随机度";
}

// ---- helpers --------------------------------------------------------------
static void emit(int tok) {
  if (tok >= vn) return;
  const unsigned char *bytes = vb + vo[tok];
  int len = vo[tok + 1] - vo[tok];
  if ((int)Serial.availableForWrite() >= len) Serial.write(bytes, len);
#if USE_DISPLAY
  display_puts(bytes, len);
#endif
}

Model model;
Scratch s;

// ---- int8 output head -----------------------------------------------------
static int8_t *head_w8 = NULL;
static float  *head_scale8 = NULL;
static int head_rows, head_cols;
static int8_t head_actq[128];
static float  head_acts;

static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

static void head_rows_range(float *y, int r0, int r1) {
  for (int r = r0; r < r1; r++)
    y[r] = (float)dot_i8(head_actq, head_w8 + (size_t)r * head_cols, head_cols)
           * head_scale8[r] * head_acts;
}

static TaskHandle_t head_worker;
static TaskHandle_t inference_task;
static float *volatile head_job_y;
static volatile int head_job_split;

static void head_worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    head_rows_range(head_job_y, 0, head_job_split);
    xTaskNotifyGive(inference_task);
  }
}

static void head_matvec_int8(const QT *t, const float *x, float *y) {
  (void)t;
  quantize_act(x, head_cols, head_actq, &head_acts);
  head_job_y = y;
  head_job_split = head_rows / 2;
  xTaskNotifyGive(head_worker);
  head_rows_range(y, head_job_split, head_rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (!p) { Serial.printf("PSRAM alloc failed (%u bytes)\n", (unsigned)n); while (1) delay(1000); }
  return p;
}

static void stage_head_int8(QT *t) {
  head_rows = t->rows; head_cols = t->cols;
  head_w8 = (int8_t *)ps((size_t)head_rows * head_cols);
  head_scale8 = (float *)ps((size_t)head_rows * sizeof(float));
  for (int r = 0; r < head_rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    int8_t *dst = head_w8 + (size_t)r * head_cols;
    for (int j = 0; j < head_cols; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      dst[j] = (int8_t)(code - 8);
    }
    head_scale8[r] = half2float(t->scales[(size_t)r * t->n_groups]);
  }
  Serial.printf("head staged int8: %.2f MB\n",
                ((size_t)head_rows * head_cols + (size_t)head_rows * 4) / 1e6);
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

// ---- temperature sampling -------------------------------------------------
static float temperature = 0.8f;

static int sample(float *logits, int n) {
  if (temperature < 0.01f) {
    // Greedy (temp≈0)
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < n; v++)
      if (logits[v] > bv) { bv = logits[v]; best = v; }
    return best;
  }
  // Numerical stability: subtract max
  float max_l = -1e30f;
  for (int v = 0; v < n; v++) if (logits[v] > max_l) max_l = logits[v];
  // Pass 1: total probability
  float total = 0;
  for (int v = 0; v < n; v++)
    total += expf((logits[v] - max_l) / temperature);
  // Random pick
  float r = (float)esp_random() / (float)UINT32_MAX * total;
  // Pass 2: accumulate
  float cum = 0;
  for (int v = 0; v < n; v++) {
    cum += expf((logits[v] - max_l) / temperature);
    if (cum >= r) return v;
  }
  return n - 1;
}

// ---- generate -------------------------------------------------------------
static void generate(const int *prompt_ids, int n_prompt) {
  int pos = 0, tok = 0;
  llm_profile_reset(&s);

  Serial.print(">>> ");
  for (int i = 0; i < n_prompt; i++) {
    tok = prompt_ids[i];
    emit(tok);
    llm_forward(&model, tok, pos++, &s);
  }

  int64_t t_start = esp_timer_get_time();
  int64_t decode_us = 0;
  int decoded = 0;
  for (int step = 0; step < N_GENERATE && pos < model.c.seq_len; step++) {
    tok = sample(s.logits, vn);
    emit(tok);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 7) == 0) delay(0);
  }
  int64_t total_us = esp_timer_get_time() - t_start;

  Serial.printf("\n\n--- %d tokens in %.2f s  (temp=%.1f) ---\n", decoded, total_us / 1e6, temperature);
  Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
  }
#if USE_DISPLAY
  display_stats(decoded * 1e6f / decode_us, decode_us / 1000.0f / decoded);
#endif
  blink(0);
}

// ---- interactive menu -----------------------------------------------------
static void flush_input() { while (Serial.read() >= 0) delay(1); }

static int read_token_ids(int *ids, int max_ids) {
  char buf[512]; int pos = 0;
  unsigned long start = millis();
  while (millis() - start < 10000) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == '\n' || c == '\r') break;
      if (pos < (int)sizeof(buf) - 1) buf[pos++] = c;
    }
  }
  buf[pos] = 0;
  if (pos == 0) return 0;
  int n = 0; char *p = buf;
  while (*p && n < max_ids) {
    while (*p == ' ' || *p == ',' || *p == '\t') p++;
    if (!*p) break;
    char *end; long v = strtol(p, &end, 10);
    if (end == p) break;
    if (v >= 0 && v < vn) ids[n++] = (int)v;
    p = end;
  }
  return n;
}

static void interactive_menu() {
  for (;;) {
    Serial.println();
    Serial.println(lang_menu_title());
    Serial.println(lang_prompt_sel());
    const Prompt *prompts = (lang == LANG_EN) ? PROMPTS_EN : PROMPTS_ZH;
    for (int i = 0; i < N_PROMPTS; i++)
      Serial.printf("  %d: %s\n", i, prompts[i].text);
    Serial.println(lang_prompt_cust());
    Serial.println(lang_temp());
    Serial.println(lang_choose());
    Serial.printf(lang == LANG_EN ? "  temp=%.1f > " : "  随机度=%.1f > ", temperature);

    while (!Serial.available()) delay(50);
    char cmd = Serial.read();

    if (cmd >= '0' && cmd <= '9') {
      int idx = cmd - '0';
      if (idx < N_PROMPTS) {
        flush_input(); Serial.println();
        generate(prompts[idx].ids, prompts[idx].len);
      }
    } else if (cmd == 'c' || cmd == 'C') {
      flush_input();
      Serial.println();
      Serial.println(lang_prompt_custom());
      Serial.println(lang_prompt_tip1());
      Serial.println(lang_prompt_tip2());
      Serial.println(lang_prompt_tip3());
      Serial.print("> ");
      int custom_ids[256];
      int n = read_token_ids(custom_ids, 256);
      if (n > 0) { Serial.println(); generate(custom_ids, n); }
      else { Serial.println(lang_invalid()); }
    } else if (cmd == 'l' || cmd == 'L') {
      if (lang == LANG_EN) {
        select_lang(LANG_ZH);
        Serial.println("语言: 中文");
      } else {
        select_lang(LANG_EN);
        Serial.println("Language: English");
      }
    } else if (cmd == 't' || cmd == 'T') {
      float temps[] = {0.1f, 0.5f, 0.8f, 1.0f, 1.2f, 1.5f};
      static int ti = 2; // start at 0.8
      ti = (ti + 1) % (sizeof(temps) / sizeof(temps[0]));
      temperature = temps[ti];
      Serial.println();
      Serial.printf(lang == LANG_EN ? "Temperature: %.1f\n" : "随机度: %.1f\n", temperature);
    }
    delay(100);
  }
}

// ---- setup ----------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1500);

  select_lang(LANG_EN);
  Serial.println("\n=== ESP32-S3 PLE TinyLM (EN/ZH) ===");
  Serial.println("Press 'l' at menu to switch language.");

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("mmap failed: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { Serial.println("bad model magic"); return; }
  Cfg *c = &model.c;
  Serial.printf("model: V=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim,
                part->size / 1e6);

#if USE_DISPLAY
  display_begin();
#endif

  model.tok_emb.rows = VOCAB_N_EN > VOCAB_N_ZH ? VOCAB_N_EN : VOCAB_N_ZH;
  stage_head_int8(&model.tok_emb);
  inference_task = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(head_worker_main, "head", 4096, NULL, 2,
                             &head_worker, 0) != pdPASS) {
    Serial.println("head worker creation failed");
    return;
  }
  model.head_matvec = head_matvec_int8;

  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn;
  int V = c->vocab, S = c->seq_len;
  s.x = (float *)ps(D * 4);
  s.h = (float *)ps((F > D ? F : D) * 4);
  s.qkv = (float *)ps(3 * D * 4);
  s.att = (float *)ps(D * 4);
  s.g1 = (float *)ps(F * 4);
  s.g2 = (float *)ps((P > F ? P : F) * 4);
  s.ple = (float *)ps(L * P * 4);
  s.tmpP = (float *)ps(L * P * 4);
  s.trow = (float *)ps(L * P * 4);
  s.logits = (float *)ps(V * 4);
  s.scores = (float *)ps(S * 4);
  s.kcache = (float *)ps((size_t)L * S * D * 4);
  s.vcache = (float *)ps((size_t)L * S * D * 4);
  Serial.printf("PSRAM free after alloc: %u KB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);

  interactive_menu();
}

void loop() {}
