// PLE TinyLM inference on the ESP32-S3 (Chinese version).
// Same as the original but with Chinese tokenizer + prompts.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../common/llm.h"
#include "../zh/vocab_zh.h"

#define USE_DISPLAY 1
#if USE_DISPLAY
#define DISPLAY_KIND DISPLAY_CORES3
#include "display.h"
#endif

#define N_GENERATE 200

// ---- Chinese story prompts ------------------------------------------------
static const struct { const char *text; int len; const int *ids; } PROMPTS[] = {
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
#define N_PROMPTS (sizeof(PROMPTS) / sizeof(PROMPTS[0]))

// Use VOCAB_N_ZH / VOCAB_BLOB_ZH / VOCAB_OFF_ZH via short aliases
#define VOCAB_N VOCAB_N_ZH
#define VOCAB_BLOB VOCAB_BLOB_ZH
#define VOCAB_OFF VOCAB_OFF_ZH

// ---- helpers --------------------------------------------------------------
static void emit(int tok) {
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
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
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < VOCAB_N; v++)
      if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
    tok = best;
    emit(tok);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 7) == 0) delay(0);
  }
  int64_t total_us = esp_timer_get_time() - t_start;

  Serial.printf("\n\n--- %d tokens in %.2f s ---\n", decoded, total_us / 1e6);
  Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
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
    if (v >= 0 && v < VOCAB_N) ids[n++] = (int)v;
    p = end;
  }
  return n;
}

static void interactive_menu() {
  for (;;) {
    Serial.println("\n===== PLE TinyLM (中文) =====");
    Serial.println("选择故事开头 (0-9):");
    for (int i = 0; i < (int)N_PROMPTS; i++) {
      Serial.printf("  %d: %s\n", i, PROMPTS[i].text);
    }
    Serial.println("  c: 输入自定义 token ID");
    Serial.print("> ");

    while (!Serial.available()) delay(50);
    char cmd = Serial.read();

    if (cmd >= '0' && cmd <= '9') {
      int idx = cmd - '0';
      if (idx < (int)N_PROMPTS) {
        flush_input(); Serial.println();
        generate(PROMPTS[idx].ids, PROMPTS[idx].len);
      }
    } else if (cmd == 'c' || cmd == 'C') {
      flush_input();
      Serial.println("\n输入 token ID (空格分隔):");
      Serial.print("> ");
      int custom_ids[256];
      int n = read_token_ids(custom_ids, 256);
      if (n > 0) { Serial.println(); generate(custom_ids, n); }
      else { Serial.println("(无效)"); }
    }
    delay(100);
  }
}

// ---- setup ----------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== ESP32-S3 PLE TinyLM (中文) ===");

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

  model.tok_emb.rows = VOCAB_N;
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
