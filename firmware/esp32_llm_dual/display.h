// Optional on-device screen for the demo — CoreS3 variant with Chinese font.
// The API (display_begin / display_puts) matches the other display.h variants.
#ifndef DISPLAY_H
#define DISPLAY_H

#define DISPLAY_OLED_I2C 1
#define DISPLAY_TFT_SPI  2
#define DISPLAY_CORES3   3
#ifndef DISPLAY_KIND
#define DISPLAY_KIND DISPLAY_OLED_I2C
#endif

// ==================== M5Stack CoreS3 (ILI9342C 320x240) ======================
#if DISPLAY_KIND == DISPLAY_CORES3
#include <M5GFX.h>

static M5GFX lcd;
#define SCR_W 320
#define SCR_H 240
#define CHAR_H 18
#define LINE_H 20

static void display_home() {
  lcd.fillScreen(TFT_BLACK);
}

static void display_begin() {
  lcd.begin();
  lcd.setRotation(1);  // landscape: 320 wide x 240 tall
  lcd.fillScreen(TFT_BLACK);
  lcd.setFont(&fonts::efontCN_16);
  lcd.setTextColor(TFT_WHITE, TFT_BLACK);
  lcd.setTextWrap(true);
  lcd.setCursor(0, 0);
}

static void display_puts(const unsigned char *s, int len) {
  lcd.write(s, len);
  // Check if we need to scroll
  if (lcd.getCursorY() > SCR_H - LINE_H) display_home();
  lcd.display();
}

static void display_stats(float tok_s, float ms) {
  lcd.fillScreen(TFT_BLACK);
  lcd.setFont(&fonts::efontCN_16);
  lcd.setTextColor(TFT_WHITE, TFT_BLACK);
  lcd.setCursor(0, 10);  lcd.print("ESP32-S3 PLE LLM");
  lcd.setCursor(0, 36);  lcd.print("28.9M params");
  lcd.setCursor(0, 62);  lcd.print("in 320KB RAM");
  lcd.setTextSize(2);
  lcd.setCursor(0, 110); lcd.printf("%.1f tok/s", tok_s);
  lcd.setTextSize(1);
  lcd.setCursor(0, 170); lcd.printf("%.0f ms/tok", ms);
  lcd.display();
}

// Other display modes — not used in dual firmware, stubs for compile
#else
static void display_home() {}
static void display_begin() {}
static void display_puts(const unsigned char *s, int len) { (void)s; (void)len; }
static void display_stats(float tok_s, float ms) { (void)tok_s; (void)ms; }
#endif
#endif
