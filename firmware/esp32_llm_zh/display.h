// Optional on-device screen for the demo: the story appears on a display wired
// to the ESP32 itself, no laptop. Two panels supported via DISPLAY_KIND:
//
//   DISPLAY_OLED_I2C (default) -- 128x64 I2C mono OLED. 4 wires:
//     GND->GND, VCC->3V3, SCL->GPIO46, SDA->GPIO18. Set OLED_CONTROLLER to match
//     the panel: 1.3" is usually SH1106, 0.96" usually SSD1306. Wrong choice ->
//     top row correct, rest noise (SH1106-as-SSD1306) -> flip the define.
//   DISPLAY_TFT_SPI -- 2.0" 240x320 SPI ST7789 (GMT020-02-7P). Wiring in the TFT
//     block below. Nicer color hero shot for later.
//
// The API (display_begin / display_puts) is identical for both, so the sketch
// integration never changes -- only this header and the wiring.
#ifndef DISPLAY_H
#define DISPLAY_H

#define DISPLAY_OLED_I2C 1
#define DISPLAY_TFT_SPI  2
#define DISPLAY_CORES3   3
#ifndef DISPLAY_KIND
#define DISPLAY_KIND DISPLAY_OLED_I2C
#endif

// ==================== 128x64 I2C OLED (SH1106 or SSD1306) ===================
#if DISPLAY_KIND == DISPLAY_OLED_I2C
#include <Adafruit_GFX.h>
#include <Wire.h>

// Set to match the panel you wired: 1.3" is usually SH1106, 0.96" usually SSD1306.
// Wrong choice -> shifted/garbled image; just switch this and reflash.
#define OLED_SH1106  1
#define OLED_SSD1306 2
#ifndef OLED_CONTROLLER
#define OLED_CONTROLLER OLED_SH1106
#endif

#define OLED_SDA 18
#define OLED_SCL 46
#define OLED_ADDR 0x3C          // some panels are 0x3D
#define SCR_W 128
#define SCR_H 64
#define CW 6                    // 6x8 base glyph at text size 1
#define CH 8

#if OLED_CONTROLLER == OLED_SH1106
#include <Adafruit_SH110X.h>
static Adafruit_SH1106G oled = Adafruit_SH1106G(SCR_W, SCR_H, &Wire, -1);
#define OLED_WHITE SH110X_WHITE
#else
#include <Adafruit_SSD1306.h>
static Adafruit_SSD1306 oled = Adafruit_SSD1306(SCR_W, SCR_H, &Wire, -1);
#define OLED_WHITE SSD1306_WHITE
#endif

static int ox = 0, oy = 0;

static void display_home() {
  oled.clearDisplay();
  ox = 0; oy = 0;
}

static void display_begin() {
  Wire.begin(OLED_SDA, OLED_SCL);
  Wire.setClock(400000);   // 400kHz fast I2C -> ~4x quicker frame flush than default
#if OLED_CONTROLLER == OLED_SH1106
  oled.begin(OLED_ADDR, true);
#else
  oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
#endif
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(OLED_WHITE);
  oled.setTextWrap(false);      // we wrap at token boundaries ourselves
  oled.display();
  display_home();
}

// Append a token's bytes; whole-token wrap, clear+home when the screen fills,
// flush the framebuffer once per token so text visibly appears.
static void display_puts(const unsigned char *s, int len) {
  if (ox + len * CW > SCR_W) { oy += CH; ox = 0; }
  if (oy + CH > SCR_H) display_home();
  for (int i = 0; i < len; i++) {
    char c = (char)s[i];
    if (c == '\n') { oy += CH; ox = 0; }
    else if (c >= 32 && c < 127) {
      if (ox + CW > SCR_W) { oy += CH; ox = 0; }
      if (oy + CH > SCR_H) display_home();
      oled.setCursor(ox, oy);
      oled.write(c);
      ox += CW;
    }
    if (oy + CH > SCR_H) display_home();
  }
  oled.display();
}

// Closing stats card shown when generation finishes.
static void display_stats(float tok_s, float ms) {
  oled.clearDisplay();
  oled.setTextColor(OLED_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);  oled.print("ESP32-S3  PLE LLM");
  oled.setCursor(0, 14); oled.print("28.9M params");
  oled.setCursor(0, 24); oled.print("in 320KB of RAM");
  oled.setTextSize(2);
  oled.setCursor(0, 40); oled.print(tok_s, 1); oled.print(" tok/s");
  oled.setTextSize(1);
  oled.setCursor(0, 57); oled.print(ms, 0); oled.print(" ms/token");
  oled.display();
}

// ======================= 2.0" SPI TFT (ST7789) ==============================
#elif DISPLAY_KIND == DISPLAY_TFT_SPI
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>

#define TFT_CS 10
#define TFT_DC 7
#define TFT_RST 6
#define TFT_SCK 12
#define TFT_MOSI 11
#define TFT_W 240
#define TFT_H 320
#define TFT_TEXTSIZE 2
#define CHAR_W (6 * TFT_TEXTSIZE)
#define CHAR_H (8 * TFT_TEXTSIZE)
#define LINE_H (CHAR_H + 2)

static Adafruit_ST7789 tft = Adafruit_ST7789(&SPI, TFT_CS, TFT_DC, TFT_RST);

static void display_home() {
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(2, 2);
}

static void display_begin() {
  SPI.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);
  tft.init(TFT_W, TFT_H);
  tft.setRotation(0);
  tft.setTextSize(TFT_TEXTSIZE);
  tft.setTextColor(ST77XX_GREEN, ST77XX_BLACK);
  tft.setTextWrap(false);
  display_home();
}

static void display_puts(const unsigned char *s, int len) {
  if (tft.getCursorX() + len * CHAR_W > TFT_W)
    tft.setCursor(2, tft.getCursorY() + LINE_H);
  if (tft.getCursorY() + LINE_H > TFT_H)
    display_home();
  for (int i = 0; i < len; i++) {
    char c = (char)s[i];
    if (c == '\n') {
      tft.setCursor(2, tft.getCursorY() + LINE_H);
    } else if (c >= 32 && c < 127) {
      if (tft.getCursorX() + CHAR_W > TFT_W)
        tft.setCursor(2, tft.getCursorY() + LINE_H);
      tft.write(c);
    }
    if (tft.getCursorY() + LINE_H > TFT_H)
      display_home();
  }
}

// Closing stats card shown when generation finishes.
static void display_stats(float tok_s, float ms) {
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_GREEN, ST77XX_BLACK);
  tft.setTextSize(2);
  tft.setCursor(4, 10);  tft.print("ESP32-S3");
  tft.setCursor(4, 40);  tft.print("PLE TinyLM");
  tft.setCursor(4, 90);  tft.print("28.9M params");
  tft.setCursor(4, 120); tft.print("in 320KB RAM");
  tft.setTextSize(3);
  tft.setCursor(4, 170); tft.print(tok_s, 1); tft.print(" t/s");
  tft.setTextSize(2);
  tft.setCursor(4, 220); tft.print(ms, 0); tft.print(" ms/token");
}

// ==================== M5Stack CoreS3 (ILI9342C 320x240) ======================
#elif DISPLAY_KIND == DISPLAY_CORES3
#include <M5GFX.h>

static M5GFX lcd;
static int cursor_x = 0, cursor_y = 0;
#define SCR_W 320
#define SCR_H 240
#define CHAR_W 12    // 6x8 Font0 at textsize 2
#define CHAR_H 16
#define LINE_H (CHAR_H + 2)

static void display_home() {
  lcd.fillScreen(TFT_BLACK);
  cursor_x = 0;
  cursor_y = 0;
}

static void display_begin() {
  lcd.begin();
  lcd.setRotation(1);  // landscape: 320 wide x 240 tall
  lcd.fillScreen(TFT_BLACK);
  lcd.setTextSize(2);
  lcd.setTextColor(TFT_WHITE, TFT_BLACK);
  lcd.setTextWrap(false);
  cursor_x = 0; cursor_y = 0;
}

static void display_puts(const unsigned char *s, int len) {
  lcd.setCursor(cursor_x, cursor_y);
  for (int i = 0; i < len; i++) {
    char c = (char)s[i];
    if (c == '\n') {
      cursor_x = 0; cursor_y += LINE_H;
    } else if (c >= 32 && c < 127) {
      if (cursor_x + CHAR_W > SCR_W) { cursor_x = 0; cursor_y += LINE_H; }
      if (cursor_y + LINE_H > SCR_H) display_home();
      lcd.drawChar(c, cursor_x, cursor_y);
      cursor_x += CHAR_W;
    }
    if (cursor_y + LINE_H > SCR_H) display_home();
    lcd.setCursor(cursor_x, cursor_y);
  }
  lcd.display();
}

static void display_stats(float tok_s, float ms) {
  lcd.fillScreen(TFT_BLACK);
  lcd.setTextSize(2);
  lcd.setCursor(4, 10);  lcd.printf("ESP32-S3 PLE LLM");
  lcd.setCursor(4, 40);  lcd.printf("28.9M params");
  lcd.setCursor(4, 70);  lcd.printf("in 320KB RAM");
  lcd.setTextSize(3);
  lcd.setCursor(4, 120); lcd.printf("%.1f tok/s", tok_s);
  lcd.setTextSize(2);
  lcd.setCursor(4, 180); lcd.printf("%.0f ms/tok", ms);
  lcd.display();
}

#endif
#endif
