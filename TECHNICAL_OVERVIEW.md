# How the NBA Jam detector works

This document explains the technology behind the detector: the computer-vision
ideas it uses, how it "learns" without a neural network, how noisy readings turn
into reliable events, and the tools that build and check all of it. It assumes no
computer-vision background.

- [1. The problem](#1-the-problem)
- [2. Computer-vision basics used here](#2-computer-vision-basics-used-here)
- [3. Reading the overlay](#3-reading-the-overlay)
- [4. Template matching](#4-template-matching)
- [5. Where the templates come from](#5-where-the-templates-come-from)
- [6. Recognising whole screens](#6-recognising-whole-screens)
- [7. Voting and the state machine](#7-voting-and-the-state-machine)
- [8. Checking that it works](#8-checking-that-it-works)
- [9. File map](#9-file-map)
- [10. Known limits and when to rebuild](#10-known-limits-and-when-to-rebuild)

---

## 1. The problem

An NBA Jam arcade cabinet (a Raspberry Pi running EmulationStation/RetroArch with
a CRT shader) sends video over HDMI into a USB capture card. The program has to
answer, from pixels alone:

- What is the game doing? (team select, tipoff, a quarter, halftime, overtime, game over)
- What is the score?
- Did someone just score, and who?

It has no access to the game's memory or network. The only input is a stream of
1920x1080 frames, and the only output is a handful of facts for Home Assistant.

Two things make this harder than it sounds:

1. **The score is only on screen for about 3 seconds** around each scoring play
   and each quarter change. The rest of the time there is nothing to read.
2. **The CRT shader damages the picture.** It slices every glyph into horizontal
   scanline stripes, so the text looks nothing like clean font rendering.

## 2. Computer-vision basics used here

**An image is an array of numbers.** A 1080p colour frame is 1080 rows x 1920
columns x 3 channels (blue, green, red), each 0-255. Everything below is
arithmetic on those arrays. OpenCV and NumPy do the work.

**Cropping.** The game's layout never moves, so the score always appears at the
same pixel coordinates. Cutting out a fixed rectangle (a "region") is just array
slicing. The regions live in `config.yaml` and `show_regions.py` draws them on a
frame so you can check them.

**Thresholding.** Turn "how bright is this pixel" into yes/no. For example,
"every channel above 165" means "white". The result is a *mask*: an array of 0s
and 255s where 255 means "this pixel is part of a glyph".

**Colour spaces.** BGR is how cameras store colour, but it's awkward for "is this
purple?". HSV (hue, saturation, value) separates the colour itself (hue) from how
vivid and how bright it is, so "purple" becomes a simple range check on hue.

**Morphology.** Operations that reshape masks. *Closing* fills small gaps: slide
a small shape over the mask and bridge pixels that are close together. The code
uses a tall, thin shape (1 wide x 7 tall) so it bridges the *horizontal*
scanline gaps in glyphs without merging neighbouring digits sideways.

**Connected components.** Group touching mask pixels into separate blobs and
report each blob's size and bounding box. This is how specks of noise get
discarded (small blobs are dropped) and how a single digit gets isolated.

**Correlation.** A single number from -1 to 1 saying how alike two
equal-sized images are (section 4). This is the core of the recognition.

## 3. Reading the overlay

When something happens in the game, a purple band appears along the bottom of the
screen showing the clock, the quarter banner ("2ND QUARTER", "HALFTIME", ...) and
both scores. Reading it is a pipeline:

```
frame ──► purple gate ──► text mask ──► cut out glyphs ──► match against templates
          "is the         white+red       one per digit,      "which digit / phrase
           band there?"   pixels, gaps    banner as one       looks most like this?"
                          bridged         piece
```

1. **Purple gate** (`BaseOCRDetector.has_purple_band`). Look at one small,
   plain stretch of the band (right of the clock) and check what fraction is
   purple in HSV. If not enough, the overlay is off and no further work is done
   for this frame. This keeps the average cost per frame very low, since the
   overlay is absent most of the time.
2. **Text mask** (`BaseOCRDetector.text_mask`). Keep pixels that are white
   (`min(B,G,R) > 165`) or red (the digit colour right after a score), apply the
   vertical close to stitch the scanline stripes back together, then drop small
   blobs.
3. **Digit slots** (`segment_glyphs`). Score digits are right-aligned and sit in
   fixed ~52 px slots (ones, tens, hundreds). Digits often touch, so connected
   components alone can't split them. Instead each slot is examined separately:
   the largest tall blob in the slot is that digit, and an empty slot ends the
   number.
4. **Banner text.** The quarter region is masked the same way but kept as one
   piece and matched as a whole phrase, not letter by letter.

## 4. Template matching

### The idea

Instead of a general text reader (OCR), the program keeps a library of example
images of each thing it needs to recognise: many examples of "0", of "1", ...
and of each banner phrase. To read a new glyph it asks: *which stored example
does this look most like?* This is called template matching with
nearest-neighbour classification.

It suits this problem because the font is fixed, the layout is fixed, and the
shader distorts every glyph in the same way every time. A general OCR engine is
built for variety this game doesn't have, and it is slower. (Tesseract was the
original approach; it is now used only for bootstrapping labels, section 5.)

### Making glyphs comparable (`_normalise`)

Before comparing, each glyph is:

1. cropped to the bounding box of its lit pixels, which removes position
   differences;
2. resized to a fixed size (28x44 for digits, 240x36 for phrases), which removes
   size differences;
3. scaled to 0..1.

### Scoring similarity (`_corr`)

The similarity measure is the **Pearson correlation coefficient**. Treat both
images as long lists of numbers. Subtract each list's average, then measure how
much the two lists rise and fall together:

```
a' = a - mean(a)          b' = b - mean(b)
corr = sum(a' * b') / sqrt( sum(a'^2) * sum(b'^2) )
```

- `1.0` means identical shape.
- `0` means unrelated.
- Negative means inverted.

Subtracting the mean makes the score insensitive to overall brightness, and
dividing by the spread makes it insensitive to contrast. Only the *pattern* of
light and dark matters. That's why the same "7" at slightly different brightness
still matches.

### Choosing the answer

For one glyph, compute the correlation against every stored template of every
digit, and take the best:

```
glyph ──► vs every "0" template ──► best 0.62
      ──► vs every "1" template ──► best 0.31
      ──► vs every "7" template ──► best 0.94   ◄── winner: "7", confidence 0.94
      ...
```

Then thresholds decide whether the winner is believable:

| Threshold | Value | Meaning |
|---|---|---|
| `MIN_DIGIT_SCORE` | 0.5 | Below this the blob isn't a digit; the whole score read is discarded |
| `MIN_HUNDREDS_SCORE` | 0.85 | The hundreds slot must match very well, because stray background blobs sometimes land there and 3-digit scores are rare |
| `MIN_PHRASE_SCORE` | 0.5 | Below this, no banner is reported |

A read is "digits joined together", for example glyphs 4 and 2 give 42. Banners
work the same way, with the phrase list as the candidates. Whole-phrase matching
is more robust than reading letters one at a time: a smudge on one letter barely
changes the correlation of a 240-pixel-wide phrase image.

## 5. Where the templates come from

There is no neural network and no training loop in the machine-learning sense.
"Training" here means **collecting labeled example glyphs and storing them**. The
templates file (`templates/nbajam.npz`) is simply a compressed bundle of those
normalised images, with names like `digit_7` or `phrase_HALFTIME`.

The part that would normally mean hand-labeling thousands of frames is handled
by bootstrapping:

```
record_frames.py        record a game as lossless PNGs
        │
        ▼
prefill_labels.py       guess every frame's overlay/scores/banner using the
        │               Tesseract-based reader, write a draft labels CSV
        ▼
   review & correct     fix the draft where it's wrong (this is the only
        │               human-judgment step, and only for disagreements)
        ▼
build_templates.py      for each labeled frame: cut out the glyphs, check the
        │               glyph COUNT matches the label's digit count, and store
        ▼               each glyph under its labeled digit
templates/nbajam.npz
```

Details worth knowing:

- **The draft is allowed to be imperfect.** The Tesseract reader is wrong often
  enough that it can't run the system, but it is right often enough that a
  corrected draft is cheap to produce. A wrong label would corrupt the library,
  so see the next point.
- **Self-checking build.** `build_templates.py` skips any frame where the number
  of segmented glyphs doesn't equal the number of digits in the label. A bad
  segmentation therefore can't poison the library. It prints how many reads were
  skipped.
- **More examples, not smarter maths.** Templates from several games and
  situations are all kept. More examples of "3" under different lighting and
  scanline phases means a better chance one is close. `--per-class` can thin the
  library evenly if it grows too big.
- **The labels CSV format** is `frame,overlay,p1,p2,quarter`. `overlay` is 1 when
  the purple band is up; the rest are the true values.
- **Game 3's labels** were derived from the tracker's own output. Because each
  overlay appearance shows exactly one score, the tracker's confirmed value can
  label every frame of that appearance. That is convenient, but it makes those
  labels less independent as a test (see section 8).

Scene templates (section 6) are built the same spirit: pick one clean frame of
each screen from the recordings (`build_scene_templates.py`, using the
downscaled frame cache that `cache_frames.py` writes), crop the title, store it.
There is no labeling beyond "this frame at this second shows that screen".

## 6. Recognising whole screens

Non-gameplay screens (team select, match-up, "GET READY FOR 2ND QUARTER", final
stats, game over) each have a title or graphic that never moves. `SceneDetector`
(`detectors/scene_detector.py`):

1. shrinks the frame to 480x270 grayscale (fast: about 2 ms);
2. cuts the fixed title crop for each known scene;
3. correlates it with the stored reference, the same Pearson measure as before;
4. takes the best match if it is at least `0.85`.

This works well because the numbers separate cleanly: real matches score about
1.0, and the best false match seen was 0.78. A threshold of 0.85 sits in the
gap.

Some rules need no template at all:

- `black`: whole-frame average brightness is nearly zero (the transition into a game).
- `win`: more than 18% of the upper-middle is magenta ("X WINS!").
- `live`: the lower part of the frame is bright enough to be gameplay.

Two details make it reliable:

- **Look-alikes are included as scenes.** `substitutions` and `half_stats`
  resemble the stats screens. Giving them their own templates stops them from
  being misread as something else, and `half_stats` is useful in its own right as
  a halftime signal.
- **Blinking titles get two templates.** Team select alternates between "PRESS
  SHOOT TO SELECT" and "PRESS TURBO TO SWAP", so both are stored.

## 7. Voting and the state machine

Even with a good matcher, individual frame reads are noisy: a frame caught
mid-animation, a digit the shader smeared, a banner misread. The tracker
(`game_tracker.py`) is what turns noisy per-frame guesses into trustworthy
events. It contains only logic, with no camera, clock or network, so a recorded
game can be replayed through it and it can be unit-tested.

### Voting

The tracker never trusts one frame. Within one *overlay appearance* (from the
band appearing until it has been gone 4 frames), it tallies reads:

```
frame 1: player1 = 42     votes: {42: 1}
frame 2: player1 = 42     votes: {42: 2}  ← 2 votes and a plurality → confirm 42
frame 3: player1 = 12     votes: {12: 1}  ← ignored, 42 already confirmed
```

A value is accepted only when it has:

- **at least 2 votes** (`SCORE_CONFIRM_VOTES`), and
- **a strict plurality**: more votes than any other value in this appearance.

Each player votes independently. Banners are voted the same way.

Why 2 and not more? Votes are counted in frames, so latency is the time for two
frames to arrive: at the 0.15 s frame interval, about 0.3 s. More votes would be
safer but slower. This setting was chosen by measuring replay accuracy against
latency (section 8).

### Plausibility filtering

Before a read is even allowed to vote, it must be believable:

- A score can **never go down**, and can rise **by at most 6** between readings
  (`MAX_PLAUSIBLE_JUMP`). The most any single basket plus a quick follow-up can
  add is small, so a read of "42 → 87" is a misread and is thrown away without
  voting.
- A banner can only move **forward** through the game: 1ST < 2ND < HALFTIME <
  3RD < 4TH < OVERTIME < DOUBLE < TRIPLE < FINAL (`BANNER_RANK`). A lower-ranked
  banner is treated as a misread. ("END OF nTH QUARTER" shares the rank of its
  quarter.)
- After a FINAL, only a new "1ST QUARTER" is accepted.

These rules use game knowledge to remove most errors before voting starts. They
are cheap, but they are also the reason the tracker needs the restart rule below.

### Recovering from a restart

"Forward only" is dangerous if the game is reset without the tracker seeing a
team-select screen: every real read now looks "backward" and would be discarded
forever. The tracker restarts itself only when **two independent signals agree**:

1. a lower-ranked banner is read at least 6 times in one appearance and is the
   plurality, **and**
2. at least one player's score is read 3+ times *below* the current score.

A banner misread alone (say OVERTIME read as HALFTIME) changes nothing, because
the scores still move forward. This was learned the hard way: an earlier looser
rule wiped a real overtime game.

### Scenes drive the rest

Scenes are checked on every frame, independent of the overlay:

- `team_select` / `matchup` start a fresh game record.
- `matchup`, then `black`, then the first `live` frame = **tipoff**.
- `get_ready` (the "GET READY FOR nTH QUARTER" board), then the first `live`
  frame = the next period begins. This is keyed on the *current* status, so
  it still works if the banner was missed. This gives the start-of-quarter
  signal without waiting for the first basket.
- `half_stats` while in Q1/Q2 = halftime (a backup for a missed HALFTIME banner).
- `final_stats`, `final_score`, `game_over` and `win` = **game over**.
- Sustained black (`BLACK_TIMEOUT_FRAMES`), or no overlay for a very long time
  (`STALE_OVERLAY_FRAMES`) mid-game = signal lost or game abandoned; reset.

### Events

The tracker emits `status` (a state change), `banner`, `scores` (the current
numbers) and `score` (one event per scoring play, with scorer, points and
period). A basket is the difference between the old and new confirmed score. The
first basket of a game is tagged `q1`, not `tipoff`.

The wiring is in `main.py`: it reads a frame every `frame_interval`, classifies
the scene, runs the recognizer when the gate is on, feeds the tracker, and hands
resulting events to `MQTTClient` for Home Assistant.

## 8. Checking that it works

A recognizer can look perfect on the data it learned from. All the numbers below
come from tests that avoid that trap.

- **Held-out evaluation** (`evaluate.py`). Build templates from some games and
  test on a different one. Per-frame digit accuracy is 96-99% when training on
  the other two games. A reading is only counted as a success if it matches the
  label for that frame.
- **Replay** (`replay.py`). `extract` runs the (slow) vision step once over a
  recorded game and saves the per-frame observations to JSON. `run` then pushes
  them through the tracker in milliseconds, so tracker settings can be tuned
  repeatedly without re-reading any images. Results at the current settings: it
  caught every scoring play in all three games (34/34, 35/35, 52/52) with no
  spurious events, at a median 0.26 s after the overlay first appeared.
- **Unit tests** (`test_game_tracker.py`, run `python -m unittest
  test_game_tracker`). Synthetic scenarios for logic that real games rarely
  exercise: restart recovery, overtime, timeouts, a missed halftime banner, play
  resuming after the get-ready board.

Caveat: game 3's labels came from the tracker, so tracker accuracy measured
against them is partly circular. Games 1 and 2 are the independent
checks. A threshold or gate should never be judged against labels derived from
the thing being tested.

## 9. File map

| File | Role |
|---|---|
| `main.py` | The loop: capture, classify, read, track, publish |
| `video_capture.py` | Threaded V4L2/OpenCV capture that keeps only the newest frame |
| `detectors/base_ocr_detector.py` | Purple gate and the scanline-bridging text mask |
| `detectors/template_matcher.py` | Digit slots, normalising, correlation, `TemplateLibrary`, `TemplateRecognizer` |
| `detectors/scene_detector.py` | Full-screen scene classification |
| `game_tracker.py` | Voting, plausibility rules, state machine (pure logic) |
| `mqtt_client.py` | Home Assistant MQTT discovery and publishing |
| `config.py` / `config.yaml` | Settings and screen regions |
| `record_frames.py` | Record a game as lossless PNGs |
| `prefill_labels.py` | Draft a labels CSV from the Tesseract-based reader |
| `build_templates.py` | Build digit/phrase templates from labeled frames |
| `cache_frames.py`, `build_scene_templates.py` | Build the 480x270 cache and scene templates |
| `evaluate.py` | Held-out accuracy of a recognizer against a labels CSV |
| `replay.py` | Extract observations once, replay through the tracker quickly |
| `show_regions.py` | Draw configured regions on a frame; print the purple gate value |
| `test_game_tracker.py` | Synthetic tracker scenarios |
| `templates/nbajam.npz`, `templates/scenes.npz` | The learned templates |
| `dataset/` (gitignored) | Recorded games, labels, caches |

## 10. Known limits and when to rebuild

**Rebuild templates** (record a game, prefill and correct labels, run
`build_templates.py` and `build_scene_templates.py`) if any of these change: the
CRT shader settings, the capture resolution, or the game version. The digit slot
offsets and the scene title crops are constants at 1080p and are not scaled
automatically.

**Not yet verified:**

- 3-digit scores. The hundreds slot is extrapolated from the other slots; no
  game has reached 100 points yet.
- One-player or CPU games, and a second player joining mid-game.
- Starting the script mid-game: the period advance relies on already knowing
  the current period, so it can't follow breaks until a banner is read.
- A 4th overtime is printed by the game as "TRIPLE OVERTIME", so overtimes past
  the third can't be told apart.
- Why one live halftime was missed. It is now covered by the `half_stats` backup;
  if it recurs, run `python main.py --debug 2>&1 | tee run.log` and keep the log.
