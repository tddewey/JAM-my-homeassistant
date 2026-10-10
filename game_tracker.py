"""Game state machine: turns per-frame observations into confirmed status/score events.

The tracker is pure logic (no I/O, no clock): feed it one observation per frame and it
returns the events to publish. Because it only needs the observations, a recorded game
can be replayed through it (see replay.py).

Observations per frame:
  scene    label from SceneDetector.classify (or None)
  reading  (player1, player2, banner) from TemplateRecognizer.read, or None when the
           purple overlay is not on screen

Debouncing is by consecutive frames, not seconds, so it is independent of the capture
interval: at 6-7 frames/s, 2 agreeing frames confirm a score in about 0.3 s.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

NOT_PLAYING = 'not_playing'
TEAM_SELECT = 'team_select'
STARTING = 'starting'
TIPOFF = 'tipoff'
HALFTIME = 'halftime'
OVERTIME = 'overtime'
GAME_OVER = 'game_over'
Q1 = 'q1'      # a score during tipoff can only be the first quarter's

# Overlay banner text -> status. "END OF nTH QUARTER" keeps the quarter's status;
# the exact text is published separately as the banner. All overtimes share one status
# (the banner says which); the game prints TRIPLE OVERTIME for the third and later ones.
BANNER_STATUS = {
    '1ST QUARTER': 'q1', 'END OF 1ST QUARTER': 'q1',
    '2ND QUARTER': 'q2',
    'HALFTIME': HALFTIME,
    '3RD QUARTER': 'q3', 'END OF 3RD QUARTER': 'q3',
    '4TH QUARTER': 'q4', 'END OF 4TH QUARTER': 'q4',
    'OVERTIME': OVERTIME, 'DOUBLE OVERTIME': OVERTIME, 'TRIPLE OVERTIME': OVERTIME,
    'FINAL': GAME_OVER,
}

# A game only moves forward through these; a lower-ranked banner is a misread, unless it
# keeps coming back with scores that went down too, which means the game was restarted
BANNER_RANK = {
    '1ST QUARTER': 1, 'END OF 1ST QUARTER': 1, '2ND QUARTER': 2, 'HALFTIME': 3,
    '3RD QUARTER': 4, 'END OF 3RD QUARTER': 4, '4TH QUARTER': 5, 'END OF 4TH QUARTER': 5,
    'OVERTIME': 6, 'DOUBLE OVERTIME': 7, 'TRIPLE OVERTIME': 8, 'FINAL': 9,
}

# Scenes that mean the game is definitively over
GAME_OVER_SCENES = {'final_stats', 'final_score', 'game_over', 'win'}

SCORE_CONFIRM_VOTES = 2       # plausible readings of one value needed to confirm a score
BANNER_CONFIRM_VOTES = 2
SCENE_CONFIRM_FRAMES = 1      # consecutive frames; scene matches are either ~1.0 or below 0.8, so one is enough
MAX_PLAUSIBLE_JUMP = 6        # most points one player can gain between observed readings
APPEARANCE_GAP_FRAMES = 4     # overlay absent this long => the next overlay is a new appearance
RESTART_BANNER_VOTES = 6      # reads of one lower-ranked banner in an overlay appearance that suggest a restart
RESTART_SCORE_VOTES = 3       # reads of one value below the current score that corroborate it
MAX_SCORE = 199               # first reading mid-game: anything up to this is believable
BLACK_TIMEOUT_FRAMES = 40     # sustained black while a game is "on" => signal lost
STALE_OVERLAY_FRAMES = 4000   # no overlay for this long mid-game => assume the game was abandoned


@dataclass
class Event:
    """Something to publish. kind is one of: status, banner, scores, score."""
    kind: str
    status: Optional[str] = None
    banner: Optional[str] = None
    player1: Optional[int] = None
    player2: Optional[int] = None
    scorer: Optional[int] = None      # 1 or 2, for kind == 'score'
    points: Optional[int] = None      # points gained, for kind == 'score'
    quarter: Optional[str] = None     # status at the time, for kind == 'score'


@dataclass
class GameTracker:
    status: str = NOT_PLAYING
    banner: Optional[str] = None
    player1: Optional[int] = None
    player2: Optional[int] = None

    _scene: Optional[str] = None
    _scene_run: int = 0
    _saw_black: bool = False
    _black_run: int = 0
    _votes: Dict[int, Dict[int, int]] = field(default_factory=lambda: {1: {}, 2: {}})
    _banner_votes: Dict[str, int] = field(default_factory=dict)
    _back_votes: Dict[str, int] = field(default_factory=dict)       # lower-ranked banner reads this appearance
    _below_votes: Dict[int, Dict[int, int]] = field(default_factory=lambda: {1: {}, 2: {}})
    _frames_since_overlay: int = 0
    events: List[Event] = field(default_factory=list)

    # ---- public ----
    def update(self, scene: Optional[str], reading: Optional[Tuple]) -> List[Event]:
        """Process one frame; returns the events it produced (possibly empty)."""
        self.events = []
        self._update_scene(scene)
        if reading is not None:
            self._frames_since_overlay = 0
            self._update_overlay(*reading)
        else:
            self._frames_since_overlay += 1
            if self._frames_since_overlay >= APPEARANCE_GAP_FRAMES:
                self._reset_votes()
            if (self.status not in (NOT_PLAYING, TEAM_SELECT, STARTING, GAME_OVER)
                    and self._frames_since_overlay > STALE_OVERLAY_FRAMES):
                self._new_game(NOT_PLAYING)
        return self.events

    # ---- scenes ----
    def _update_scene(self, scene: Optional[str]) -> None:
        self._scene_run = self._scene_run + 1 if scene == self._scene else 1
        self._scene = scene
        need = 1 if scene == 'win' else SCENE_CONFIRM_FRAMES
        confirmed = self._scene_run >= need

        if scene == 'black':
            self._black_run += 1
            self._saw_black = True
            if self._black_run >= BLACK_TIMEOUT_FRAMES and self.status not in (NOT_PLAYING, STARTING, GAME_OVER):
                self._new_game(NOT_PLAYING)
        else:
            self._black_run = 0

        if not confirmed:
            return
        if scene == 'team_select' and self.status != TEAM_SELECT:
            self._new_game(TEAM_SELECT)
        elif scene == 'matchup' and self.status != STARTING:
            self._new_game(STARTING)
        elif scene == 'live' and self.status == STARTING and self._saw_black:
            self._set_status(TIPOFF)
            self._set_scores(0, 0)
        elif scene in GAME_OVER_SCENES and self.status != GAME_OVER and self.status != NOT_PLAYING:
            self._set_status(GAME_OVER)
        elif scene in GAME_OVER_SCENES and self.status == NOT_PLAYING:
            # Script started during the post-game screens: report it, scores unknown
            self._set_status(GAME_OVER)

    def _new_game(self, status: str) -> None:
        self._saw_black = False
        self.banner = None
        self._set_status(status)
        self.player1 = self.player2 = None
        self._reset_votes()
        self._publish_banner(None)
        self.events.append(Event('scores', player1=None, player2=None))

    # ---- overlay ----
    def _reset_votes(self) -> None:
        self._votes = {1: {}, 2: {}}
        self._banner_votes = {}
        self._back_votes = {}
        self._below_votes = {1: {}, 2: {}}

    def _update_overlay(self, p1: Optional[int], p2: Optional[int], banner: Optional[str]) -> None:
        # Banner: plurality vote within the appearance. Applied before the scores so the
        # status is already right when a score event fires on the same frame.
        if banner is not None and self._banner_plausible(banner):
            self._banner_votes[banner] = self._banner_votes.get(banner, 0) + 1
            if self._banner_votes[banner] >= BANNER_CONFIRM_VOTES and self._is_plurality(self._banner_votes, banner):
                self._apply_banner(banner)
                self._banner_votes = {}
        elif banner in BANNER_RANK and self.status != GAME_OVER:
            self._back_votes[banner] = self._back_votes.get(banner, 0) + 1
            if self._restart_suspected(banner):
                self._restart(banner)
                return

        # Scores: each player votes independently; implausible readings are discarded
        if self.status == GAME_OVER:
            return
        accepted = {}
        for who, reading in ((1, p1), (2, p2)):
            if reading is None:
                continue
            if not self._score_plausible(who, reading):
                if self._current(who) is not None and reading < self._current(who):
                    below = self._below_votes[who]
                    below[reading] = below.get(reading, 0) + 1
                continue
            votes = self._votes[who]
            votes[reading] = votes.get(reading, 0) + 1
            if (votes[reading] >= SCORE_CONFIRM_VOTES and self._is_plurality(votes, reading)
                    and reading != self._current(who)):
                accepted[who] = reading
                self._votes[who] = {}
        if accepted:
            self._accept_scores(accepted)

    @staticmethod
    def _is_plurality(votes: Dict, key) -> bool:
        return all(votes[key] > n for k, n in votes.items() if k != key)

    def _current(self, who: int) -> Optional[int]:
        return self.player1 if who == 1 else self.player2

    def _score_plausible(self, who: int, value: int) -> bool:
        old = self._current(who)
        if old is None:
            return 0 <= value <= MAX_SCORE
        return old <= value <= old + MAX_PLAUSIBLE_JUMP      # scores never go down

    def _banner_plausible(self, banner: str) -> bool:
        if banner not in BANNER_RANK:
            return False
        if self.status == GAME_OVER:
            return banner == '1ST QUARTER'       # only a fresh game can follow a final
        return BANNER_RANK[banner] >= (BANNER_RANK[self.banner] if self.banner else 0)

    def _restart_suspected(self, banner: str) -> bool:
        """A lower-ranked banner that keeps being read while the scores go backwards too
        means the game was reset without a team-select/match-up scene being seen. Both
        are needed: a banner misread alone (OVERTIME read as HALFTIME, say) must not
        wipe a game in progress."""
        if self._back_votes.get(banner, 0) < RESTART_BANNER_VOTES or not self._is_plurality(self._back_votes, banner):
            return False
        return any(n >= RESTART_SCORE_VOTES for votes in self._below_votes.values() for n in votes.values())

    def _restart(self, banner: str) -> None:
        self._new_game(NOT_PLAYING)       # forget scores and banner, then start over from this banner
        self._apply_banner(banner)

    def _apply_banner(self, banner: str) -> None:
        status = BANNER_STATUS[banner]
        if self.status == GAME_OVER:     # missed the start screens: treat as a fresh game
            self._new_game(NOT_PLAYING)
        if banner != self.banner:
            self._publish_banner(banner)
        self._set_status(status)

    def _accept_scores(self, accepted: Dict[int, int]) -> None:
        old = {1: self.player1, 2: self.player2}
        self.player1 = accepted.get(1, self.player1)
        self.player2 = accepted.get(2, self.player2)
        self.events.append(Event('scores', player1=self.player1, player2=self.player2))
        for who in (1, 2):
            if who in accepted and old[who] is not None and accepted[who] > old[who]:
                self.events.append(Event('score', scorer=who, points=accepted[who] - old[who],
                                         player1=self.player1, player2=self.player2,
                                         quarter=Q1 if self.status == TIPOFF else self.status))

    # ---- emit helpers ----
    def _set_status(self, status: str) -> None:
        if status != self.status:
            self.status = status
            self.events.append(Event('status', status=status))

    def _publish_banner(self, banner: Optional[str]) -> None:
        self.banner = banner
        self.events.append(Event('banner', banner=banner))

    def _set_scores(self, p1: int, p2: int) -> None:
        self.player1, self.player2 = p1, p2
        self.events.append(Event('scores', player1=p1, player2=p2))
