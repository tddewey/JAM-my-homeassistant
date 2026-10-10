"""Synthetic scenarios for GameTracker. Run: python -m unittest test_game_tracker"""

import unittest

from game_tracker import GameTracker


def overlay(tracker, p1, p2, banner, frames=8):
    """One overlay appearance followed by a gap, returning every event produced."""
    events = []
    for _ in range(frames):
        events += tracker.update('live', (p1, p2, banner))
    for _ in range(6):
        events += tracker.update('live', None)
    return events


def reach(tracker, p1, p2, banner):
    """Overlay appearances that walk the score up to (p1, p2) in plausible steps, as real play does."""
    events = []
    while True:
        a = min(p1, (tracker.player1 or 0) + 5)
        b = min(p2, (tracker.player2 or 0) + 5)
        events += overlay(tracker, a, b, banner)
        if (tracker.player1, tracker.player2) == (p1, p2):
            return events


def kinds(events, kind):
    return [e for e in events if e.kind == kind]


def start_game(tracker):
    """team_select -> matchup -> black -> live, then the first basket."""
    for scene in ['team_select'] * 2 + ['matchup'] * 2 + ['black'] * 2 + ['live'] * 2:
        tracker.update(scene, None)


class TrackerTest(unittest.TestCase):
    def test_first_basket_is_tagged_q1(self):
        t = GameTracker()
        start_game(t)
        self.assertEqual(t.status, 'tipoff')
        events = overlay(t, 3, 0, '1ST QUARTER')
        scored = kinds(events, 'score')
        self.assertEqual([(e.scorer, e.points, e.quarter) for e in scored], [(1, 3, 'q1')])
        self.assertEqual(t.status, 'q1')

    def test_overtime_banners_move_forward(self):
        t = GameTracker()
        start_game(t)
        overlay(t, 2, 0, '1ST QUARTER')
        reach(t, 30, 30, '4TH QUARTER')
        overlay(t, 30, 30, 'END OF 4TH QUARTER')
        overlay(t, 32, 30, 'OVERTIME')
        self.assertEqual((t.status, t.banner), ('overtime', 'OVERTIME'))
        overlay(t, 32, 32, 'DOUBLE OVERTIME')
        overlay(t, 34, 32, 'TRIPLE OVERTIME')
        overlay(t, 34, 32, 'TRIPLE OVERTIME')          # a fourth overtime reads the same
        self.assertEqual((t.status, t.banner), ('overtime', 'TRIPLE OVERTIME'))
        overlay(t, 34, 34, 'DOUBLE OVERTIME')            # backward misread alone changes nothing
        self.assertEqual((t.status, t.banner, t.player1, t.player2), ('overtime', 'TRIPLE OVERTIME', 34, 34))

    def test_banner_misread_alone_does_not_restart(self):
        t = GameTracker()
        start_game(t)
        overlay(t, 2, 0, '1ST QUARTER')
        reach(t, 40, 38, '4TH QUARTER')
        for _ in range(3):                               # consistent misread, scores still go forward
            overlay(t, 42, 38, 'HALFTIME', frames=12)
        self.assertEqual((t.status, t.player1, t.player2), ('q4', 42, 38))

    def test_restart_without_scenes_recovers(self):
        t = GameTracker()
        start_game(t)
        overlay(t, 2, 0, '1ST QUARTER')
        reach(t, 40, 38, '3RD QUARTER')
        # Game reset with no team_select/matchup ever seen: first basket of the new game
        events = overlay(t, 2, 0, '1ST QUARTER', frames=12)
        self.assertEqual((t.status, t.banner, t.player1, t.player2), ('q1', '1ST QUARTER', 2, 0))
        self.assertTrue([e for e in events if e.kind == 'status' and e.status == 'q1'])
        events = overlay(t, 2, 3, '1ST QUARTER')
        self.assertEqual([(e.scorer, e.points) for e in kinds(events, 'score')], [(2, 3)])

    def test_game_over_then_new_game(self):
        t = GameTracker()
        start_game(t)
        overlay(t, 2, 0, '1ST QUARTER')
        overlay(t, 40, 38, 'FINAL')
        self.assertEqual(t.status, 'game_over')
        start_game(t)
        self.assertEqual(t.status, 'tipoff')
        self.assertEqual((t.player1, t.player2), (0, 0))

    def test_black_timeout_clears_scores(self):
        t = GameTracker()
        start_game(t)
        reach(t, 40, 38, '4TH QUARTER')
        for _ in range(60):
            t.update('black', None)
        self.assertEqual(t.status, 'not_playing')
        overlay(t, 2, 0, '1ST QUARTER')
        self.assertEqual((t.status, t.player1, t.player2), ('q1', 2, 0))


if __name__ == '__main__':
    unittest.main()
