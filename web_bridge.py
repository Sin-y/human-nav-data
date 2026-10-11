"""Browser bridge for collect_demo.

Runs the unmodified simulation (collect_demo.py / visual_gym.py) under Pyodide.
The page sends key states and draws the returned state; recording and saving
go through collect_demo.Recorder, so the .npz output matches the desktop tool.
"""

import io
import json
import time
import base64
from types import SimpleNamespace

from collect_demo import Recorder, build_env, episode_seed, action_from_direction
from visual_gym import WAIT

# Must match the argparse defaults in collect_demo.main().
DEFAULT_ARGS = dict(
    subject="anon", episodes=10, practice=2, control="direct", fps=10,
    map=None, no_random=False, n_obstacles=3, player="cross",
    player_speed=0.22, player_radius=0.30, styles="open_clusters,rooms",
)


def make_args(**overrides):
    from visual_gym import DEFAULT_MAP
    d = dict(DEFAULT_ARGS)
    d["map"] = DEFAULT_MAP
    d.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**d)


class WebRecorder(Recorder):
    """Recorder that also stores the number of trimmed leading WAIT steps (for replay)."""

    def commit_episode(self, info, seed, trim_lead_wait=True):
        a = self._cur["acts"] if self._cur else []
        lead = 0
        if trim_lead_wait:
            while lead < len(a) - 1 and a[lead] == WAIT:
                lead += 1
        ok = super().commit_episode(info, seed, trim_lead_wait)
        if ok:
            meta = json.loads(self.ep_meta[-1])
            meta["lead_trim"] = lead
            meta["client"] = "web"
            self.ep_meta[-1] = json.dumps(meta)
        return ok


class Session:
    def __init__(self, subject, episodes=10, practice=2, stamp=None):
        self.args = make_args(subject=subject, episodes=int(episodes), practice=int(practice))
        self.env = build_env(self.args)
        self.rec = WebRecorder()
        self.ep = 0
        self.saved = 0
        self.seed = None
        self.obs = None
        self.info = {}
        self.done = False
        # Pyodide runs in UTC, so the page passes its local timestamp.
        self.stamp = stamp or time.strftime('%m%d_%H%M%S')

    @property
    def total(self):
        return self.args.practice + self.args.episodes

    def start_episode(self):
        self.seed = episode_seed(self.args.subject, self.ep)
        self.obs, _ = self.env.reset(seed=self.seed)
        self.info = {}
        self.done = False
        self.rec.begin_episode()
        e = self.env
        return json.dumps({
            "ep": self.ep, "total": self.total,
            "practice": self.ep < self.args.practice,
            "saved": self.saved,
            "w": int(e.cfg.width), "h": int(e.cfg.height),
            "grid": e.grid.astype(int).tolist(),
            "start": [float(v) for v in e.start],
            "goal": [float(v) for v in e.goal],
            "goal_radius": float(e.cfg.goal_radius),
            **self._dyn(),
        })

    def _dyn(self):
        e = self.env
        movers = []
        for i in range(e.n_active):
            ox, oy = e.obs_pos[i]
            if ox > -5:
                person = bool(i == 0 and e.has_player)
                movers.append([float(ox), float(oy),
                               float(e.cfg.player_radius if person else e.cfg.obstacle_radius),
                               1 if person else 0])
        return {"agent": [float(v) for v in e.agent_pos],
                "heading": float(e.heading_rad()), "movers": movers}

    def step(self, up, down, left, right):
        e = self.env
        a = action_from_direction(bool(up), bool(down), bool(left), bool(right),
                                  e.heading_idx, e.n_headings, e.cfg.heading_step_deg)
        self.rec.record(self.obs, a, e.agent_pos)
        self.obs, r, term, trunc, self.info = e.step(a)
        out = self._dyn()
        out["end"] = bool(term or trunc)
        if out["end"]:
            out["success"] = bool(self.info.get("is_success"))
        return json.dumps(out)

    def finish_episode(self):
        practice = self.ep < self.args.practice
        if practice:
            self.rec.discard_episode()
        elif self.rec.commit_episode(self.info, self.seed):
            self.saved += 1
        self.ep += 1
        return json.dumps({"saved": self.saved, "ep": self.ep, "total": self.total,
                           "all_done": self.ep >= self.total, "recorded": not practice})

    def discard_episode(self):
        if self.rec._cur is not None:
            self.rec.discard_episode()

    def filename(self):
        return f"{self.args.subject}_{self.stamp}.npz"

    def npz_bytes(self):
        buf = io.BytesIO()
        self.rec.save(buf, self.args.subject, self.args.fps)
        return buf.getvalue()

    def export(self):
        if not self.rec.n_episodes:
            return json.dumps({"n": 0})
        data = self.npz_bytes()
        succ = sum(json.loads(m).get("is_success", False) for m in self.rec.ep_meta)
        return json.dumps({"n": self.rec.n_episodes, "succ": int(succ),
                           "steps": len(self.rec.acts), "filename": self.filename(),
                           "b64": base64.b64encode(data).decode("ascii")})


_session = None


def new_session(subject, episodes=10, practice=2, stamp=None):
    global _session
    _session = Session(subject, episodes, practice, stamp)
    return _session
