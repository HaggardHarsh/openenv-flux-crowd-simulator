"""
Flux Visualization Server
==================================
Lightweight HTTP server that bridges the web dashboard to the CrowdManagementEnv.
Serves static files and exposes a REST API for reset/step/state.
"""

import http.server
import json
import os
import sys
import random
from urllib.parse import urlparse

# Handle Windows console encoding (cp1252 can't render emojis/box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crowd_env import CrowdManagementEnv, Action, ActionType
from crowd_env.models import RiskLevel
from crowd_env.agent import smart_heuristic
from stable_baselines3 import PPO
from crowd_env.rl_wrapper import CrowdGymWrapper

# ─── Global RL Model ────────────────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ppo_crowd_final.zip")
rl_model = None
rl_wrapper = None

try:
    if os.path.exists(model_path):
        rl_model = PPO.load(model_path)
        rl_wrapper = CrowdGymWrapper(task_id="hard")
except Exception as e:
    print(f"Warning: Failed to load RL model ({e}). Falling back to heuristic.")


# ─── Environment Instance ────────────────────────────────────────────────────

env = CrowdManagementEnv()
_initialized = False

# ─── Request Handler ─────────────────────────────────────────────────────────

VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization")


class FluxHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for both static files and API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=VIZ_DIR, **kwargs)

    def do_GET(self):
        """Handle API GET requests or fallback to static files."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/state":
            self._handle_state()
        elif path == "/health":
            self._json_response({"status": "ok"})
        else:
            # Serve static files
            super().do_GET()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        """Handle API POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
            try:
                req = json.loads(post_data)
            except json.JSONDecodeError:
                req = {}
        else:
            req = {}

        if path == "/reset":
            self._handle_reset(req)
        elif path == "/step":
            self._handle_step(req)
        else:
            self.send_error(404, "API endpoint not found")

    # ── API Handlers ──

    def _handle_reset(self, body):
        global _initialized
        task = body.get("task", "easy") if body else "easy"
        seed = body.get("seed", 42) if body else 42

        obs = env.reset(seed=seed, options={"task": task})
        _initialized = True

        self._json_response({
            "observation": obs.to_dict(),
            "state": env.state().to_dict(),
        })

    def _handle_step(self, body):
        global _initialized
        if not _initialized:
            self._json_response({"error": "Environment not initialized. Call /reset first."}, 400)
            return

        # Parse action
        if body and body.get("action_type") == "auto":
            if env._sim is None:
                obs = env.reset(seed=42, options={"task": "easy"})
            else:
                obs = env._build_observation()
                
            if rl_model and rl_wrapper:
                obs_array = rl_wrapper._encode_observation(obs)
                action_idx, _ = rl_model.predict(obs_array, deterministic=True)
                action = rl_wrapper._action_table[int(action_idx)]
            else:
                action = smart_heuristic(obs)
        elif body:
            action_type = body.get("action_type", "no_op")
            action = Action(
                action_type=action_type,
                source_zone=body.get("source_zone", ""),
                target_zone=body.get("target_zone", ""),
                gate_index=body.get("gate_index", 0),
                gate_open=body.get("gate_open", True),
            )
        else:
            action = Action.noop()

        result = env.step(action)
        grade = env.grade()

        self._json_response({
            "observation": result.observation.to_dict(),
            "reward": result.reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "info": result.info,
            "env_state": env.state().to_dict(),
            "grade": grade.to_dict(),
        })

    def _handle_state(self):
        if not _initialized:
            self._json_response({"error": "Not initialized"}, 400)
            return
        self._json_response(env.state().to_dict())

    # ── Helpers ──

    def _read_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            raw = self.rfile.read(content_length)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode("utf-8"))

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        """Suppress default request logging for cleaner output."""
        if "/api/" in str(args[0]):
            return  # Skip API call logs
        super().log_message(format, *args)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    server = http.server.HTTPServer(("0.0.0.0", port), FluxHandler)

    print("═" * 60)
    print("  🛡  Flux — AI Crowd Management Simulator")
    print("═" * 60)
    print(f"  Server running at: http://localhost:{port}")
    print(f"  Visualization:     http://localhost:{port}/index.html")
    print(f"  API endpoints:")
    print(f"    POST /reset   — Reset environment")
    print(f"    POST /step    — Execute action")
    print(f"    GET  /state   — Get state")
    print(f"    GET  /health  — Health check")
    print("═" * 60)
    print("  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
