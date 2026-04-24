# Stage 03 Docker isolation

Stage 03 (capture) runs `yt-dlp` and `ffmpeg` against untrusted video
content. Threat Model §11 R1 flags these as the largest residual risk
because a crafted YouTube response or a malicious input file could
exercise a 0day in either binary while it runs with full host
privileges.

The Docker backend runs the same binaries inside a hardened container,
eliminating that privilege. Stage 01/02/04/05 continue to execute on
the host because `claude -p` requires the OAuth session in the user's
home directory (Threat Model §16: "claude -p は譲らない").

## Scope

| Stage | Risky binary? | Backend |
|-------|---------------|---------|
| 01 Scripts | youtube-transcript-api (Python only, no subprocess) | host |
| 02 Summary | `claude -p` (OAuth) | host |
| 03 Capture | **yt-dlp + ffmpeg + gif2webp** | **host or docker** |
| 04 Learning | `claude -p` (OAuth) | host |
| 05 Synthesis | `claude -p` (OAuth) | host |

## Build the image

```bash
cd pipeline-youtube-repo
docker build -f docker/Dockerfile.capture -t pipeline-youtube-capture:latest .
```

The build installs `ffmpeg`, `gif2webp` (via the `webp` package), and a
pinned `yt-dlp` into a `python:3.13-slim` base. Total image size is
around 380 MB.

## Enable it

Either set `capture_backend` in `config.json`:

```json
{
  "vault_root": "/path/to/vault",
  "capture_backend": "docker",
  "capture_docker_image": "pipeline-youtube-capture:latest"
}
```

…or pass `--capture-backend docker` on the CLI for a one-off run.

## What the container can and cannot do

Every Stage 03 operation issues a new `docker run --rm` with these
flags (baked into `DockerCaptureBackend` — not user-configurable):

- `--read-only` — root filesystem is immutable
- `--cap-drop=ALL` — no kernel capabilities (no ptrace, no mount, etc.)
- `--security-opt=no-new-privileges:true` — setuid blocked
- `--user=<caller-uid>:<caller-gid>` — container runs as the same UID/GID that owns the host bind mounts, so writes to `tmp/` and `_assets/pipeline-youtube/` land with correct ownership. The caller's actual IDs are used (not a hard-coded 1000), which is required on hosts where the pipeline runs as root, a CI agent, or any non-1000 user
- `--tmpfs=/tmp:rw,size=512m,nosuid,nodev` — scratch space that vanishes on exit
- `--network=none` for `ffmpeg` / `gif2webp`, `--network=bridge` only for `yt-dlp`
- Bind mounts:
  - host `tmp/` → container `/work` (downloaded mp4 lives here)
  - host `Permanent Note/_assets/2026/pipeline-youtube/` → container `/assets` (extracted images land here)

Nothing else from the Vault or home directory is visible to the
container. If a 0day gives an attacker code execution inside the
container, the blast radius is bounded to the two bind-mounted
directories.

## Preflight

When `capture_backend="docker"` is active **and** Stage 03 will actually
run this invocation, the CLI runs a single preflight before any video
is processed:

1. `docker` CLI present in `PATH`
2. `docker image inspect pipeline-youtube-capture:latest` succeeds
3. Daemon responds within 15 s

If any check fails the CLI exits with a clear error and tells the user
how to fix it (build the image, start the daemon, or switch to `host`).

`--synthesis-only` and `--resume-reviewed` skip Stage 03, so the
preflight is deferred in those modes — you can run those workflows
even if Docker Desktop happens to be down.

## Trade-offs

- **Latency**: each `docker run` adds ~200-500 ms of container start
  overhead. With ~5 ffmpeg invocations per video, that's a few seconds
  per video at most.
- **Disk**: one 380 MB image.
- **Capability probe**: `ffmpeg -encoders` runs inside the container
  on first use (not cached across runs — each process starts cold).
- **No fallback**: if Docker is configured but the daemon is down at
  runtime, the stage fails loudly rather than silently reverting to
  host mode. This is intentional (Threat Model §14 principle 7:
  "検知だけでなく fail させる").

## When to keep host mode

- You trust your environment and the videos you process
- You already run the pipeline inside a VM / sandbox / Nix shell
- You need the lowest possible per-video latency

The default is still `host` to avoid surprising existing users. This
doc is linked from the README so new users can make an informed choice.
