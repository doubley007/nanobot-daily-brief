# Nanobot Cron Worker

Cloudflare Worker that triggers the GitHub Actions `Daily Financial Brief`
workflow on time. GitHub's native cron can lag by hours during busy periods;
this Worker fires within seconds of its scheduled slot.

## Flow

```
Cloudflare Worker cron (UTC 01:30 = SGT 09:30)
   └── POST https://api.github.com/repos/<owner>/<repo>/dispatches
          event_type = "daily-brief"
                ↓
   GitHub Actions workflow listens on repository_dispatch
                ↓
   Runs app/daily_job.py
```

## One-time setup

```bash
# 1. Install Wrangler (Cloudflare CLI)
npm install -g wrangler

# 2. Log in (opens browser)
wrangler login

# 3. Set secrets
cd cron-worker
wrangler secret put GH_REPO        # paste: doubley007/nanobot-daily-brief
wrangler secret put GH_TOKEN       # paste: your fine-grained PAT

# 4. Deploy
wrangler deploy
```

## Required GitHub token

Create a **fine-grained** personal access token at
<https://github.com/settings/tokens?type=beta> with:

- Repository access: only `doubley007/nanobot-daily-brief`
- Repository permissions:
  - **Contents**: Read-only
  - **Actions**: Read and write
  - **Metadata**: Read-only (auto-included)

## Verify

After deploying, the Worker exposes a public URL like
`https://nanobot-daily-brief-cron.<your-subdomain>.workers.dev`.

- `GET /` → liveness check
- `GET /trigger` → manual dispatch (same payload as the cron)

If `GET /trigger` returns `{ "ok": true, "status": 204 }` you'll see a new
workflow run appear on GitHub within a few seconds under
**Actions → Daily Financial Brief** with the badge "repository_dispatch".

## Schedule

See `wrangler.toml`. Currently:

```
crons = ["30 1 * * *"]   # UTC 01:30 = SGT 09:30
```
