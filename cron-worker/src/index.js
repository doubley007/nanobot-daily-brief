/**
 * Cloudflare Worker cron trigger for nanobot daily brief.
 *
 * Why this exists:
 *   GitHub Actions' native cron is best-effort and routinely delayed
 *   by tens of minutes — sometimes hours — during busy periods. This
 *   Worker fires with sub-minute accuracy and pokes the GitHub API
 *   via `repository_dispatch`, which triggers the same workflow.
 *
 * Secrets (set with `wrangler secret put <name>`):
 *   GH_REPO   = "doubley007/nanobot-daily-brief"
 *   GH_TOKEN  = fine-grained PAT with Actions: read & write
 */

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(triggerWorkflow(env, { reason: "scheduled" }));
  },

  // Lets us verify the Worker is alive by curling its public URL.
  // Also handy for manual kicks when debugging.
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/trigger") {
      const result = await triggerWorkflow(env, { reason: "manual" });
      return new Response(JSON.stringify(result, null, 2), {
        status: result.ok ? 200 : 500,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response(
      "nanobot cron worker — POST /trigger or wait for the cron schedule.\n",
      { status: 200 },
    );
  },
};

async function triggerWorkflow(env, { reason }) {
  const repo = env.GH_REPO;
  const token = env.GH_TOKEN;

  if (!repo || !token) {
    return { ok: false, error: "GH_REPO or GH_TOKEN not configured" };
  }

  const resp = await fetch(
    `https://api.github.com/repos/${repo}/dispatches`,
    {
      method: "POST",
      headers: {
        "Accept": "application/vnd.github+json",
        "Authorization": `Bearer ${token}`,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "nanobot-cron-worker",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        event_type: "daily-brief",
        client_payload: {
          reason,
          triggered_at: new Date().toISOString(),
        },
      }),
    },
  );

  // GitHub returns 204 No Content on success.
  if (resp.status === 204) {
    return { ok: true, status: 204, reason };
  }
  const body = await resp.text();
  return { ok: false, status: resp.status, body };
}
