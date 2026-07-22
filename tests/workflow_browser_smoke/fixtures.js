// Browser specs live with the cross-field test artifacts while their locked Node
// runner is owned by the frontend package.
import { expect, test as base } from '../../frontend/node_modules/@playwright/test/index.js';

const HOSTED_FEATURES = {
  version: 'browser-smoke',
  build_commit: 'browser-smoke-commit',
  update_channel: 'main',
  api_contract_version: 'build-g-smoke',
  generic_mode: true,
  lm_studio_enabled: false,
  pdf_download_available: false,
  openai_codex_oauth_available: false,
  xai_grok_oauth_available: false,
  sakana_fugu_available: false,
};

const IDLE_STATUS = {
  is_running: false,
  current_tier: null,
  is_tier3_active: false,
  tier3_status: null,
};

const json = (route, body, status = 200) => route.fulfill({
  status,
  contentType: 'application/json',
  body: JSON.stringify(body),
});

function responseFor(method, path, state) {
  if (method === 'GET' && path === '/api/features') return state.features;
  if (method === 'GET' && path === '/api/openrouter/api-key-status') {
    return { success: true, has_key: state.hasOpenRouterKey, enabled: state.hasOpenRouterKey };
  }
  if (method === 'GET' && path === '/api/cloud-access/status') return { providers: {} };
  if (method === 'GET' && path === '/api/cloud-access/provider-notifications') return { notifications: [] };
  if (method === 'GET' && path === '/api/connectivity/status') {
    return {
      inference: {
        openrouter_oauth: { status: state.hasOpenRouterKey ? 'ready' : 'inactive' },
        lm_studio: { status: 'inactive' },
      },
      skills: {
        syntheticlib4: { status: 'Coming soon', enabled: false },
        agent_conversation_memory: { status: 'ready', enabled: true },
        wolfram_alpha: { status: 'inactive', enabled: false },
      },
    };
  }
  if (method === 'GET' && path === '/api/update-notice') return { update_available: false };
  if (method === 'GET' && path === '/api/aggregator/models') return { models: [] };
  if (method === 'GET' && path === '/api/aggregator/status') return { is_running: false };
  if (method === 'GET' && path === '/api/aggregator/prompt') return { prompt: '' };
  if (method === 'GET' && path === '/api/compiler/status') return { is_running: false };
  if (method === 'GET' && path === '/api/leanoj/status') return { is_running: false, phase: 'idle' };
  if (method === 'GET' && path === '/api/auto-research/status') return state.autonomousStatus;
  if (method === 'GET' && path === '/api/auto-research/prompt') return { prompt: '' };
  if (method === 'GET' && path === '/api/auto-research/brainstorms') return { brainstorms: [] };
  if (method === 'GET' && path === '/api/auto-research/papers') return { papers: [] };
  if (method === 'GET' && path === '/api/auto-research/stats') {
    return { total_papers_completed: 0, paper_counts: { active: 0, pruned: 0 } };
  }
  if (method === 'GET' && path === '/api/auto-research/current-session') return { session: null };
  if (method === 'GET' && path === '/api/auto-research/tier3/status') return { is_active: false, status: 'idle' };
  if (method === 'GET' && path === '/api/auto-research/tier3/final-answer') return { has_final_answer: false };
  if (method === 'GET' && path === '/api/auto-research/tier3/volume-progress') return { is_long_form: false };
  if (method === 'GET' && path === '/api/proofs/status') {
    return { lean4_enabled: false, smt_enabled: false, manual_check_ready: false };
  }
  if (method === 'GET' && path === '/api/proof-search/assistant/latest-pack') {
    return { available: false, supports: [] };
  }
  if (method === 'GET' && path === '/api/workflow/predictions') return { mode: null, tasks: [] };
  if (method === 'GET' && path === '/api/workflow/solution-path') return { active: false, state: null };
  if (method === 'GET' && path === '/api/token-stats') {
    return { total_input: 0, total_output: 0, by_model: {}, elapsed_seconds: 0 };
  }
  if (method === 'GET' && path === '/api/boost/status') {
    return { enabled: false, boost_next_count: 0, always_prefer: false, boosted_categories: [] };
  }
  if (method === 'GET' && path === '/api/boost/categories') return { categories: [] };
  if (method === 'POST' && path === '/api/auto-research/start') {
    state.autonomousStatus = { ...IDLE_STATUS, is_running: true, current_tier: 'topic_exploration' };
    return { success: true, status: 'started' };
  }
  if (method === 'POST' && path === '/api/auto-research/stop') {
    state.autonomousStatus = { ...IDLE_STATUS };
    return { success: true, status: 'stopped' };
  }
  return undefined;
}

export const test = base.extend({
  mockApp: async ({ page }, use) => {
    const state = {
      features: { ...HOSTED_FEATURES },
      hasOpenRouterKey: true,
      autonomousStatus: { ...IDLE_STATUS },
      requests: [],
      unexpectedRequests: [],
    };

    await page.addInitScript(() => {
      class MockWebSocket {
        static OPEN = 1;
        static CLOSED = 3;
        static instances = [];

        constructor(url) {
          this.url = url;
          this.readyState = MockWebSocket.OPEN;
          MockWebSocket.instances.push(this);
          queueMicrotask(() => this.onopen?.({ type: 'open' }));
        }

        send() {}

        close() {
          this.readyState = MockWebSocket.CLOSED;
          this.onclose?.({ type: 'close' });
        }

        serverSend(message) {
          this.onmessage?.({ data: JSON.stringify(message) });
        }
      }

      window.WebSocket = MockWebSocket;
      window.__motoBrowserSmoke = {
        send(type, data, timestamp = '2026-07-14T12:00:00.000Z') {
          const socket = MockWebSocket.instances.at(-1);
          if (!socket) throw new Error('MOTO WebSocket has not connected');
          socket.serverSend({ type, data, timestamp });
        },
      };
    });

    await page.route('**/*', async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      const isLocal = ['127.0.0.1', 'localhost'].includes(url.hostname);
      if (!isLocal) {
        state.unexpectedRequests.push(`${request.method()} ${request.url()}`);
        await route.abort('blockedbyclient');
        return;
      }
      if (!url.pathname.startsWith('/api/')) {
        await route.continue();
        return;
      }

      const entry = {
        method: request.method(),
        path: `${url.pathname}${url.search}`,
        body: request.postDataJSON?.() ?? null,
      };
      state.requests.push(entry);
      const response = responseFor(request.method(), url.pathname, state);
      if (response === undefined) {
        state.unexpectedRequests.push(`${request.method()} ${url.pathname}`);
        await json(route, { detail: `Unmocked browser-smoke API: ${request.method()} ${url.pathname}` }, 501);
        return;
      }
      await json(route, response);
    });

    const api = {
      state,
      async open({ acknowledgeDisclaimer = true } = {}) {
        await page.goto('/');
        if (acknowledgeDisclaimer) {
          await page.getByRole('button', { name: 'I Have Read and Acknowledge This Disclaimer' }).click();
        }
      },
      requests(method, path) {
        return state.requests.filter((request) => request.method === method && request.path === path);
      },
      async sendWebSocket(type, data, timestamp) {
        await page.evaluate(({ eventType, payload, eventTimestamp }) => {
          window.__motoBrowserSmoke.send(eventType, payload, eventTimestamp);
        }, { eventType: type, payload: data, eventTimestamp: timestamp });
      },
      async assertNoUnexpectedRequests() {
        expect(state.unexpectedRequests).toEqual([]);
      },
    };

    await use(api);
  },
});

export { expect };
