import { mkdir, writeFile } from 'node:fs/promises';
import { expect, test } from './fixtures.js';

// Built-app integration only: no component mounts, source imports, fake Workers,
// private renderer globals, live providers, Lean, or production testing branches.
const SECTION_COUNT = 100;
const ROOT = '.live-paper-progress';
function documentFixture(kind = 'prose', tag = 'STRESS') {
  const paragraph = kind === 'math'
    ? 'Independent evidence $\\frac{x^2+1}{1+x^2}=1$ supports the conditional result while $\\sum_{i=1}^{n}i=\\frac{n(n+1)}{2}$ records an explicit mathematical identity. '
    : 'Independent evidence supports this conditional research result with clear assumptions reproducible methods bounded uncertainty and explicitly documented limitations. ';
  const words = paragraph.trim().split(/\s+/).length;
  const repetitions = Math.ceil(500 / words);
  return Array.from({ length: SECTION_COUNT }, (_, index) => (
    `## ${tag}SECTION${String(index).padStart(3, '0')}\n\n${paragraph.repeat(repetitions)}\n\n${tag}END${String(index).padStart(3, '0')}\n\n`
  )).join('');
}

function progress(content, title = 'Renderer stress document') {
  return { has_paper: true, paper_id: 'renderer_stress', title, content,
    outline: 'Introduction\nBody\nConclusion', word_count: content.split(/\s+/).length,
    phase: 'body', is_compiling: true };
}

async function instrumentation(page) {
  const workers = [];
  page.on('worker', worker => workers.push(worker.url()));
  await page.addInitScript(() => {
    window.__rendererStressMetrics = { longTasks: [], longTaskSupported: false };
    if (PerformanceObserver.supportedEntryTypes?.includes('longtask')) {
      window.__rendererStressMetrics.longTaskSupported = true;
      new PerformanceObserver(list => {
        for (const entry of list.getEntries()) {
          window.__rendererStressMetrics.longTasks.push({ start: entry.startTime, duration: entry.duration });
        }
      }).observe({ type: 'longtask', buffered: true });
    }
  });
  return workers;
}

async function openDocument(page, mockApp, content) {
  mockApp.state.autonomousStatus = {
    is_running: true, current_tier: 'tier2_paper_writing', current_paper_id: 'renderer_stress',
    current_topic_id: 'stress_topic', is_tier3_active: false,
  };
  mockApp.state.responses.set('GET /api/auto-research/current-paper-progress', progress(content));
  await mockApp.open();
  const root = page.locator(ROOT);
  await expect(root).toBeVisible();
  await root.getByRole('checkbox', { name: 'Auto-scroll' }).uncheck();
  return root;
}

async function metrics(page) {
  return page.evaluate(() => ({
    ...window.__rendererStressMetrics,
    nodes: document.querySelectorAll('*').length,
    mountedChunks: document.querySelectorAll('.latex-chunk:not(.latex-chunk-placeholder)').length,
    totalChunks: document.querySelectorAll('.latex-chunk').length,
    heapBytes: performance.memory?.usedJSHeapSize ?? null,
  }));
}

async function attachReport(page, testInfo, workers, extra = {}) {
  const observed = page.isClosed() ? { metricsUnavailable: 'page closed after test failure' }
    : await metrics(page).catch(error => ({ metricsUnavailable: error.message }));
  const report = JSON.stringify({
    baseline: 'not measured: renderer changes are concurrent; no pristine built baseline was available',
    scope: 'Single local Chromium built-app run; observed timings and heap are not universal performance guarantees.',
    hostedLimitations: 'Root-path Vite preview only; production non-root proxy deployment, hosted CSP and real PDF backend were not validated.',
    workers, ...observed, ...extra,
    summary: {
      longTaskCount: observed.longTasks?.length ?? null,
      longestTaskMs: Math.max(0, ...(observed.longTasks || []).map(item => item.duration)),
      maxMountedChunks: Math.max(0, ...(extra.samples || []).map(item => item.mountedChunks || 0)),
      maxNodes: Math.max(0, ...(extra.samples || []).map(item => item.nodes || 0)),
      inputTimings: (extra.samples || []).filter(item => item.inputLatencyMs !== undefined),
    },
  }, null, 2);
  // Persist even with the default line reporter and successful trace removal.
  await mkdir(testInfo.outputDir, { recursive: true });
  const path = testInfo.outputPath('renderer-stress-metrics.json');
  await writeFile(path, report);
  await testInfo.attach('renderer-stress-metrics.json', { contentType: 'application/json', path });
}

test.describe('built renderer worker stress', () => {
  test.setTimeout(180_000);
  test.use({ actionTimeout: 15_000 });

  for (const kind of ['prose', 'math']) {
    test(`50k words ${kind}: real worker and bounded DOM across the entire document`, async ({ page, mockApp }, testInfo) => {
      const content = documentFixture(kind);
      expect(content.trim().split(/\s+/).length).toBeGreaterThanOrEqual(50_000);
      const workers = await instrumentation(page);
      const samples = [];
      try {
        const root = await openDocument(page, mockApp, content);
        const inputStarted = Date.now();
        await root.getByRole('checkbox', { name: 'Auto-scroll' }).check();
        await expect(root.getByRole('checkbox', { name: 'Auto-scroll' })).toBeChecked();
        samples.push({ inputLatencyMs: Date.now() - inputStarted, action: 'checkbox while initial rendering is active' });
        await root.getByRole('checkbox', { name: 'Auto-scroll' }).uncheck();
        await expect.poll(() => workers.length, { timeout: 20_000 }).toBeGreaterThan(0);
        expect(workers.some(url => /\/assets\/.*\.js(?:\?|$)/.test(url))).toBeTruthy();
        // The viewer separately renders its outline/disclaimer prefix. Traverse
        // the final renderer (the actual paper), not prefix chunk index zero.
        const chunks = root.locator('.latex-rendered-content').last().locator('.latex-chunk');
        await expect.poll(() => chunks.count()).toBeGreaterThan(10);
        const count = await chunks.count();
        // Visit every chunk, not just a final jump. Offscreen chunks must be
        // evicted again; progressive mount-once rendering fails this bound.
        for (let index = 0; index < count; index += 1) {
          const chunk = chunks.nth(index);
          await chunk.scrollIntoViewIfNeeded();
          await expect(chunk).not.toHaveClass(/latex-chunk-placeholder/, { timeout: 20_000 });
          const sample = await metrics(page);
          samples.push({ index, nodes: sample.nodes, mountedChunks: sample.mountedChunks, heapBytes: sample.heapBytes });
          expect(sample.mountedChunks, `mounted after visiting chunk ${index}`).toBeLessThanOrEqual(24);
          expect(sample.nodes, `DOM after visiting chunk ${index}`).toBeLessThan(40_000);
        }
        await expect(root.locator('.paper-proof-rendered-content')).toContainText('STRESSEND099');
        await chunks.first().scrollIntoViewIfNeeded();
        await expect(chunks.first()).toContainText('STRESSSECTION000');
        const started = Date.now();
        await root.getByRole('button', { name: 'Raw View', exact: true }).click();
        await expect(root.locator('.paper-proof-raw-content')).toContainText('STRESSEND099');
        samples.push({ inputLatencyMs: Date.now() - started, action: 'click raw through visible content' });
        expect(await root.locator('.paper-proof-raw-content').textContent()).toContain(content);
      } finally {
        await attachReport(page, testInfo, workers, { kind, words: content.split(/\s+/).length, samples });
      }
    });
  }

  test('rapid live updates and raw/document switches reject stale worker output', async ({ page, mockApp }, testInfo) => {
    const workers = await instrumentation(page);
    try {
      const root = await openDocument(page, mockApp, documentFixture('math', 'OLD'));
      await expect.poll(() => workers.length).toBeGreaterThan(0);
      for (let index = 0; index < 8; index += 1) {
        mockApp.state.responses.set('GET /api/auto-research/current-paper-progress', progress(documentFixture('prose', `REV${index}`)));
        await mockApp.sendWebSocket('paper_updated', { paper_id: 'renderer_stress' });
        await expect.poll(() => mockApp.requests('GET', '/api/auto-research/current-paper-progress').length).toBeGreaterThanOrEqual(index + 2);
      }
      await root.getByRole('button', { name: 'Raw View', exact: true }).click();
      await expect(root.locator('.paper-proof-raw-content')).toContainText('REV7END099');
      const replacement = '## FINALREPLACEMENT\n\nOnly the latest document may remain visible. $x^2=4$';
      mockApp.state.responses.set('GET /api/auto-research/current-paper-progress', progress(replacement));
      await mockApp.sendWebSocket('paper_updated', { paper_id: 'renderer_stress' });
      await expect(root.locator('.paper-proof-raw-content')).toContainText('FINALREPLACEMENT');
      await root.getByRole('button', { name: 'Rendered View', exact: true }).click();
      await expect(root.locator('.paper-proof-rendered-content')).toContainText('FINALREPLACEMENT');
      await expect(root).not.toContainText('OLDSECTION');
      await expect(root).not.toContainText('REV7SECTION');
      await expect(root.locator('.katex')).not.toHaveCount(0);
    } finally { await attachReport(page, testInfo, workers); }
  });

  for (const [kind, content] of [
    ['oversized atomic math', `ATOMICSTART\n\n$$${'x+'.repeat(100_000)}x$$\n\nATOMICEND`],
    ['unclosed malformed environment', `MALFORMEDSTART\n\\begin{align}\n${'x & = \\unknowncommand{y} \\\\\n'.repeat(12_000)}MALFORMEDEND`],
  ]) {
    test(`${kind}: bounded safe fallback preserves raw input`, async ({ page, mockApp }, testInfo) => {
      const workers = await instrumentation(page);
      try {
        const root = await openDocument(page, mockApp, content);
        // Fallback may be a readable escaped chunk or an explicit error notice;
        // it must never leave a perpetual loading-only viewer.
        await expect(root).toContainText(/ATOMICSTART|MALFORMEDSTART|unable|failed|too large|limit|unavailable/i, { timeout: 30_000 });
        await root.getByRole('button', { name: 'Raw View', exact: true }).click();
        expect(await root.locator('.paper-proof-raw-content').textContent()).toContain(content);
        expect((await metrics(page)).nodes).toBeLessThan(40_000);
      } finally { await attachReport(page, testInfo, workers, { kind }); }
    });
  }

  test('CSP worker denial gives a safe error and usable raw view', async ({ page, mockApp }, testInfo) => {
    const workers = await instrumentation(page);
    mockApp.state.documentCsp = "worker-src 'none'";
    const content = documentFixture('math', 'CSP');
    try {
      const root = await openDocument(page, mockApp, content);
      await expect(root).toContainText(/unable|failed|unavailable|blocked|could not|error/i, { timeout: 20_000 });
      expect(workers).toHaveLength(0);
      await root.getByRole('button', { name: 'Raw View', exact: true }).click();
      expect(await root.locator('.paper-proof-raw-content').textContent()).toContain(content);
    } finally { await attachReport(page, testInfo, workers, { workerBlockedBy: 'HTTP Content-Security-Policy' }); }
  });

  test('PDF preparation sends full offscreen contents to fake backend', async ({ page, mockApp }, testInfo) => {
    const workers = await instrumentation(page);
    // 50K dense KaTeX words correctly exceed the backend's 2 MiB HTML cap.
    // Keep a full 50K-word export below that cap, with math still exercised.
    const content = documentFixture('prose', 'PDF') + '\n\n$\\frac{x^2+1}{1+x^2}=1$';
    mockApp.state.features.pdf_download_available = true;
    const pdfBodies = [];
    const pdfErrors = [];
    const consoleErrors = [];
    page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
    page.on('pageerror', error => consoleErrors.push(error.message));
    page.on('dialog', async dialog => { pdfErrors.push(dialog.message()); await dialog.dismiss(); });
    // This route owns PDF generation completely: assert HTML, never call a real
    // backend/browser PDF renderer or claim the fake PDF validates pagination.
    await page.route('**/api/download/pdf', async route => {
      pdfBodies.push(route.request().postDataJSON());
      await route.fulfill({ status: 200, contentType: 'application/pdf', body: '%PDF-1.4\n% test-only fake backend\n%%EOF' });
    });
    try {
      const root = await openDocument(page, mockApp, content);
      const started = Date.now();
      let downloaded = false;
      page.once('download', () => { downloaded = true; });
      const pdfButton = root.getByRole('button', { name: 'PDF', exact: true });
      await page.evaluate(() => {
        window.__pdfPointerTrace = [];
        for (const type of ['pointerdown', 'pointerup', 'click']) {
          document.addEventListener(type, event => {
            const button = document.querySelector('.live-paper-progress .btn-download-pdf');
            const rect = button?.getBoundingClientRect();
            window.__pdfPointerTrace.push({ type, target: event.target?.outerHTML?.slice(0, 400),
              x: event.clientX, y: event.clientY, trusted: event.isTrusted, scrollY: window.scrollY,
              ancestors: button ? Array.from((function* () { for (let node = button.parentElement; node; node = node.parentElement) yield node; })()).map(node => ({ tag: node.tagName, className: node.className, top: node.getBoundingClientRect().top, scrollTop: node.scrollTop })) : [],
              checkbox: (() => { const node = document.querySelector('.live-paper-controls input'); const style = getComputedStyle(node); return { height: node.getBoundingClientRect().height, padding: style.padding, appearance: style.appearance, labelHeight: node.parentElement.getBoundingClientRect().height }; })(),
              button: rect ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height, disabled: button.disabled } : null });
          }, true);
        }
      });
      await pdfButton.click();
      await expect.poll(() => page.evaluate(() => window.__pdfPointerTrace.some(event => event.type === 'click' && event.target.includes('btn-download-pdf') && !event.target.startsWith('<div'))), { timeout: 3000 }).toBeTruthy();
      await expect.poll(() => downloaded || pdfErrors.length > 0 || consoleErrors.some(message => /PDF|download|render/i.test(message)), { timeout: 120_000 }).toBeTruthy();
      expect(consoleErrors.filter(message => /PDF|download|render/i.test(message))).toEqual([]);
      expect(pdfErrors, 'PDF preparation must succeed, not silently fail through an alert').toEqual([]);
      expect(downloaded).toBeTruthy();
      const pointerEvents = await page.evaluate(() => window.__pdfPointerTrace);
      const down = pointerEvents.find(event => event.type === 'pointerdown');
      const up = pointerEvents.find(event => event.type === 'pointerup');
      expect(down.trusted).toBeTruthy();
      expect(up.trusted).toBeTruthy();
      expect(up.target).toMatch(/^<button class="btn-download-pdf"/);
      expect(Math.abs(up.button.y - down.button.y)).toBeLessThan(2);
      expect(up.checkbox.height).toBe(down.checkbox.height);
      expect(pdfBodies).toHaveLength(1);
      const html = pdfBodies[0].html_body;
      expect(typeof html).toBe('string');
      for (let index = 0; index < SECTION_COUNT; index += 1) {
        expect(html).toContain(`PDFSECTION${String(index).padStart(3, '0')}`);
        expect(html).toContain(`PDFEND${String(index).padStart(3, '0')}`);
      }
      // Check repeated body text too: section boundary markers alone would not
      // catch a renderer that silently drops the middle of every section.
      expect((html.match(/Independent evidence/g) || []).length)
        .toBe((content.match(/Independent evidence/g) || []).length);
      expect((html.match(/records an explicit mathematical identity/g) || []).length)
        .toBe((content.match(/records an explicit mathematical identity/g) || []).length);
      expect(html).toContain('katex');
      expect(html).not.toMatch(/latex-chunk-placeholder|<script\b/i);
      testInfo.annotations.push({ type: 'pdfPreparationMs', description: String(Date.now() - started) });
      // The complementary dense-math export must fail explicitly without ever
      // publishing a partial second payload. Preserve the backend's real cap.
      mockApp.state.responses.set('GET /api/auto-research/current-paper-progress', progress(documentFixture('math', 'CAP')));
      await mockApp.sendWebSocket('paper_updated', { paper_id: 'renderer_stress' });
      await root.getByRole('button', { name: 'Raw View', exact: true }).click();
      await expect(root.locator('.paper-proof-raw-content')).toContainText('CAPEND099');
      await root.getByRole('button', { name: 'PDF', exact: true }).focus();
      await root.getByRole('button', { name: 'PDF', exact: true }).press('Enter');
      await expect.poll(() => pdfErrors.join('\n'), { timeout: 30_000 }).toMatch(/exceeds the PDF export limit/);
      expect(pdfBodies).toHaveLength(1);
    } finally { await attachReport(page, testInfo, workers, { pointerTrace: await page.evaluate(() => window.__pdfPointerTrace).catch(() => null), pdfRequests: pdfBodies.length, pdfErrors, consoleErrors, apiRequests: mockApp.state.requests.map(({ method, path }) => ({ method, path })), pdfScope: 'Preparation and full HTML transport only, not real PDF output.' }); }
  });
});
