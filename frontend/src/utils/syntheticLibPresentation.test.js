import { describe, expect, test } from 'vitest';
import {
  getSyntheticLibPresentation,
  normalizeSyntheticLibProductState,
} from './syntheticLibPresentation';

describe('SyntheticLib presentation policy', () => {
  test.each([undefined, null, '', 'future_state', {}, 'LAUNCHED'])(
    'normalizes unknown lifecycle value %p to coming soon',
    (value) => {
      expect(normalizeSyntheticLibProductState(value)).toBe('coming_soon');
      expect(getSyntheticLibPresentation(value, true)).toMatchObject({
        showComingSoon: true,
        showIntegration: false,
        isPrivatePreview: false,
      });
    },
  );

  test.each(['coming_soon', 'developer_preview'])(
    'keeps %s private until browser developer mode is enabled',
    (state) => {
      expect(getSyntheticLibPresentation(state, false).showComingSoon).toBe(true);
      expect(getSyntheticLibPresentation(state, true)).toMatchObject({
        showComingSoon: false,
        showIntegration: true,
        isPrivatePreview: true,
      });
    },
  );

  test('shows a launched integration without developer mode', () => {
    expect(getSyntheticLibPresentation('launched', false)).toMatchObject({
      showComingSoon: false,
      showIntegration: true,
      isLaunched: true,
    });
  });

  test('keeps the developer workspace behind developer mode after launch', () => {
    expect(getSyntheticLibPresentation('launched', false)).toMatchObject({
      showTrade: true,
      showDeveloperWorkspace: false,
    });
    expect(getSyntheticLibPresentation('launched', true).showDeveloperWorkspace).toBe(true);
    expect(getSyntheticLibPresentation('future_state', true).showDeveloperWorkspace).toBe(false);
  });

  test.each(['coming_soon', 'developer_preview'])(
    'keeps %s on Coming soon in generic/hosted mode even in developer mode',
    (state) => {
      expect(getSyntheticLibPresentation(state, true, true)).toMatchObject({
        showComingSoon: true,
        showIntegration: false,
        showTrade: false,
        showDeveloperWorkspace: false,
        isPrivatePreview: false,
      });
    },
  );

  test('never shows proof-trade or developer tooling in generic mode after launch', () => {
    expect(getSyntheticLibPresentation('launched', true, true)).toMatchObject({
      showIntegration: true,
      showTrade: false,
      showDeveloperWorkspace: false,
    });
  });
});
