import {
  formatOpenRouterProviderLabel,
  getOpenRouterProviderTitle,
  isOpenRouterUsHostProvider,
  USA_HOST_TOOLTIP,
} from './openRouterSelection';

test('marks known USA OpenRouter host providers with a flag label and tooltip', () => {
  expect(isOpenRouterUsHostProvider('OpenAI')).toBe(true);
  expect(isOpenRouterUsHostProvider('Amazon Bedrock')).toBe(true);
  expect(formatOpenRouterProviderLabel('OpenAI')).toBe('🇺🇸 OpenAI');
  expect(getOpenRouterProviderTitle('OpenAI')).toBe(USA_HOST_TOOLTIP);
});

test('leaves non-USA OpenRouter host providers unmarked', () => {
  expect(isOpenRouterUsHostProvider('Mistral')).toBe(false);
  expect(formatOpenRouterProviderLabel('Mistral')).toBe('Mistral');
  expect(getOpenRouterProviderTitle('Mistral')).toBeUndefined();
});
