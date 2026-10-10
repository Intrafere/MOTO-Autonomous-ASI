import { render, screen } from '@testing-library/react';
import { describe, expect, test } from 'vitest';
import SecretInput from './SecretInput';

describe('SecretInput', () => {
  test('uses autofill-safe API credential attributes', () => {
    render(
      <SecretInput
        aria-label="Provider key"
        id="provider-key"
        name="moto-provider-key"
        value=""
        onChange={() => {}}
      />
    );

    const input = screen.getByLabelText('Provider key');
    expect(input).toHaveAttribute('type', 'password');
    expect(input).toHaveAttribute('name', 'moto-provider-key');
    expect(input).toHaveAttribute('autocomplete', 'new-password');
    expect(input).toHaveAttribute('autocapitalize', 'none');
    expect(input).toHaveAttribute('autocorrect', 'off');
    expect(input).toHaveAttribute('spellcheck', 'false');
  });

  test('allows one-time OAuth material to disable autocomplete', () => {
    render(
      <SecretInput
        aria-label="OAuth callback"
        id="oauth-callback"
        name="moto-oauth-callback"
        autoComplete="off"
        value=""
        onChange={() => {}}
      />
    );
    expect(screen.getByLabelText('OAuth callback')).toHaveAttribute('autocomplete', 'off');
  });
});
