import React from 'react';

export default function SecretInput({
  id,
  name,
  autoComplete = 'new-password',
  ...props
}) {
  return (
    <input
      {...props}
      id={id}
      name={name}
      type="password"
      autoComplete={autoComplete}
      autoCapitalize="none"
      autoCorrect="off"
      spellCheck={false}
    />
  );
}
