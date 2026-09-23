import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, test, vi } from 'vitest';

vi.mock('./LatexRenderer', () => ({ default: ({ content, documentId }) => <div data-testid="latex-segment" data-document={documentId}>{content}</div> }));
import PaperProofViewer from './PaperProofViewer';

const content = [
  'Readable paper body.',
  '=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===',
  'Proof 1: Addition identity',
  'Proof ID: proof_add',
  'Lean 4 Code:',
  'by omega',
  '---',
].join('\n');

describe('PaperProofViewer', () => {
  test('propagates document and segment identity across document switches', () => {
    const { rerender, container } = render(<PaperProofViewer documentId="session:a" prefixContent="Outline" content={content} />);
    const ids = () => screen.getAllByTestId('latex-segment').map(node => node.dataset.document);
    expect(ids()).toEqual(['session:a:prefix', 'session:a:paper:0', 'session:a:proof:proof_add:1']);
    container.querySelector('details').open = true;
    rerender(<PaperProofViewer documentId="session:b" prefixContent="New outline" content={content.replace('Readable', 'New')} />);
    expect(ids()).toEqual(['session:b:prefix', 'session:b:paper:0', 'session:b:proof:proof_add:1']);
    expect(container.querySelector('details').open).toBe(false);
    expect(screen.getByRole('button', { name: 'Rendered View' })).toHaveAttribute('aria-pressed', 'true');
  });
  test('defaults rendered with collapsed proofs and preserves complete raw source', async () => {
    const user = userEvent.setup();
    const { container } = render(<PaperProofViewer content={content} />);
    const proof = screen.getByText('Proof 1: proof_add').closest('details');
    expect(proof.open).toBe(false);
    expect(screen.getByRole('button', { name: 'Rendered View' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByLabelText('Paper content metrics')).toHaveTextContent('Proofs (1)');

    await user.click(screen.getByRole('button', { name: 'Raw View' }));
    expect(container.querySelector('.paper-proof-raw-content').textContent).toBe(content);
  });
});
