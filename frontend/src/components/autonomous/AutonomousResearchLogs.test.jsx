import { render, screen } from '@testing-library/react';
import { describe, expect, test, vi } from 'vitest';

import AutonomousResearchLogs from './AutonomousResearchLogs';

vi.mock('../../services/api', () => ({
  autonomousAPI: {
    getProofCompetitionBenchmarks: vi.fn().mockResolvedValue({ reports: [] }),
  },
}));

vi.mock('../ApiCallLogs', () => ({
  default: () => <div>API logs</div>,
}));

vi.mock('./ProofCompetitionBenchmarkReport', () => ({
  default: () => <div>Benchmarks</div>,
}));

describe('AutonomousResearchLogs', () => {
  test('renders bounded normalized validator feedback for rejected submissions', () => {
    render(
      <AutonomousResearchLogs
        stats={{}}
        events={[
          {
            event: 'submission_rejected',
            timestamp: '2026-10-06T12:00:00Z',
            validatorFeedback: 'The proposed route duplicates accepted evidence.',
            data: {
              submitter_id: 2,
              submitter_model: 'vendor/model',
              total_rejections: 4,
            },
          },
        ]}
      />,
    );

    expect(
      screen.getByText(/Validator feedback: The proposed route duplicates accepted evidence\./),
    ).toBeInTheDocument();
  });
});
