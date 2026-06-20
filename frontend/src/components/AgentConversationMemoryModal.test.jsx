import { render, screen } from '@testing-library/react';
import { describe, expect, test, vi } from 'vitest';
import AgentConversationMemoryModal from './AgentConversationMemoryModal';

const baseConnectivityStatus = {
  skills: {
    agent_conversation_memory: {
      enabled: true,
      status: 'active',
      message: 'Ready',
      local_records: 7,
    },
  },
};

describe('AgentConversationMemoryModal', () => {
  test('explains Assistant workflow-memory scope and critique exclusion', () => {
    render(
      <AgentConversationMemoryModal
        isOpen
        onClose={vi.fn()}
        connectivityStatus={baseConnectivityStatus}
      />
    );

    expect(screen.getByText(/Assistant runs in parallel during brainstorming, writing, and proof work/i)).toBeInTheDocument();
    expect(screen.getByText(/retrieves up to 7 relevant records/i)).toBeInTheDocument();
    expect(screen.getByText(/disabled during critique phases/i)).toBeInTheDocument();
    expect(screen.getByText(/not raw provider transcript storage/i)).toBeInTheDocument();
  });
});

