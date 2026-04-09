function toTimestamp(value) {
  if (!value) return 0;
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function parsePaperSequence(paperId = '') {
  const match = String(paperId).match(/(\d+)(?!.*\d)/);
  return match ? Number(match[1]) : null;
}

function compareStage2Papers(a, b) {
  const aSequence = parsePaperSequence(a.paper_id);
  const bSequence = parsePaperSequence(b.paper_id);

  if (aSequence !== null && bSequence !== null && aSequence !== bSequence) {
    return aSequence - bSequence;
  }

  return toTimestamp(a.created_at) - toTimestamp(b.created_at);
}

function compareStage3Answers(a, b) {
  return toTimestamp(b.completion_date) - toTimestamp(a.completion_date);
}

function buildFallbackRun(sessionId, seedItem = null) {
  return {
    sessionId,
    displaySessionId: sessionId === 'legacy' ? 'Legacy' : sessionId,
    userPrompt: seedItem?.user_prompt || (sessionId === 'legacy' ? 'Legacy research session' : 'Unknown research question'),
    createdAt: seedItem?.created_at || seedItem?.completion_date || null,
    brainstormCount: null,
    sessionPaperCount: null,
    isLegacy: sessionId === 'legacy',
    isCurrent: false,
    stage2Papers: [],
    stage3Answers: [],
  };
}

export function buildResearchRunGroups({
  sessionsResponse = null,
  stage2Papers = [],
  stage3Answers = [],
}) {
  const sessions = sessionsResponse?.sessions || [];
  const currentSessionId = sessionsResponse?.current_session_id || null;
  const sessionMap = new Map();

  for (const session of sessions) {
    sessionMap.set(session.session_id, session);
  }

  const groups = new Map();

  const ensureGroup = (sessionId, seedItem = null) => {
    if (groups.has(sessionId)) {
      return groups.get(sessionId);
    }

    const session = sessionMap.get(sessionId);
    const group = session
      ? {
          sessionId,
          displaySessionId: sessionId === 'legacy' ? 'Legacy' : sessionId,
          userPrompt: session.user_prompt || seedItem?.user_prompt || 'Unknown research question',
          createdAt: session.created_at || seedItem?.created_at || seedItem?.completion_date || null,
          brainstormCount: session.brainstorm_count ?? null,
          sessionPaperCount: session.paper_count ?? null,
          isLegacy: sessionId === 'legacy',
          isCurrent: sessionId === currentSessionId,
          stage2Papers: [],
          stage3Answers: [],
        }
      : buildFallbackRun(sessionId, seedItem);

    groups.set(sessionId, group);
    return group;
  };

  for (const paper of stage2Papers) {
    const group = ensureGroup(paper.session_id, paper);
    group.stage2Papers.push(paper);
    if (!group.userPrompt && paper.user_prompt) {
      group.userPrompt = paper.user_prompt;
    }
    if (!group.createdAt) {
      group.createdAt = paper.created_at || null;
    }
  }

  for (const answer of stage3Answers) {
    const group = ensureGroup(answer.session_id, answer);
    group.stage3Answers.push(answer);
    if (!group.userPrompt && answer.user_prompt) {
      group.userPrompt = answer.user_prompt;
    }
    if (!group.createdAt) {
      group.createdAt = answer.completion_date || null;
    }
  }

  const runGroups = Array.from(groups.values()).map((group) => {
    const sortedStage2Papers = [...group.stage2Papers].sort(compareStage2Papers);
    const sortedStage3Answers = [...group.stage3Answers].sort(compareStage3Answers);
    const latestStage2At = sortedStage2Papers.length > 0 ? Math.max(...sortedStage2Papers.map((paper) => toTimestamp(paper.created_at))) : 0;
    const latestStage3At = sortedStage3Answers.length > 0 ? Math.max(...sortedStage3Answers.map((answer) => toTimestamp(answer.completion_date))) : 0;
    const latestActivityAt = Math.max(toTimestamp(group.createdAt), latestStage2At, latestStage3At);

    return {
      ...group,
      stage2Papers: sortedStage2Papers,
      stage3Answers: sortedStage3Answers,
      stage2PaperCount: sortedStage2Papers.length,
      stage3AnswerCount: sortedStage3Answers.length,
      hasStage3Answer: sortedStage3Answers.length > 0,
      latestActivityAt,
    };
  });

  runGroups.sort((a, b) => {
    if (a.isCurrent && !b.isCurrent) return -1;
    if (!a.isCurrent && b.isCurrent) return 1;
    return b.latestActivityAt - a.latestActivityAt;
  });

  return runGroups;
}
