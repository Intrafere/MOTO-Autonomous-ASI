const COMPILER_APPENDIX_START =
  '[HARD CODED THEOREMS APPENDIX START -- LEAN 4 VERIFIED THEOREMS BELOW]';
const COMPILER_APPENDIX_END =
  '[HARD CODED THEOREMS APPENDIX END -- ALL APPENDIX CONTENT SHOULD BE ABOVE THIS LINE]';
const EMPTY_APPENDIX =
  '[Theorems appendix - verified Lean 4 theorems not placed inline will appear here]';

const LEGACY_HEADER =
  /^=== PROOFS (?:GENERATED FROM|ATTACHED TO) THIS (?:PAPER|BRAINSTORM)(?: \(Lean 4 Verified\))? ===\s*$/gim;

const countWords = (value) => (value.match(/\S+/g) || []).length;

export const countText = (value = '') => ({
  words: countWords(value),
  characters: value.length,
});

const readProofId = (text) => {
  const explicit = text.match(/^\s*Proof ID:\s*(.+?)\s*$/im)?.[1]?.trim();
  if (explicit && explicit.toLowerCase() !== 'n/a') return explicit;
  return text.match(/^\s*Theorem\s*\(([^)]+)\)/im)?.[1]?.trim() || '';
};

const readTheoremName = (text, proofId) => {
  const explicit = text.match(/^\s*Theorem Name:\s*(.+?)\s*$/im)?.[1]?.trim();
  if (explicit) return explicit;
  const compilerHeader = text.match(
    /^\s*Theorem\s*\([^)]+\)(?:\s*\[[^\]]+\])?\s*-\s*(.+?)\s*$/im,
  )?.[1]?.trim();
  if (compilerHeader) return compilerHeader;
  if (proofId) return proofId;
  const legacyHeader = text.match(/^\s*Proof\s+\d+\s*:\s*(.+?)\s*$/im)?.[1]?.trim();
  if (legacyHeader) return legacyHeader;
  const statement = text.match(/^\s*(?:Theorem )?Statement:\s*(.+?)\s*$/im)?.[1]?.trim();
  return statement || proofId || 'Lean 4 verified theorem';
};

const splitProofEntries = (body) => {
  const cleaned = body.replace(EMPTY_APPENDIX, '').trim();
  if (!cleaned) return [];
  const starts = [...cleaned.matchAll(/^\s*(?:Theorem\s*\([^)]+\)|Proof\s+\d+\s*:)/gim)]
    .map((match) => match.index);
  if (!starts.length) return [];
  return starts.map((start, index) => {
    const nextStart = starts[index + 1] ?? cleaned.length;
    let end = nextStart;
    const separator = cleaned.indexOf('\n---', start);
    if (separator >= 0 && separator < nextStart) {
      const separatorEnd = cleaned.indexOf('\n', separator + 1);
      end = separatorEnd >= 0 ? separatorEnd + 1 : cleaned.length;
    }
    return cleaned.slice(start, end).trim();
  }).filter(Boolean);
};

const appendTextSegment = (segments, text) => {
  if (!text) return;
  const previous = segments[segments.length - 1];
  if (previous?.type === 'paper') previous.content += text;
  else segments.push({ type: 'paper', content: text });
};

/** Split paper/volume display content while preserving the original source. */
export function parsePaperProofs(source = '') {
  const ranges = [];
  let searchFrom = 0;
  while (searchFrom < source.length) {
    const start = source.indexOf(COMPILER_APPENDIX_START, searchFrom);
    if (start < 0) break;
    const endMarker = source.indexOf(COMPILER_APPENDIX_END, start + COMPILER_APPENDIX_START.length);
    if (endMarker < 0) break;
    ranges.push({
      start,
      end: endMarker + COMPILER_APPENDIX_END.length,
      bodyStart: start + COMPILER_APPENDIX_START.length,
      bodyEnd: endMarker,
    });
    searchFrom = endMarker + COMPILER_APPENDIX_END.length;
  }

  LEGACY_HEADER.lastIndex = 0;
  const legacyMatches = [...source.matchAll(LEGACY_HEADER)];
  legacyMatches.forEach((match, index) => {
    const bodyStart = match.index + match[0].length;
    const nextCompiler = source.indexOf(COMPILER_APPENDIX_START, bodyStart);
    const nextLegacy = legacyMatches[index + 1]?.index ?? -1;
    let end = source.length;
    for (const candidate of [nextCompiler, nextLegacy]) {
      if (candidate >= 0) end = Math.min(end, candidate);
    }
    const body = source.slice(bodyStart, end);
    const entries = splitProofEntries(body);
    if (!entries.length) return;
    const lastEntry = entries[entries.length - 1];
    ranges.push({
      start: match.index,
      end: bodyStart + body.lastIndexOf(lastEntry) + lastEntry.length,
      bodyStart,
      bodyEnd: bodyStart + body.lastIndexOf(lastEntry) + lastEntry.length,
    });
  });

  ranges.sort((a, b) => a.start - b.start);
  const nonOverlapping = ranges.filter((range, index) => (
    index === 0 || range.start >= ranges[index - 1].end
  ));
  const segments = [];
  const proofs = [];
  let cursor = 0;
  nonOverlapping.forEach((range) => {
    appendTextSegment(segments, source.slice(cursor, range.start));
    splitProofEntries(source.slice(range.bodyStart, range.bodyEnd)).forEach((content) => {
      const proofId = readProofId(content);
      const proof = {
        type: 'proof',
        content,
        proofId,
        theoremName: readTheoremName(content, proofId),
        number: proofs.length + 1,
      };
      proofs.push(proof);
      segments.push(proof);
    });
    cursor = range.end;
  });
  appendTextSegment(segments, source.slice(cursor));

  const paperText = segments.filter((segment) => segment.type === 'paper')
    .map((segment) => segment.content).join('');
  const proofText = proofs.map((proof) => proof.content).join('\n\n');
  return {
    source,
    segments,
    proofs,
    paperText,
    proofText,
    fallbackMetrics: {
      paper: countText(paperText),
      proofs: countText(proofText),
      total: countText(source),
    },
  };
}

const firstNumber = (record, keys) => {
  for (const key of keys) {
    if (Number.isFinite(record?.[key])) return record[key];
  }
  return undefined;
};

export function resolvePaperMetrics(record, parsed) {
  const fallback = parsed.fallbackMetrics;
  return {
    paper: {
      words: firstNumber(record, ['paper_word_count', 'content_word_count']) ?? fallback.paper.words,
      characters: firstNumber(record, ['paper_character_count', 'paper_char_count', 'content_character_count']) ?? fallback.paper.characters,
    },
    proofs: {
      words: firstNumber(record, ['proof_word_count', 'proofs_word_count']) ?? fallback.proofs.words,
      characters: firstNumber(record, ['proof_character_count', 'proof_char_count', 'proofs_character_count']) ?? fallback.proofs.characters,
    },
    total: {
      words: firstNumber(record, ['total_word_count', 'word_count']) ?? fallback.total.words,
      characters: firstNumber(record, ['total_character_count', 'character_count', 'char_count']) ?? fallback.total.characters,
    },
    proofCount: firstNumber(record, ['proof_count', 'proofs_count']) ?? parsed.proofs.length,
  };
}

export const paperProofMarkers = {
  COMPILER_APPENDIX_START,
  COMPILER_APPENDIX_END,
  EMPTY_APPENDIX,
};
