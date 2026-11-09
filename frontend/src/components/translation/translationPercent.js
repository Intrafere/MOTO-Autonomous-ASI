// Translation progress as whole percentages. Units are the service's internal paragraph
// boundaries, so the interface reports how much of a document is done instead of counts.

function toCount(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? Math.floor(number) : 0;
}

// Whole percent of `total` that is done, held below 100 until everything is done.
export function translationPercent(translated, total) {
  const all = toCount(total);
  const done = toCount(translated);
  if (all === 0 || done >= all) {
    return 100;
  }
  return Math.min(99, Math.floor((done * 100) / all));
}

export function formatPercent(translated, total) {
  return `${translationPercent(translated, total)}%`;
}

// Where a part (1-based first and last unit) sits in its document: "12–15%", or "12%"
// when both ends round to the same percent. Null when the total is unknown.
export function formatPartRange(start, end, total) {
  const all = toCount(total);
  const first = toCount(start);
  if (!all || !first) {
    return null;
  }
  const last = Math.max(first, toCount(end));
  const from = Math.min(100, Math.floor(((first - 1) * 100) / all));
  const to = Math.min(100, Math.max(from, Math.ceil((last * 100) / all)));
  return from === to ? `${from}%` : `${from}–${to}%`;
}
