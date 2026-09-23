import { renderLatexToHtml } from './latexConverter';
self.onmessage = ({ data: { id, text } }) => {
  try { self.postMessage({ id, html: renderLatexToHtml(text) }); }
  catch (error) { self.postMessage({ id, error: String(error?.message || error) }); }
};
