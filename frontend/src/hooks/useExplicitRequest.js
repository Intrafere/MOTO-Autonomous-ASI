import { useEffect, useRef, useState } from 'react';

export const IDLE_REQUEST = Object.freeze({ status: 'idle', data: null, error: '' });

// One explicit request per call; becoming inactive or unmounting advances the
// generation so late responses are ignored and the state returns to idle. The
// fetcher receives a trailing `{ isCurrent }` context so multi-step requests
// can stop before a follow-up call once the surface has closed.
export default function useExplicitRequest(isActive, fetcher, describeError) {
  const [state, setState] = useState(IDLE_REQUEST);
  const generationRef = useRef(0);

  useEffect(() => {
    if (!isActive) {
      generationRef.current += 1;
      setState(IDLE_REQUEST);
    }
  }, [isActive]);

  useEffect(() => () => {
    generationRef.current += 1;
  }, []);

  const run = async (...args) => {
    generationRef.current += 1;
    const generation = generationRef.current;
    setState({ status: 'running', data: null, error: '' });
    const isCurrent = () => generation === generationRef.current;
    try {
      const data = await fetcher(...args, { isCurrent });
      if (generation !== generationRef.current) return false;
      setState({ status: 'complete', data, error: '' });
      return true;
    } catch (error) {
      if (generation !== generationRef.current) return false;
      setState({ status: 'error', data: null, error: describeError(error) });
      return false;
    }
  };

  return [state, run];
}
