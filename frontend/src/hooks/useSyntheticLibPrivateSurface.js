import { useCallback, useEffect, useState } from 'react';

// Owns the SyntheticLib connectivity surface. The private developer workspace
// opens only while it is allowed and closes at once when that gate is lost
// (for example, developer mode is switched off with Shift + Z + X).
export default function useSyntheticLibPrivateSurface(showDeveloperWorkspace) {
  const [modalOpen, setModalOpen] = useState(false);
  const [privateSurfaceOpen, setPrivateSurfaceOpen] = useState(false);

  const open = useCallback(() => {
    setPrivateSurfaceOpen(Boolean(showDeveloperWorkspace));
    setModalOpen(true);
  }, [showDeveloperWorkspace]);

  const close = useCallback(() => {
    setModalOpen(false);
    setPrivateSurfaceOpen(false);
  }, []);

  useEffect(() => {
    if (privateSurfaceOpen && !showDeveloperWorkspace) close();
  }, [privateSurfaceOpen, showDeveloperWorkspace, close]);

  return { modalOpen, privateSurfaceOpen, open, close };
}
