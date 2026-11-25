import { useEffect, useState } from 'react';

/** Page-wide drop target, so a miss never makes the browser navigate to the file. */
export function useFileDrop(enabled: boolean, onFile: (file: File) => void, onMany: () => void) {
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes('Files');
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      if (enabled && ++depth === 1) setDragging(true);
    };
    const leave = () => {
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDragging(false);
    };
    const over = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setDragging(false);
      if (!enabled) return;
      const files = e.dataTransfer?.files;
      if (!files?.length) return;
      if (files.length > 1) onMany();
      else onFile(files[0]!);
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('dragover', over);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('dragover', over);
      window.removeEventListener('drop', drop);
    };
  }, [enabled, onFile, onMany]);

  return dragging;
}
