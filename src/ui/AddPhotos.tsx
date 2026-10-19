import { useEffect, useRef, useState } from 'react';
import { addFiles } from './queue';
import { CameraModal } from './CameraModal';

const ACCEPT = 'image/*,.heic,.heif';

export function AddPhotos({ compact }: { compact?: boolean }) {
  const [camera, setCamera] = useState(false);
  const [dragging, setDragging] = useState(false);
  const hasCamera = typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia;

  // Whole-window drop target on desktop.
  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => !!e.dataTransfer?.types.includes('Files');
    const enter = (e: DragEvent) => { if (hasFiles(e)) { depth++; setDragging(true); } };
    const leave = (e: DragEvent) => { if (hasFiles(e) && --depth <= 0) { depth = 0; setDragging(false); } };
    const over = (e: DragEvent) => { if (hasFiles(e)) e.preventDefault(); };
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      setDragging(false);
      if (e.dataTransfer?.files.length) void addFiles(e.dataTransfer.files);
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
  }, []);

  return (
    <section className={compact ? 'add compact' : 'add'} aria-label="Add photos">
      {!compact && (
        <div className="add-intro">
          <h2>Count the pieces in a LEGO mosaic</h2>
          <p>Take a straight-on photo of a flat, framed mosaic. Everything runs on this device.</p>
        </div>
      )}
      <div className="add-actions">
        {hasCamera ? (
          <button type="button" className="btn primary" onClick={() => setCamera(true)}>
            Take photo
          </button>
        ) : (
          <FilePick className="btn primary" capture label="Take photo" />
        )}
        <FilePick className="btn" label="Upload photos" multiple testId="file-input" />
      </div>
      {!compact && <p className="drop-hint">or drop JPEG, PNG or HEIC files anywhere</p>}
      {dragging && <div className="drop-overlay" aria-hidden>Drop photos to count</div>}
      {camera && <CameraModal onClose={() => setCamera(false)} />}
    </section>
  );
}

function FilePick({ label, className, multiple, capture, testId }: { label: string; className: string; multiple?: boolean; capture?: boolean; testId?: string }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <label className={className}>
      {label}
      <input
        ref={input}
        className="visually-hidden"
        type="file"
        accept={ACCEPT}
        multiple={multiple}
        data-testid={testId}
        {...(capture ? { capture: 'environment' as const } : {})}
        onChange={(e) => {
          const files = e.target.files ? [...e.target.files] : [];
          e.target.value = '';
          if (files.length) void addFiles(files);
        }}
      />
    </label>
  );
}
