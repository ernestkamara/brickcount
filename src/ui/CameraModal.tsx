import { useEffect, useRef, useState } from 'react';
import { addFiles } from './queue';

/** In-app camera (getUserMedia) with a square framing guide. Falls back to the native capture input. */
export function CameraModal({ onClose }: { onClose(): void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [shots, setShots] = useState(0);

  useEffect(() => {
    dialog.current?.showModal();
    let stream: MediaStream | null = null;
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 3840 }, height: { ideal: 2160 } }, audio: false })
      .then((s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        if (video.current) {
          video.current.srcObject = s;
          void video.current.play().catch(() => {});
        }
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Camera unavailable'));
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const capture = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d')!.drawImage(v, 0, 0);
    c.toBlob(
      (b) => {
        if (!b) return;
        const n = shots + 1;
        setShots(n);
        void addFiles([new File([b], `Camera ${new Date().toLocaleTimeString()}.jpg`, { type: 'image/jpeg' })]);
      },
      'image/jpeg',
      0.92,
    );
  };

  return (
    <dialog ref={dialog} className="camera" onClose={onClose} aria-label="Camera">
      {error ? (
        <div className="camera-error">
          <p>Couldn't open the camera ({error}).</p>
          <label className="btn primary">
            Use system camera
            <input
              className="visually-hidden"
              type="file"
              accept="image/*"
              capture="environment"
              onChange={(e) => {
                const f = e.target.files ? [...e.target.files] : [];
                if (f.length) void addFiles(f);
                dialog.current?.close();
              }}
            />
          </label>
          <button type="button" className="btn ghost" onClick={() => dialog.current?.close()}>Close</button>
        </div>
      ) : (
        <>
          <div className="camera-view">
            <video ref={video} playsInline muted autoPlay />
            <div className="frame-guide" aria-hidden>
              <span />
            </div>
            <p className="camera-tip">Fill the square with the mosaic, hold the phone parallel to it</p>
          </div>
          <div className="camera-bar">
            <button type="button" className="btn ghost" onClick={() => dialog.current?.close()}>
              {shots ? `Done (${shots})` : 'Cancel'}
            </button>
            <button type="button" className="shutter" aria-label="Capture photo" onClick={capture} />
            <span className="spacer" />
          </div>
        </>
      )}
    </dialog>
  );
}
