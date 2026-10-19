import { useCallback, useState } from 'react';
import { removePhoto, type PhotoItem } from '../lib/store';
import { annotatedPng, safeName, shareOrDownload, toCsv, toJson } from '../lib/export';
import { CornerEditor } from './CornerEditor';
import { defaultCorners } from './geometry';
import { StatusChip } from './QueueList';
import { rescan, toggleSeam } from './queue';
import { Results } from './Results';
import { ReviewView } from './ReviewView';

export function PhotoDetail({ photo, onBack }: { photo: PhotoItem; onBack(): void }) {
  const [editCorners, setEditCorners] = useState(false);
  const { result } = photo;
  const onToggle = useCallback((c: number, r: number, d: 'h' | 'v') => void toggleSeam(photo.id, c, r, d), [photo.id]);
  const initialCorners = photo.corners ?? result?.grid?.corners ?? defaultCorners(photo.width, photo.height);
  const submitCorners = (q: Parameters<typeof rescan>[1]) => {
    setEditCorners(false);
    rescan(photo.id, q);
  };

  return (
    <article className="detail" aria-labelledby="detail-title">
      <div className="detail-head">
        <button type="button" className="btn ghost small" onClick={onBack}>← All photos</button>
        <h2 id="detail-title" className="ellipsis">{photo.name}</h2>
        <StatusChip p={photo} />
      </div>

      {(photo.state === 'queued' || photo.state === 'processing') && (
        <div className="pending">
          <img src={photo.url} alt="Photo being scanned" className="photo" />
          <progress max={1} value={photo.frac ?? 0} aria-label="Scan progress" />
        </div>
      )}

      {photo.state === 'grid_failed' && (
        <div className="failed">
          <p className="alert warn" role="alert">Couldn't find the grid — adjust corners or retake.</p>
          <CornerEditor url={photo.url} width={photo.width} height={photo.height} initial={initialCorners} onSubmit={submitCorners} />
        </div>
      )}

      {photo.state === 'error' && (
        <div className="failed">
          <p className="alert err" role="alert">Scan failed: {photo.message ?? 'unknown error'}</p>
          <button type="button" className="btn" onClick={() => rescan(photo.id)}>Try again</button>
        </div>
      )}

      {photo.state === 'done' && result?.grid && (
        editCorners ? (
          <CornerEditor url={photo.url} width={photo.width} height={photo.height} initial={initialCorners} onSubmit={submitCorners} onCancel={() => setEditCorners(false)} />
        ) : (
          <>
            <ReviewView url={photo.url} width={photo.width} height={photo.height} result={result} onToggle={onToggle} />
            <p className="hint">
              Tap the line between two studs to split or merge pieces. <span className="amber">Amber</span> = low confidence, worth a check.
            </p>
            <Results result={result} />
            <div className="row wrap">
              <ExportButtons photo={photo} />
              <button type="button" className="btn ghost" onClick={() => setEditCorners(true)}>Adjust corners</button>
              <button type="button" className="btn ghost" onClick={() => rescan(photo.id)}>Re-scan</button>
            </div>
          </>
        )
      )}

      <div className="row">
        <button
          type="button"
          className="btn ghost danger small"
          onClick={() => {
            removePhoto(photo.id);
            onBack();
          }}
        >
          Remove photo
        </button>
      </div>
    </article>
  );
}

function ExportButtons({ photo }: { photo: PhotoItem }) {
  const r = photo.result!;
  const base = safeName(photo.name);
  const data = [{ name: photo.name, result: r }];
  return (
    <>
      <button type="button" className="btn" onClick={() => void shareOrDownload(new Blob([toCsv(data)], { type: 'text/csv' }), `${base}-parts.csv`)}>
        Export CSV
      </button>
      <button type="button" className="btn" onClick={() => void shareOrDownload(new Blob([toJson(data)], { type: 'application/json' }), `${base}-parts.json`)}>
        JSON
      </button>
      <button type="button" className="btn" onClick={async () => void shareOrDownload(await annotatedPng(photo.url, r), `${base}-annotated.png`)}>
        Annotated PNG
      </button>
    </>
  );
}
