import { useEffect, useState } from 'react';
import { useAppState } from './lib/store';
import { AddPhotos } from './ui/AddPhotos';
import { BatchSummary } from './ui/BatchSummary';
import { PhotoDetail } from './ui/PhotoDetail';
import { QueueList } from './ui/QueueList';
import { SettingsDialog } from './ui/SettingsDialog';
import { wantsMock } from './ui/workerClient';

const photoFromHash = () => new URLSearchParams(location.hash.slice(1)).get('photo');

export default function App() {
  const photos = useAppState((s) => s.photos);
  const [openId, setOpenId] = useState<string | null>(photoFromHash);
  const [settings, setSettingsOpen] = useState(false);
  const mock = wantsMock();

  useEffect(() => {
    const onPop = () => setOpenId(photoFromHash());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const open = (id: string) => {
    history.pushState(null, '', `#photo=${id}`);
    setOpenId(id);
    window.scrollTo(0, 0);
  };
  const back = () => {
    if (photoFromHash()) history.back();
    setOpenId(null);
  };

  const current = openId ? photos.find((p) => p.id === openId) : undefined;

  return (
    <>
      <header className="topbar">
        <button type="button" className="brand" onClick={back} aria-label="BrickCount home">
          <img src="/favicon.svg" alt="" width={28} height={28} />
          <span>BrickCount</span>
        </button>
        <button type="button" className="icon-btn" aria-label="Settings" onClick={() => setSettingsOpen(true)}>
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden>
            <path fill="currentColor" d="M19.4 13a7.5 7.5 0 0 0 0-2l2-1.6-2-3.4-2.4 1a7.4 7.4 0 0 0-1.7-1L15 3.5h-4l-.4 2.5a7.4 7.4 0 0 0-1.7 1l-2.4-1-2 3.4L6.6 11a7.5 7.5 0 0 0 0 2l-2 1.6 2 3.4 2.4-1a7.4 7.4 0 0 0 1.7 1l.4 2.5h4l.4-2.5a7.4 7.4 0 0 0 1.7-1l2.4 1 2-3.4zM13 15.5a3.5 3.5 0 1 1 0-7 3.5 3.5 0 0 1 0 7z" transform="translate(-1 0)" />
          </svg>
        </button>
      </header>
      {mock && <div className="banner" role="status">Demo mode: using the mock worker, counts are fake.</div>}
      <main>
        {current ? (
          <PhotoDetail key={current.id} photo={current} onBack={back} />
        ) : (
          <>
            <AddPhotos compact={photos.length > 0} />
            {photos.length > 0 && <QueueList photos={photos} onOpen={open} />}
            <BatchSummary photos={photos} />
          </>
        )}
      </main>
      <SettingsDialog open={settings} onClose={() => setSettingsOpen(false)} />
    </>
  );
}
