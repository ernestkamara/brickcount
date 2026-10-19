import { useEffect, useRef } from 'react';
import { clearSettings, DEFAULT_SETTINGS, type GridChoice, type Settings } from '../lib/settings';
import { clearPhotos, setSettings, useAppState } from '../lib/store';

export function SettingsDialog({ open, onClose }: { open: boolean; onClose(): void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const s = useAppState((st) => st.settings);
  const set = (patch: Partial<Settings>) => setSettings({ ...s, ...patch });

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog ref={ref} className="settings" onClose={onClose} aria-labelledby="settings-title">
      <form method="dialog">
        <h2 id="settings-title">Settings</h2>

        <fieldset>
          <legend>Grid size</legend>
          <div className="seg" role="radiogroup" aria-label="Grid size">
            {(['auto', '16', '32', '48', 'custom'] as GridChoice[]).map((g) => (
              <label key={g} className={s.grid === g ? 'on' : ''}>
                <input type="radio" name="grid" value={g} checked={s.grid === g} onChange={() => set({ grid: g })} />
                {g === 'auto' ? 'Auto' : g === 'custom' ? 'Custom' : `${g}×${g}`}
              </label>
            ))}
          </div>
          {s.grid === 'custom' && (
            <div className="row">
              <label className="num-field">
                Columns
                <input type="number" min={1} max={256} value={s.customCols} onChange={(e) => set({ customCols: Number(e.target.value) })} />
              </label>
              <label className="num-field">
                Rows
                <input type="number" min={1} max={256} value={s.customRows} onChange={(e) => set({ customRows: Number(e.target.value) })} />
              </label>
            </div>
          )}
        </fieldset>

        <label className="toggle">
          <input type="checkbox" checked={s.countBaseplate} onChange={(e) => set({ countBaseplate: e.target.checked })} />
          <span>
            Count baseplate
            <small>Off: studs matching the border/background colour are not counted as pieces.</small>
          </span>
        </label>

        <fieldset>
          <legend>Seam classifier</legend>
          <div className="seg" role="radiogroup" aria-label="Seam classifier">
            {(
              [
                ['heuristic', 'Heuristic'],
                ['cnn', 'Neural (beta)'],
              ] as const
            ).map(([v, l]) => (
              <label key={v} className={s.classifier === v ? 'on' : ''}>
                <input type="radio" name="classifier" value={v} checked={s.classifier === v} onChange={() => set({ classifier: v })} />
                {l}
              </label>
            ))}
          </div>
        </fieldset>

        <label className="toggle">
          <input type="checkbox" checked={s.cloudCheck} onChange={(e) => set({ cloudCheck: e.target.checked })} />
          <span>
            Cloud double-check
            <small>Sends the photo to a server for a second opinion. Off by default; everything else stays on this device.</small>
          </span>
        </label>

        <p className="hint">Changes apply to the next scan. Use “Re-scan” on a photo to apply them.</p>

        <div className="row spread">
          <button
            type="button"
            className="btn ghost danger"
            onClick={() => {
              if (!confirm('Remove all photos and reset settings?')) return;
              clearPhotos();
              clearSettings();
              setSettings({ ...DEFAULT_SETTINGS });
            }}
          >
            Clear all
          </button>
          <button type="submit" className="btn primary">Done</button>
        </div>
      </form>
    </dialog>
  );
}
