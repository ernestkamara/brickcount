// CNN seam classifier (model trained in ml/, exported to /models/seam.onnx).
// Model spec: input "input" float32 [N,3,16,32] (RGB/255, crop spec in ./crop.ts),
// output "logit" [N,1]; seam = sigmoid(logit) > 0.5.
// The ONNX runtime is injected (session or loader) so the same code runs in the worker
// (onnxruntime-web WASM) and in Node eval. If the model can't be loaded the classifier falls
// back to the heuristic and records why in `fallbackReason`.
import { deltaE76 } from '../colour';
import { gaussianBlur } from '../image';
import type { PairDecision, StudPair } from '../types';
import { extractGapCrops } from './crop';
import { HeuristicSeamClassifier } from './heuristic';
import { CROP_H, CROP_W, type SeamClassifier, type SeamContext } from './index';

export interface OrtTensorLike {
  data: ArrayLike<number>;
}
export interface OrtSessionLike {
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>>;
}
export interface OrtLike {
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown;
  InferenceSession: { create(model: string | Uint8Array, opts?: unknown): Promise<OrtSessionLike> };
}

export interface CnnOptions {
  /** Ready session (tests / eval). Requires `ort` for the Tensor constructor. */
  session?: OrtSessionLike;
  ort?: OrtLike;
  /** Lazily load the runtime. Default: onnxruntime-web WASM build (browser). */
  loadOrt?: () => Promise<OrtLike>;
  /** Model URL or bytes. Default '/models/seam.onnx'. */
  model?: string | Uint8Array;
  batchSize?: number;
  /**
   * Gaussian pre-blur before cropping, in units of pitch/16 (crop pixel size). The model was
   * trained on synthetic crops; a light blur helps on sharp real photos (real-photo eval:
   * grid-ok piece P/R 0.72/0.73 -> 0.78/0.82 with 0.5). Default DEFAULT_BLUR_CROP_PX.
   */
  blurCropPx?: number;
}

export const DEFAULT_MODEL_URL = '/models/seam.onnx';
export const DEFAULT_BLUR_CROP_PX = 0.5;

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export class CnnSeamClassifier implements SeamClassifier {
  readonly name = 'cnn' as const;
  /** Set when the model failed to load/run and the heuristic was used instead. */
  fallbackReason: string | null = null;
  private session: OrtSessionLike | null;
  private ort: OrtLike | null;
  private loading: Promise<void> | null = null;
  private readonly heuristic = new HeuristicSeamClassifier();
  private readonly opts: CnnOptions;

  constructor(opts: CnnOptions = {}) {
    this.opts = opts;
    this.session = opts.session ?? null;
    this.ort = opts.ort ?? null;
  }

  /** Load the model (idempotent). Resolves even on failure; check `ready`. */
  async load(): Promise<void> {
    if (this.session && this.ort) return;
    this.loading ??= (async () => {
      try {
        const ort = this.ort ?? (await (this.opts.loadOrt ?? loadOrtWeb)());
        this.ort = ort;
        this.session ??= await ort.InferenceSession.create(this.opts.model ?? DEFAULT_MODEL_URL, {
          executionProviders: ['wasm'],
        });
        this.fallbackReason = null;
      } catch (e) {
        this.fallbackReason = `seam model unavailable (${e instanceof Error ? e.message : String(e)}); used heuristic`;
      }
    })();
    await this.loading;
  }

  get ready(): boolean {
    return !!(this.session && this.ort);
  }

  async classify(ctx: SeamContext, pairs: StudPair[]): Promise<PairDecision[]> {
    await this.load();
    if (!this.ready) return this.heuristic.classify(ctx, pairs);
    try {
      return await this.run(ctx, pairs);
    } catch (e) {
      this.fallbackReason = `seam model failed (${e instanceof Error ? e.message : String(e)}); used heuristic`;
      return this.heuristic.classify(ctx, pairs);
    }
  }

  private async run(ctx: SeamContext, pairs: StudPair[]): Promise<PairDecision[]> {
    const { cols, rows } = ctx.grid;
    const bs = this.opts.batchSize ?? 256;
    const blur = this.opts.blurCropPx ?? DEFAULT_BLUR_CROP_PX;
    const src = blur > 0 ? gaussianBlur(ctx.warped, (blur * ctx.warped.width) / cols / 16) : ctx.warped;
    const per = 3 * CROP_H * CROP_W;
    const probs = new Float32Array(pairs.length);
    for (let s = 0; s < pairs.length; s += bs) {
      const batch = pairs.slice(s, s + bs);
      const data = extractGapCrops(src, cols, rows, batch);
      const input = new this.ort!.Tensor('float32', data, [batch.length, 3, CROP_H, CROP_W]);
      const out = await this.session!.run({ input });
      const logits = (out.logit ?? Object.values(out)[0]).data;
      if (logits.length < batch.length || data.length !== batch.length * per) throw new Error('bad model output');
      for (let i = 0; i < batch.length; i++) probs[s + i] = sigmoid(Number(logits[i]));
    }
    const cell = (c: number, r: number) => ctx.cells[r * cols + c];
    return pairs.map((p, i) => {
      const a = cell(p.col, p.row);
      const b = p.dir === 'h' ? cell(p.col + 1, p.row) : cell(p.col, p.row + 1);
      const pr = probs[i];
      // A clearly different colour is a seam whatever the model says.
      if (a && b && a.colorId !== b.colorId && deltaE76(a.lab, b.lab) > 20) {
        return { ...p, seam: true, confidence: Math.max(0.8, pr) };
      }
      const seam = pr > 0.5;
      let conf = seam ? pr : 1 - pr;
      const g = Math.max(a?.glare ?? 0, b?.glare ?? 0);
      if (g > 0.3) conf = Math.min(conf, 0.55);
      return { ...p, seam, confidence: conf };
    });
  }
}

/** Browser default: the plain WASM build of onnxruntime-web (no JSEP/WebGPU; keeps precache small). */
async function loadOrtWeb(): Promise<OrtLike> {
  const [ort, wasm] = await Promise.all([
    import('onnxruntime-web/wasm'),
    import('onnxruntime-web/ort-wasm-simd-threaded.wasm?url'),
  ]);
  ort.env.wasm.wasmPaths = { wasm: wasm.default };
  // Workers can't spawn nested threads without cross-origin isolation.
  if (typeof crossOriginIsolated === 'undefined' || !crossOriginIsolated) ort.env.wasm.numThreads = 1;
  return ort as unknown as OrtLike;
}
