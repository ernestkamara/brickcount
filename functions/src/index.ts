// Optional cloud double-check: asks a vision LLM for an independent piece count.
// Off by default in the app; never the primary counter. The photo is sent to Anthropic.
import Anthropic from '@anthropic-ai/sdk';
import { betaZodOutputFormat } from '@anthropic-ai/sdk/helpers/beta/zod';
import { onCall, HttpsError } from 'firebase-functions/v2/https';
import { defineSecret } from 'firebase-functions/params';
import { z } from 'zod';

const ANTHROPIC_API_KEY = defineSecret('ANTHROPIC_API_KEY');

const VisionCount = z.object({
  pieceCount: z.number().int(),
  studCount: z.number().int(),
  gridCols: z.number().int(),
  gridRows: z.number().int(),
  notes: z.string(),
});
export type VisionCount = z.infer<typeof VisionCount>;

const PROMPT = `This is a top-down photo of a flat LEGO mosaic made of plates on a baseplate.
Count the individual LEGO pieces (plates), NOT the studs. Adjacent same-colour plates are separated by thin dark seam lines; each seam means a separate piece.
Also report the total number of studs and the grid size in studs. Work row by row. In notes, mention regions where seams were hard to see (glare, blur).`;

const MAX_B64 = 8 * 1024 * 1024;

export const visionCheck = onCall({ secrets: [ANTHROPIC_API_KEY], timeoutSeconds: 300, memory: '512MiB' }, async (req) => {
  const image = req.data?.imageBase64;
  const mediaType = req.data?.mediaType ?? 'image/jpeg';
  if (typeof image !== 'string' || image.length === 0 || image.length > MAX_B64) {
    throw new HttpsError('invalid-argument', 'imageBase64 (≤8 MB) is required');
  }
  if (mediaType !== 'image/jpeg' && mediaType !== 'image/png') {
    throw new HttpsError('invalid-argument', 'mediaType must be image/jpeg or image/png');
  }

  const client = new Anthropic({ apiKey: ANTHROPIC_API_KEY.value() });
  try {
    const response = await client.beta.messages.parse({
      model: 'claude-opus-5-5',
      max_tokens: 16000,
      output_config: { effort: 'high', format: betaZodOutputFormat(VisionCount) },
      betas: ['server-side-fallback-2026-07-01'],
      fallbacks: 'default',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: mediaType, data: image } },
            { type: 'text', text: PROMPT },
          ],
        },
      ],
    });
    if (response.stop_reason === 'refusal') {
      throw new HttpsError('failed-precondition', 'The model declined this image');
    }
    if (!response.parsed_output) {
      throw new HttpsError('internal', `No structured output (stop_reason=${response.stop_reason})`);
    }
    return { ...response.parsed_output, model: response.model };
  } catch (err) {
    if (err instanceof HttpsError) throw err;
    if (err instanceof Anthropic.RateLimitError) throw new HttpsError('resource-exhausted', 'Rate limited, try again shortly');
    if (err instanceof Anthropic.BadRequestError) throw new HttpsError('invalid-argument', err.message);
    if (err instanceof Anthropic.APIError) throw new HttpsError('unavailable', `Vision API error ${err.status}`);
    throw err;
  }
});
