import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

// Runs the real OpenCV/ONNX worker (no mock) on a golden photo, with each seam classifier.
const golden = (path: string) => fileURLToPath(new URL(`../../fixtures/${path}`, import.meta.url));

for (const classifier of ['heuristic', 'cnn'] as const) {
  test(`real worker counts a golden photo (${classifier})`, async ({ page }) => {
    test.setTimeout(120_000);
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
    await page.addInitScript((c) => {
      localStorage.setItem('brickcount.settings.v1', JSON.stringify({ classifier: c }));
    }, classifier);
    await page.goto('/');
    await page.getByTestId('file-input').setInputFiles(golden('images/03_flower.jpg'));

    const card = page.getByTestId('photo-card');
    await expect(card.getByTestId('card-count').or(card.getByText('Error'))).toBeVisible({ timeout: 90_000 });
    await expect(card.getByText('Error'), errors.join('\n')).toHaveCount(0);

    const n = Number(await card.getByTestId('card-count').innerText());
    const label = JSON.parse(await readFile(golden('labels/03_flower.json'), 'utf8'));
    expect(n).toBeGreaterThan(label.pieces.length * 0.8);
    expect(n).toBeLessThan(label.pieces.length * 1.2);

    await card.click();
    await expect(page.getByTestId('piece-count')).toHaveText(String(n));
    expect(errors).toEqual([]);
  });
}
