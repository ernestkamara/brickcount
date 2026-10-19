import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const fixture = (name: string) => fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url));

async function openMock(page: Page) {
  // Force the download path for exports (no Web Share in tests).
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'share', { value: undefined, configurable: true });
    Object.defineProperty(navigator, 'canShare', { value: undefined, configurable: true });
  });
  await page.goto('/?mock=1');
  await expect(page.getByText('Demo mode')).toBeVisible();
}

test('upload → queue → count, toggle a seam, export CSV', async ({ page }) => {
  await openMock(page);
  await page.getByTestId('file-input').setInputFiles(fixture('mosaic16.png'));

  const card = page.getByTestId('photo-card');
  await expect(card).toHaveCount(1);
  await expect(card.getByTestId('card-count')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId('batch-count')).toHaveText(await card.getByTestId('card-count').innerText());

  await card.click();
  const count = page.getByTestId('piece-count');
  await expect(count).toBeVisible();
  const before = Number(await count.innerText());
  expect(before).toBeGreaterThan(0);

  // Split a piece: tap seams inside pieces until the count changes (a single cut
  // inside a 2x2 keeps it connected, so it may take more than one).
  const inner = page.locator('line[data-seam="0"]');
  let changed = false;
  for (let i = 0; i < 12 && !changed; i++) {
    // A <line>'s box excludes its stroke (zero width for vertical lines), so tap its midpoint.
    const b = (await inner.nth(i * 7 + 20).boundingBox())!;
    await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2);
    changed = await expect(count).not.toHaveText(String(before), { timeout: 1500 }).then(() => true, () => false);
  }
  expect(changed).toBe(true);
  await expect(page.locator('line.seam.manual').first()).toBeAttached();

  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Export CSV' }).click()]);
  expect(download.suggestedFilename()).toMatch(/\.csv$/);
  const csv = await readFile(await download.path(), 'utf8');
  expect(csv.split('\n')[0]).toBe('photo,color_id,color_name,size,quantity');
  const total = csv.trim().split('\n').slice(1).reduce((s, l) => s + Number(l.split(',').at(-1)), 0);
  expect(total).toBe(Number(await count.innerText()));
});

test('grid failure shows corner editor, no count; re-scan with corners succeeds', async ({ page }) => {
  await openMock(page);
  await page.getByTestId('file-input').setInputFiles(fixture('wide.png'));
  const card = page.getByTestId('photo-card');
  await expect(card.getByText('Grid not found')).toBeVisible({ timeout: 15_000 });
  await expect(card.getByTestId('card-count')).toHaveCount(0);
  await card.click();
  await expect(page.getByRole('alert')).toContainText("Couldn't find the grid — adjust corners or retake");
  await expect(page.getByTestId('piece-count')).toHaveCount(0);

  // drag the top-left handle a bit (mouse), then re-scan
  const h = page.getByRole('slider', { name: 'top-left corner' });
  const box = (await h.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 5, box.y + box.height / 2 + 5, { steps: 3 });
  await page.mouse.up();
  await page.getByRole('button', { name: 'Re-scan with these corners' }).click();
  await expect(page.getByTestId('piece-count')).toBeVisible({ timeout: 15_000 });
});
