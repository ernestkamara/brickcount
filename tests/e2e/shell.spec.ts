import { expect, test } from '@playwright/test';

test('app shell loads with PWA manifest and works offline (real worker path, no mock)', async ({ page, context }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /Count the pieces/ })).toBeVisible();
  await expect(page.getByText('Demo mode')).toHaveCount(0);

  const href = await page.locator('link[rel="manifest"]').getAttribute('href');
  expect(href).toBeTruthy();
  const manifest = await (await page.request.get(href!)).json();
  expect(manifest.name).toBe('BrickCount');
  expect(manifest.icons.some((i: { purpose?: string }) => i.purpose === 'maskable')).toBe(true);

  // Service worker takes control, then the shell reloads offline.
  await page.evaluate(() => navigator.serviceWorker.ready.then(() => undefined));
  await page.waitForFunction(() => !!navigator.serviceWorker.controller, null, { timeout: 15_000 }).catch(() => page.reload());
  await page.waitForFunction(() => !!navigator.serviceWorker.controller, null, { timeout: 15_000 });
  await context.setOffline(true);
  await page.reload();
  await expect(page.getByRole('heading', { name: /Count the pieces/ })).toBeVisible();
  await context.setOffline(false);
});
