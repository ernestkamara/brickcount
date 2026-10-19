// LEGO plate colours seen in the sample mosaics plus common mosaic colours.
// ids are stable keys used in labels, breakdowns and ml/. rgb = typical photo appearance.
export interface PaletteColor {
  id: string;
  name: string;
  rgb: [number, number, number];
}

export const PALETTE: PaletteColor[] = [
  { id: 'medium_azure', name: 'Medium Azure', rgb: [104, 195, 226] },
  { id: 'dark_azure', name: 'Dark Azure', rgb: [7, 139, 201] },
  { id: 'blue', name: 'Bright Blue', rgb: [0, 85, 191] },
  { id: 'red', name: 'Bright Red', rgb: [201, 26, 9] },
  { id: 'dark_red', name: 'Dark Red', rgb: [160, 22, 50] },
  { id: 'yellow', name: 'Bright Yellow', rgb: [242, 205, 55] },
  { id: 'light_yellow', name: 'Bright Light Yellow', rgb: [245, 215, 110] },
  { id: 'tan', name: 'Tan', rgb: [228, 205, 158] },
  { id: 'light_nougat', name: 'Light Nougat', rgb: [246, 215, 179] },
  { id: 'reddish_brown', name: 'Reddish Brown', rgb: [88, 42, 18] },
  { id: 'dark_brown', name: 'Dark Brown', rgb: [53, 33, 0] },
  { id: 'black', name: 'Black', rgb: [27, 42, 52] },
  { id: 'white', name: 'White', rgb: [244, 244, 244] },
  { id: 'green', name: 'Green', rgb: [35, 120, 65] },
  { id: 'bright_pink', name: 'Bright Pink', rgb: [228, 173, 200] },
  { id: 'orange', name: 'Orange', rgb: [254, 138, 24] },
  { id: 'lime', name: 'Lime', rgb: [187, 233, 11] },
  { id: 'light_bluish_gray', name: 'Light Bluish Gray', rgb: [160, 165, 169] },
  { id: 'dark_bluish_gray', name: 'Dark Bluish Gray', rgb: [108, 110, 104] },
];

export const PALETTE_BY_ID: Record<string, PaletteColor> = Object.fromEntries(PALETTE.map((c) => [c.id, c]));

/** Real plate sizes (short side first). Used by validation and the synthetic renderer. */
export const LEGO_PLATE_SIZES: [number, number][] = [
  [1, 1], [1, 2], [1, 3], [1, 4], [1, 6], [1, 8], [1, 10], [1, 12],
  [2, 2], [2, 3], [2, 4], [2, 6], [2, 8], [2, 10], [2, 12], [2, 16],
  [3, 3], [4, 4], [4, 6], [4, 8], [6, 6],
];
