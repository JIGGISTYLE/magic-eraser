# EraseAI: UI and erase audit

Tested locally on 18 November 2026 at http://localhost:8501.

## Real erase test

Used the browser to open the wind-turbine sample, paint over the distant turbine, and run Restore image. The turbine was removed. The masked area retained a softer, less detailed field texture than its surroundings. Runs after the dimension fix reported 0.5–1.0 s on CPU; these timings apply only to this image and selection.

Downloaded a PNG through the browser and verified it with Pillow: **512 × 365 pixels**. Before the fix, the same source produced **512 × 360 pixels**. Export evidence is saved alongside the repository as `erase-browser-result.png`; the visual comparison is `erase-comparison.png`.

## Findings and changes

| Finding | Resolution |
| --- | --- |
| Large title, tall uploader, repeated instructions pushed editing down the page. | Compact system-font heading, shorter spacing, upload/sample controls collapse after source selection. |
| Fixed-width canvas required horizontal scrolling on narrow screens. | Measure the editor's available width and fit the canvas while preserving proportions. |
| Large empty action column beside the canvas. | Move actions into a horizontal toolbar; center the bounded canvas on a matte surface. |
| Oversized result image looked softer than the export. | Bounded main layout and equal-width before/after comparison. |
| Neural inference silently discarded image edges not divisible by eight. | Pad input for inference, then remove padding from outputs. Preserve the intended image dimensions. |
| Old results remained current after painting or changing settings. | Match results against image content, mask bytes, selected models, and output size. Hide outdated exports with an explicit message. |
| Same filename and size could identify different uploaded images as one image. | Use a content hash; respect EXIF orientation. |
| Resizing initially displayed the selection but returned no mask pixels. | Preserve the full-size mask until the resized canvas emits pixel data; rescale existing drawing objects. |
| Extreme image aspect ratios could round one dimension to zero. | Clamp prepared and canvas dimensions to at least one pixel. |

## Verification performed

- Browser: paint a brush selection, restore, inspect before/after, download PNG.
- Browser: desktop at 1280 × 900 and mobile width at 390 × 844.
- Browser: paint at mobile width, resize to desktop, and successfully restore the retained selection.
- Browser: switch to rectangle, add a mark, and confirm the stale-result warning replaces the prior result/export.
- Streamlit AppTest: startup, all three samples, empty-mask warning, clear selection.
- Four unit tests: canvas bounds, padded dimensions including thin images, removing only inference padding, alpha-mask interpretation.
- Real model inference independently confirmed the preserved 512 × 365 output dimensions.
- Python compilation and git whitespace checks.

## Remaining shortcomings

1. **Repair quality:** DeepFill can produce blur, repeated texture, or implausible detail. One successful turbine removal does not establish quality for faces, large objects, or complex backgrounds.
2. **Resolution:** The UI processes at a longest edge of 256–768 px. Large uploads are resized, so exports are not original-resolution restorations.
3. **Interaction smoothness:** Streamlit reruns Python after drawing and widget changes. It can briefly refresh the UI; no continuous progress percentage or cancellation is available. It is not equivalent to a native graphics editor.
4. **Precision tools:** No zoom/pan, mask subtraction brush, feathering control, or brush-size keyboard shortcuts. Undo/redo is available in the canvas toolbar, but its history resets when the canvas is remounted on resizing; the visible selection is retained.
5. **Model availability:** Only the downloaded Places2 model is available. Portrait and converted models were not evaluated.
6. **Persistence:** Browser reloads can reset the edit session; no saved projects or automatic recovery.

Touch hardware, a full keyboard/screen-reader audit, very large-file stress testing, and all model/resolution combinations were not tested.

## Code cleanup follow-up

The UI and API now share padded inference through `editor_core.py`. API files use unique names, input errors return actionable responses, and project paths work independently of the launch directory. Requirements now list 11 direct runtime dependencies, including `gdown`, rather than a full development environment. The downloader runs only when invoked as a script. Duplicate CSS and retired branding assets were removed.

The official SVGs embed lossless crops instead of the entire original artwork: their combined size fell from 3,153,010 to 448,859 bytes (about 86%). A decoded-pixel comparison verified the crops match the original. The original source artwork remains intact.
