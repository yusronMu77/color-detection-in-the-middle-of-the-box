# color-detection-in-the-middle-of-the-box
Mendetkesi warna pada tengah-tengah kotak mengunakan rata-rata warna dan warna dominan.
rata-rata warna mengunakan algoritma rata-rata biasa.
warna dominan mengunakan algoritma kmeans.

## English

A small OpenCV prototype that samples a fixed region of interest (ROI) from a live webcam feed
and reports its color two ways: a plain per-channel average, and the dominant color found by
k-means clustering (`cv2.kmeans`, 5 clusters) over the pixels in that region.

### Running it

```bash
python progv3.py
```

Requires a camera at index 0 (`cv2.VideoCapture(0)`) and `opencv-python` + `numpy` installed.

### Controls

- `w` / `s` — move the ROI up / down
- `a` / `d` — move the ROI left / right
- `Esc` — quit

### What you'll see

Two windows: `ViOut` shows the live camera feed with a rectangle marking the current ROI, and
`Color` shows a solid swatch filled with the ROI's dominant color plus its HSV value printed on
top of the swatch.

### Note

This is an early prototype (2018) and mixes Python 2 and 3 syntax (e.g. a bare `print` statement
alongside f-string-free Python 3 code), so it may need small changes to run on a modern
interpreter.
