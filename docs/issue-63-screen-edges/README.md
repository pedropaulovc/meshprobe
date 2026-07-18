# Issue 63 screen-space edge experiment

This compares MeshProbe's Freestyle `shaded_edges` render with an experimental GPU
compositor pass on the 409-component harmonic-analyzer GLTF. Both images use the same
1024x1024 camera, scene hash, lighting, and 64 EEVEE samples.

| Render | Wall time |
| --- | ---: |
| Plain `shaded` | 5.16 s |
| Freestyle `shaded_edges` | 50.71 s |
| Experimental `screen_edges` | 10.61 s |

The screen pass is 4.78 times faster than Freestyle in this run. It applies Sobel filters to
the depth and normal passes in Blender's GPU compositor, thresholds both results, and overlays
the combined mask on the shaded image.

![Freestyle and screen-space crop](side-by-side-crop.png)

The screen pass catches most of the prominent boundaries, but its lines are heavier. Narrow
posts, curved tubing, and the dense lower mechanism can become nearly solid black because an
edge occupies a large fraction of their projected width.

This overlay compares pixels changed from the plain shaded render by at least 24 in one sRGB
channel. White pixels changed in both edge renders, magenta pixels changed only under
Freestyle, and cyan pixels changed only under the screen pass.

![Edge coverage differences](edge-difference-crop.png)

| Coverage metric | Pixels |
| --- | ---: |
| Shared | 14,409 |
| Freestyle only | 561 |
| Screen only | 16,474 |
| Freestyle total | 14,970 |
| Screen-space total | 30,883 |

At this threshold, the screen pass overlaps 96.25% of Freestyle's changed pixels but draws
more than twice as many changed pixels overall. Its intersection-over-union with Freestyle is
45.82%. This is a coverage comparison, not a claim that overlapping pixels have the same
geometric meaning.

The prototype uses fixed normalized-depth and normal thresholds of 0.01 and 0.3. Tuning them
can trade missed edges for heavier lines, but cannot make a screen-space discontinuity test
equivalent to Freestyle's topology and visibility classifications.

`tools/compare_edge_renders.py` regenerates the paired views, overlay, and `metrics.json` from
the three source renders.
