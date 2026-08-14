# Query-Adaptive Hierarchical Stage 1

This experiment keeps the **same Neural-Preset data construction and the same primary objective** as
`experiments/ariadnet_lut_neural_preset_setting`, but replaces the dense `8^3 -> 16^3 -> 32^3`
high-channel LUT decoder.

## Core representation

For the image that will actually query a LUT, compute the trilinear query mass

\[
Q_I(v)=\frac{1}{HW}\sum_p w_{p,v}.
\]

The high-resolution active support is the smallest set covering a chosen mass plus a one-cell safety shell.

The LUT is represented as

\[
L^{32}=\operatorname{Up}(L^8)+\Delta L^{16}_{\mathcal A_{16}}+\Delta L^{32}_{\mathcal A_{32}}.
\]

`L^8` is dense and global. The two high-resolution terms are **RGB color-displacement tokens only on
queried vertices**. Unqueried high-resolution locations have no free fine parameters for that image and fall
back to the smooth coarse mapping.

## Normalization

For input grade `A`:

\[
A \xrightarrow{Q_A}\ L_N^A \rightarrow Z_A.
\]

The raw/LUT-graded input itself decides where the normalization transform receives fine capacity.

## Styling

For `A -> B`, style `B` is applied to `Z_A`. Therefore:

- `Q_{Z_A}` decides **where to compute** fine style displacement;
- `Q_{Z_B}` is an optional **style-evidence** feature;
- the dense `8^3` coarse style prior depends on the style code only.

This is intentionally different from choosing active cells from the style image itself: the LUT is queried by
`Z_A`, not by `B`.

## Compact color change

The sparse decoder predicts **residual color displacement**, not an independent full LUT. Each active token sees:

- RGB lattice coordinate;
- query mass and local query density;
- coarse mapped RGB;
- coarse displacement `L_base(v)-v`;
- global condition code;
- query distribution summary;
- for the style branch, reference evidence mass/summary.

This makes high-frequency model capacity follow the actual color transport support rather than the empty cube.

## Controlled comparison

The original primary loss is retained exactly:

\[
L_{NP}=L_1(A\to B,B)+L_1(B\to A,A)+10\,MSE(Z_A,Z_B).
\]

Only small query-local regularizers are added for sparse displacement smoothness and shell tapering.

The first scientific comparison should be:

1. existing dense Stage 1;
2. query-adaptive Stage 1 trained from scratch under identical data/optimizer schedule;
3. optionally query-adaptive Stage 1 initialized with compatible dense encoder + 8^3 weights.

Report PSNR / canonical PSNR / hard arbitrary-pair visualization together with parameters, wall-clock latency,
peak VRAM, and active 16/32 fractions. Do not equate active fraction with measured speedup; `benchmark.py`
reports both separately.
