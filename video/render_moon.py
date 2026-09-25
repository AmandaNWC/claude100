"""Coastar Therapeutics — Mid-Autumn Festival greeting film (10 s, 16:9).

Procedural render: lipid nanoparticles drift in a midnight-blue microscopic
space, converge along spiral paths into a glowing ring, and resolve into a
full moon framed by a cell-membrane bilayer and a faint molecular network.
No text is rendered; the final frame leaves negative space below the moon
for typography.

The default render is 13 s: the 10 s motion piece plus an end card
("Happy Mid-Autumn Festival" / tagline / company name) that fades in
below the moon. Pass --no-text for the clean 10 s version.

Usage: python3 render_moon.py [out.mp4] [--no-text] [--preview t1,t2,...]
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

W, H = 1920, 1080
FPS = 30
MOTION = 10.0  # length of the motion piece; timings below are keyed to it
WITH_TEXT = '--no-text' not in sys.argv
DUR = 13.0 if WITH_TEXT else MOTION
NF = int(FPS * DUR)
F = np.array([960.0, 450.0])  # moon centre (world == screen at zoom 1)
R = 190.0                     # moon radius in world px
rng = np.random.default_rng(20260925)


def hexc(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


NAVY = hexc('#0B1A2E')
DEEP = hexc('#04070E')
MOON = hexc('#F4F1EA')
MARIA = hexc('#BCC5D0')
GOLD = hexc('#CDB57F')
TEAL = hexc('#5FB3B3')
IVORY = hexc('#F2E8D5')


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smoother(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def fnoise(h, w, octaves, base, persist=0.5, seed=None):
    g = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = base / (2 ** o)
        small = g.random((max(3, int(h / s) + 3), max(3, int(w / s) + 3))).astype(np.float32)
        out += amp * np.asarray(Image.fromarray(small, 'F').resize((w, h), Image.BICUBIC))
        tot += amp
        amp *= persist
    return out / tot


def downsample(arr, size):
    return np.asarray(Image.fromarray(arr.astype(np.float32), 'F').resize(size, Image.LANCZOS))


def upsample(arr3, size):
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(arr3[..., c], dtype=np.float32), 'F')
                                .resize(size, Image.BILINEAR)) for c in range(arr3.shape[2])], -1)


# ---------------------------------------------------------------- camera
def camera(t):
    u = t / MOTION
    z = 1.0 + 0.075 * (0.45 * u + 0.55 * smoother(u))
    pan = (1 - smoother(t / 8.5)) * np.array([-22.0, 12.0])
    return z, pan


# ---------------------------------------------------------------- background
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
_d = np.hypot((XX - F[0]) / W, (YY - F[1]) / W)
BG = (NAVY * 1.15)[None, None] * (1 - smoothstep(0.0, 0.8, _d))[..., None] \
    + DEEP[None, None] * smoothstep(0.0, 0.8, _d)[..., None]
_haze = fnoise(H, W, 4, 520, seed=7)
BG += TEAL[None, None] * (0.030 * smoothstep(0.45, 0.85, _haze))[..., None]
_haze2 = fnoise(H, W, 3, 300, seed=11)
BG += GOLD[None, None] * (0.012 * smoothstep(0.55, 0.9, _haze2))[..., None]
VIGN = 1 - 0.5 * smoothstep(0.45, 1.25, np.hypot((XX - W / 2) / (W / 2), (YY - H / 2) / (H / 2)))


# ---------------------------------------------------------------- moon texture
def make_moon():
    S, r = 1100, 512.0
    c = S / 2
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = (x - c) / r, (y - c) / r
    rr = np.sqrt(dx * dx + dy * dy)
    mu = np.sqrt(np.clip(1 - rr * rr, 0, 1))
    maria = smoothstep(0.50, 0.66, fnoise(S, S, 4, 460, seed=3))
    maria *= smoothstep(1.0, 0.55, np.hypot(dx - 0.1, dy + 0.15))  # keep maria mostly off the limb
    fine = fnoise(S, S, 6, 90, seed=5)
    alb = 0.92 - 0.27 * maria + 0.10 * (fine - 0.5)
    g = np.random.default_rng(21)
    for _ in range(80):
        cr = 5.0 * math.exp(g.random() * 3.2)
        ang, rad = g.random() * 2 * math.pi, math.sqrt(g.random()) * r * 0.97
        cx, cy = c + rad * math.cos(ang), c + rad * math.sin(ang)
        x0, x1 = int(max(cx - 2.2 * cr, 0)), int(min(cx + 2.2 * cr, S))
        y0, y1 = int(max(cy - 2.2 * cr, 0)), int(min(cy + 2.2 * cr, S))
        if x1 <= x0 or y1 <= y0:
            continue
        d = np.hypot(x[y0:y1, x0:x1] - cx, y[y0:y1, x0:x1] - cy) / cr
        floor = -0.03 * np.clip(1 - d, 0, 1) ** 0.5
        rim = 0.022 * np.exp(-((d - 1.0) / 0.3) ** 2)
        alb[y0:y1, x0:x1] += floor + rim
    for _ in range(3):  # young bright-ray craters
        ang, rad = g.random() * 2 * math.pi, math.sqrt(g.random()) * r * 0.8
        cx, cy = c + rad * math.cos(ang), c + rad * math.sin(ang)
        d = np.hypot(x - cx, y - cy)
        th = np.arctan2(y - cy, x - cx)
        rays = np.clip(np.sin(th * 23 + g.random() * 6) * np.sin(th * 9 + 1.3), 0, 1) ** 3
        alb += 0.10 * np.exp(-d / 28) + 0.05 * rays * np.exp(-d / 170)
    shade = 0.58 + 0.42 * mu ** 0.55
    tint = MOON[None, None] * (1 - 0.35 * maria[..., None]) + MARIA[None, None] * 0.35 * maria[..., None]
    rgb = 0.97 * np.clip(alb, 0, 1.1)[..., None] * shade[..., None] * tint
    alpha = np.clip(r - rr * r + 0.5, 0, 1)
    size = (494, 494)  # ~1.15x of the final on-screen size, pre-filtered
    rgbp = [downsample(rgb[..., k] * alpha, size) for k in range(3)]
    return rgbp + [downsample(alpha, size)], 494 / S * r / R


MOON_CH, MOON_S = make_moon()
MOON_IMG = [Image.fromarray(np.ascontiguousarray(a, dtype=np.float32), "F") for a in MOON_CH]


def affine(chans, s_l, c_l, cscreen, z, theta):
    """Map a world-space layer (scale s_l px/world, centre c_l) to the screen."""
    k = s_l / z
    ct, st = math.cos(theta), math.sin(theta)
    a, b, d, e = k * ct, k * st, -k * st, k * ct
    c = c_l[0] - a * cscreen[0] - b * cscreen[1]
    f = c_l[1] - d * cscreen[0] - e * cscreen[1]
    return np.stack([np.asarray(ch.transform((W, H), Image.AFFINE, (a, b, c, d, e, f),
                                             resample=Image.BICUBIC)) for ch in chans], -1)


# ---------------------------------------------------------------- molecular network
LS = 1.1        # layer px per world px
SS = 3          # supersampling while drawing


def make_network():
    g = np.random.default_rng(33)
    pts = []
    while len(pts) < 95:
        r = R * (1.38 + 1.6 * g.random() ** 1.3)
        th = g.random() * 2 * math.pi
        p = np.array([1.5 * r * math.cos(th), r * math.sin(th)])
        if p[1] > 1.05 * R:  # keep the typography zone below the moon clear
            continue
        pts.append(p)
    pts = np.array(pts)
    tree = cKDTree(pts)
    edges = set()
    for i, p in enumerate(pts):
        dists, idx = tree.query(p, 4)
        for dd, j in zip(dists[1:], idx[1:]):
            if dd < 1.05 * R:
                edges.add((min(i, j), max(i, j)))
    edges = sorted(edges)
    hw, hh = 820, 620
    sc = LS * SS
    cw, ch = int(2 * hw * sc), int(2 * hh * sc)
    lines = Image.new('L', (cw, ch))
    nodes = Image.new('L', (cw, ch))
    dl, dn = ImageDraw.Draw(lines), ImageDraw.Draw(nodes)
    to = lambda p: ((p[0] + hw) * sc, (p[1] + hh) * sc)
    for i, j in edges:
        dl.line([to(pts[i]), to(pts[j])], fill=255, width=SS)
    for k, p in enumerate(pts):
        x, y = to(p)
        rad = (1.6 + 1.6 * g.random()) * sc
        dn.ellipse([x - rad, y - rad, x + rad, y + rad], fill=255)
        if k % 11 == 0:  # a few ring "molecules"
            hr = 9 * sc
            poly = [(x + 2.2 * rad + hr * math.cos(a), y + hr * math.sin(a))
                    for a in np.linspace(0, 2 * math.pi, 7) + math.pi / 6]
            dl.line(poly, fill=255, width=SS)
    size = (int(2 * hw * LS), int(2 * hh * LS))
    L = downsample(np.asarray(lines, np.float32) / 255, size)
    N = downsample(np.asarray(nodes, np.float32) / 255, size)
    glow = gaussian_filter(N, 5) * 3.0
    rgb = (L[..., None] * GOLD * 0.20 + N[..., None] * (TEAL * 0.40 + IVORY * 0.18)
           + glow[..., None] * TEAL * 0.16)
    imgs = [Image.fromarray(np.ascontiguousarray(rgb[..., c], dtype=np.float32), 'F') for c in range(3)]
    return imgs, (hw * LS, hh * LS), pts, edges


NET_IMG, NET_C, NET_PTS, NET_EDGES = make_network()


# ---------------------------------------------------------------- membrane bilayer + cells
def make_membrane():
    hw = 1.8 * R
    sc = LS * SS
    cs = int(2 * hw * sc)
    heads = Image.new('L', (cs, cs))
    tails = Image.new('L', (cs, cs))
    dh, dt = ImageDraw.Draw(heads), ImageDraw.Draw(tails)
    c = cs / 2
    mid = 1.105 * R
    for ring_r, sgn in ((1.075 * R, 1), (1.135 * R, -1)):
        n = int(2 * math.pi * ring_r / 6.2)
        for k in range(n):
            a = 2 * math.pi * k / n + (0.5 * 2 * math.pi / n if sgn < 0 else 0)
            hx, hy = c + ring_r * sc * math.cos(a), c + ring_r * sc * math.sin(a)
            rad = 1.45 * sc
            dh.ellipse([hx - rad, hy - rad, hx + rad, hy + rad], fill=255)
            for off in (-0.006, 0.006):
                tx, ty = c + (mid - sgn * 1.0) * sc * math.cos(a + off), c + (mid - sgn * 1.0) * sc * math.sin(a + off)
                dt.line([(hx, hy), (tx, ty)], fill=255, width=max(1, SS // 2 + 1))
    size = (int(2 * hw * LS),) * 2
    Hm = downsample(np.asarray(heads, np.float32) / 255, size)
    Tm = downsample(np.asarray(tails, np.float32) / 255, size)
    n = size[0]
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    rx, ry = (x - n / 2) / LS, (y - n / 2) / LS
    rr = np.hypot(rx, ry)
    th = np.arctan2(ry, rx)
    amod = 0.35 + 0.65 * np.clip(0.5 + 0.28 * np.sin(th + 0.7) + 0.2 * np.sin(2 * th + 2.1)
                                 + 0.12 * np.sin(3 * th + 0.4), 0, 1)
    g = np.random.default_rng(44)
    seeds = []
    while len(seeds) < 420:
        r = R * (1.17 + 0.6 * g.random())
        a = g.random() * 2 * math.pi
        seeds.append((r * math.cos(a), r * math.sin(a)))
    d, _ = cKDTree(np.array(seeds)).query(np.stack([rx.ravel(), ry.ravel()], 1), 2)
    edge = np.exp(-(((d[:, 1] - d[:, 0]).reshape(n, n)) / 1.6) ** 2)
    band = smoothstep(1.17 * R, 1.27 * R, rr) * (1 - smoothstep(1.35 * R, 1.75 * R, rr))
    cells = edge * band
    rgb = (Hm[..., None] * IVORY * 0.34 + Tm[..., None] * GOLD * 0.16) * amod[..., None] \
        + cells[..., None] * TEAL * 0.11 * (0.5 + 0.5 * amod[..., None])
    imgs = [Image.fromarray(np.ascontiguousarray(rgb[..., k], dtype=np.float32), 'F') for k in range(3)]
    return imgs, (n / 2, n / 2)


MEM_IMG, MEM_C = make_membrane()


# ---------------------------------------------------------------- particles
def golden_disc(n, r0, r1):
    i = np.arange(n) + 0.5
    r = np.sqrt(r0 ** 2 + (r1 ** 2 - r0 ** 2) * i / n)
    th = i * math.pi * (3 - math.sqrt(5))
    return r, th


def make_group(n, rim_frac, rim_t0, rim_dur, in_t0, in_dur, conv_frac=1.0):
    p0 = rng.uniform([-0.12 * W, -0.12 * H], [1.12 * W, 1.12 * H], (n, 2))
    ang = rng.uniform(0, 2 * math.pi, n)
    spd = rng.uniform(5, 16, n)
    g = dict(
        p0=p0, v=np.stack([np.cos(ang), np.sin(ang)], 1) * spd[:, None],
        wa=rng.uniform(3, 10, (n, 1)), wf=rng.uniform(0.25, 0.7, n), wp=rng.uniform(0, 6.3, (n, 2)),
        rim=rng.random(n) < rim_frac, conv=rng.random(n) < conv_frac,
        swirl=rng.uniform(0.3, 0.7, n), tw=rng.uniform(0, 6.3, n), twf=rng.uniform(1.0, 2.5, n),
        fj=rng.uniform(-0.4, 0.4, n),
    )
    nr = int(g['rim'].sum())
    rt = np.empty(n)
    tt = np.empty(n)
    rt[g['rim']] = R * rng.uniform(0.94, 1.015, nr)
    tt[g['rim']] = rng.uniform(0, 2 * math.pi, nr)
    r_in, th_in = golden_disc(n - nr, 0.0, 0.93 * R)
    perm = rng.permutation(n - nr)
    rt[~g['rim']] = r_in[perm]
    tt[~g['rim']] = th_in[perm] + rng.uniform(-0.02, 0.02, n - nr)
    g['rt'], g['tt'] = rt, tt
    g['t0'] = np.where(g['rim'], rng.uniform(*rim_t0, n), rng.uniform(*in_t0, n))
    g['dur'] = np.where(g['rim'], rng.uniform(*rim_dur, n), rng.uniform(*in_dur, n))
    pick = rng.random(n)
    col = np.where((pick < 0.72)[:, None], TEAL, np.where((pick < 0.9)[:, None], IVORY, GOLD))
    g['col'] = col * rng.uniform(0.85, 1.1, (n, 1))
    return g


def drift(g, t):
    return (g['p0'] + g['v'] * t
            + g['wa'] * np.stack([np.sin(g['wf'] * t + g['wp'][:, 0]),
                                  np.cos(0.8 * g['wf'] * t + g['wp'][:, 1])], 1))


def positions(g, t):
    """Drift, then spiral in (polar interpolation) towards the target on the moon disc."""
    p = drift(g, t)
    e = smoother((t - g['t0']) / g['dur']) * g['conv']
    t0 = g['t0'][:, None]
    p_t0 = (g['p0'] + g['v'] * t0 + g['wa'] * np.stack(
        [np.sin(g['wf'] * g['t0'] + g['wp'][:, 0]), np.cos(0.8 * g['wf'] * g['t0'] + g['wp'][:, 1])], 1))
    rel, rel0 = p - F, p_t0 - F
    rd = np.hypot(rel[:, 0], rel[:, 1])
    thd, thd0 = np.arctan2(rel[:, 1], rel[:, 0]), np.arctan2(rel0[:, 1], rel0[:, 0])
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    delta = wrap(thd - thd0)
    d0 = wrap(g['tt'] - thd0)
    th = thd0 + delta * (1 - e) + d0 * e + g['swirl'] * np.sin(math.pi * e)
    r = rd + (g['rt'] - rd) * e
    return F + np.stack([r * np.cos(th), r * np.sin(th)], 1), e


GA = make_group(1500, 0.45, (1.4, 2.4), (2.8, 3.4), (1.9, 3.0), (3.0, 3.8))
GA['inten'] = rng.uniform(0.45, 0.9, 1500)
GB = make_group(320, 0.4, (1.3, 2.3), (3.0, 3.6), (1.8, 2.9), (3.2, 3.9), conv_frac=0.55)
GB['inten'] = rng.uniform(0.22, 0.45, 320)
GC = make_group(45, 0.0, (9, 9), (1, 1), (99, 99), (1, 1), conv_frac=0.0)
GC['inten'] = rng.uniform(0.05, 0.11, 45)

NL = 26
LNP = make_group(NL, 1.0, (1.2, 2.2), (3.0, 3.6), (0, 0), (1, 1))
LNP['tt'] = np.linspace(0, 2 * math.pi, NL, endpoint=False) + rng.uniform(-0.1, 0.1, NL)
LNP['rt'] = np.full(NL, R)
LNP['rad'] = rng.uniform(9, 22, NL)
LNP['soft'] = rng.uniform(0.07, 0.16, NL)
LNP['cargo'] = [rng.uniform(-0.5, 0.5, (5, 2)) for _ in range(NL)]
LNP['spin'] = rng.uniform(-0.3, 0.3, NL)

PULSES = [NET_EDGES[i] for i in rng.choice(len(NET_EDGES), 16, replace=False)]
PULSE_PH = rng.uniform(0, 1, 16)
PULSE_SP = rng.uniform(0.25, 0.45, 16)


def splat(xs, ys, vals, sigma, f):
    h, w = int(H * f), int(W * f)
    buf = np.zeros((h, w, 3), np.float32)
    x, y = xs * f - 0.5, ys * f - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    for ox, oy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + ox, y0 + oy
        m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        np.add.at(buf, (yi[m], xi[m]), vals[m] * wt[m, None])
    buf = gaussian_filter(buf, (sigma, sigma, 0), truncate=3.0)
    return buf * (2 * math.pi * sigma * sigma)


def draw_lnp(buf, x, y, rad, soft, cargo, rot, col, a):
    n = int(rad * 1.9) + 3
    xi, yi = int(round(x)), int(round(y))
    if xi + n < 0 or xi - n >= W or yi + n < 0 or yi - n >= H or a <= 0.002:
        return
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
    xx -= x - xi
    yy -= y - yi
    r = np.hypot(xx, yy) / rad
    val = (np.exp(-((r - 1.0) / soft) ** 2) * 0.9
           + np.exp(-((r - 0.84) / (0.7 * soft)) ** 2) * 0.35
           + np.clip(1 - r, 0, 1) ** 0.5 * 0.08
           + np.exp(-((r - 1.0) / 0.45) ** 2) * 0.10)
    cr, sr = math.cos(rot), math.sin(rot)
    for cx, cy in cargo:
        px, py = (cx * cr - cy * sr) * rad, (cx * sr + cy * cr) * rad
        val += 0.45 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * (0.09 * rad) ** 2))
    x0, x1 = max(xi - n, 0), min(xi + n + 1, W)
    y0, y1 = max(yi - n, 0), min(yi + n + 1, H)
    sub = val[y0 - (yi - n):y1 - (yi - n), x0 - (xi - n):x1 - (xi - n)]
    buf[y0:y1, x0:x1] += sub[..., None] * col * a


# ---------------------------------------------------------------- end card typography
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
TS = 2  # supersampling for type


def _font(name, size, var):
    f = ImageFont.truetype(os.path.join(FONT_DIR, name), size * TS)
    f.set_variation_by_name(var)
    return f


def _text_mask(text, fnt, y, tracking=0.0, glyphs=None):
    """Full-frame coverage mask with the line centred horizontally on baseline y."""
    im = Image.new('L', (W * TS, H * TS))
    d = ImageDraw.Draw(im)
    if tracking:
        adv = [fnt.getlength(ch) + tracking * fnt.size for ch in text]
        x = (W * TS - (sum(adv) - tracking * fnt.size)) / 2
        for ch, a in zip(text, adv):
            d.text((x, y * TS), ch, font=fnt, fill=255, anchor='ls')
            x += a
    else:
        d.text((W * TS / 2, y * TS), text, font=fnt, fill=255, anchor='ms')
    return np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255


def _rule_mask(y, half_w):
    im = Image.new('L', (W * TS, H * TS))
    ImageDraw.Draw(im).line([((W / 2 - half_w) * TS, y * TS), ((W / 2 + half_w) * TS, y * TS)],
                            fill=255, width=TS)
    return np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255


def _line(mask, color, opacity, t0, dur=1.5, rise=12.0, glow=0.0):
    ys, xs = np.nonzero(mask > 0.002)
    y0, y1 = max(ys.min() - 24, 0), min(ys.max() + 24, H)
    x0, x1 = max(xs.min() - 24, 0), min(xs.max() + 24, W)
    return dict(img=Image.fromarray(np.ascontiguousarray(mask[y0:y1, x0:x1]), 'F'),
                box=(y0, y1, x0, x1), color=color, opacity=opacity, t0=t0, dur=dur,
                rise=rise, glow=glow)


if WITH_TEXT:
    TEXT_LINES = [
        _line(_text_mask('Happy Mid-Autumn Festival', _font('Cormorant.ttf', 66, 'Light'), 842),
              MOON, 0.96, 8.8, glow=0.18),
        _line(_text_mask('Celebrating connection, collaboration, and shared progress.',
                         _font('CormorantItalic.ttf', 31, 'Light Italic'), 892),
              IVORY * 0.92 + TEAL * 0.08, 0.80, 9.4),
        _line(_rule_mask(928, 34), GOLD, 0.70, 9.9, rise=0.0),
        _line(_text_mask('COASTAR THERAPEUTICS', _font('Montserrat.ttf', 17, 'Regular'), 966, tracking=0.34),
              GOLD * 1.08, 0.95, 10.1),
    ]


def draw_text(img, t):
    for ln in TEXT_LINES:
        e = smoother((t - ln['t0']) / ln['dur'])
        if e <= 0:
            continue
        y0, y1, x0, x1 = ln['box']
        dy = (1 - e) * ln['rise']
        m = np.asarray(ln['img'].transform(ln['img'].size, Image.AFFINE, (1, 0, 0, 0, 1, -dy),
                                           resample=Image.BICUBIC))
        a = np.clip(m, 0, 1) * (e * ln['opacity'])
        sub = img[y0:y1, x0:x1]
        if ln['glow']:
            sub += ln['color'] * (gaussian_filter(a, 7) * ln['glow'])[..., None]
        img[y0:y1, x0:x1] = sub * (1 - a[..., None]) + ln['color'] * a[..., None]
    return img


# ---------------------------------------------------------------- frame
def render(fi):
    t = fi / FPS
    z, pan = camera(t)
    C = F + pan
    intro = smoothstep(0.0, 1.1, t)
    img = BG * (0.35 + 0.65 * intro)

    dist = np.hypot(XX - C[0], YY - C[1])
    rs = R * z
    rn = dist / rs

    # --- moon (revealed from the rim inwards)
    p = 1.35 * smoothstep(5.0, 7.5, t)
    if p > 0:
        m = affine(MOON_IMG, MOON_S, (247, 247), C, z, 0.02 * t / MOTION)
        reveal = np.clip((p - (1 - rn)) / 0.35, 0, 1)
        img = img * (1 - (m[..., 3] * reveal)[..., None]) + m[..., :3] * reveal[..., None]
        front = np.exp(-((p - (1 - rn) - 0.08) / 0.07) ** 2) * (rn < 1) * (1 - smoothstep(7.0, 7.8, t))
        img += IVORY * (0.22 * front)[..., None]

    # --- ring glow + halo
    ringg = math.exp(-((t - 5.5) / 1.1) ** 2)
    moonk = smoothstep(5.0, 7.8, t)
    out = np.clip(rn - 1, 0, None)
    img += IVORY * (0.35 * ringg * np.exp(-((rn - 1) / 0.035) ** 2))[..., None]
    halo = (0.20 * np.exp(-out / 0.10) + 0.07 * np.exp(-out / 0.9)) * (rn >= 0.995) * moonk
    img += (IVORY * 0.6 + GOLD * 0.4) * halo[..., None]

    # --- membrane bilayer (sweeps in clockwise from the top)
    mk = smoothstep(6.3, 8.3, t)
    if mk > 0:
        mem = affine(MEM_IMG, LS, MEM_C, C, z, -0.03 * t / MOTION)
        ang = (np.arctan2(XX - C[0], -(YY - C[1])) % (2 * math.pi))
        sweep = 2 * math.pi * 1.15 * smoother((t - 6.3) / 2.2)
        img += mem * (np.clip((sweep - ang) / 0.9, 0, 1) * mk)[..., None]

    # --- molecular network (grows outward from the moon)
    nk = smoothstep(6.8, 8.9, t)
    theta_net = 0.035 * t / MOTION
    if nk > 0:
        net = affine(NET_IMG, LS, NET_C, C, z, theta_net)
        rv = R * (1.2 + 2.4 * smoother((t - 6.8) / 2.4))
        vis = np.clip((rv - dist / z) / (0.7 * R), 0, 1) * nk
        img += net * vis[..., None]

    # --- particles
    fade_all = 1 - smoothstep(6.0, 8.0, t)
    light = np.zeros((H, W, 3), np.float32)

    for g, sigma, f, zpar in ((GA, 1.0, 1.0, 1.0), (GB, 1.4, 0.5, 1.2)):
        pos, e = positions(g, t)
        sp = C + (z * zpar) * (pos - F) - pan * (zpar - 1)
        tw = 1 + 0.18 * np.sin(g['twf'] * t + g['tw'])
        conv = g['conv']
        fade = np.where(conv, 1 - smoothstep(6.0 + g['fj'], 8.0 + g['fj'], t), 1 - smoothstep(4.5, 6.8, t))
        inten = g['inten'] * tw * (0.7 + 0.5 * e) * fade * intro
        col = g['col'] * (1 - e[:, None]) + IVORY * e[:, None]
        layer = splat(sp[:, 0], sp[:, 1], col * inten[:, None], sigma, f)
        light += layer if f == 1.0 else upsample(layer, (W, H))

    zc = 1 + (z - 1) * 2.5
    pc = C + zc * (drift(GC, t) - F)
    inten = GC['inten'] * (1 - smoothstep(5.0, 7.2, t)) * intro
    light += upsample(splat(pc[:, 0], pc[:, 1], GC['col'] * inten[:, None], 3.5, 0.25), (W, H))

    pos, e = positions(LNP, t)
    sp = C + z * (pos - F)
    for i in range(NL):
        a = (1 - smoothstep(0.65, 1.0, e[i])) * intro * 0.55
        col = TEAL * (1 - e[i]) * 0.8 + IVORY * (0.2 + 0.8 * e[i])
        draw_lnp(light, sp[i, 0], sp[i, 1], LNP['rad'][i] * z * (1 - 0.7 * e[i]), LNP['soft'][i],
                 LNP['cargo'][i], LNP['spin'][i] * t, col, a)

    # signal pulses travelling along network edges
    if nk > 0:
        ct, st = math.cos(theta_net), math.sin(theta_net)
        xs, ys, vals = [], [], []
        for k, (i, j) in enumerate(PULSES):
            u = (PULSE_PH[k] + PULSE_SP[k] * t) % 1.0
            w = NET_PTS[i] * (1 - u) + NET_PTS[j] * u
            wr = np.array([w[0] * ct - w[1] * st, w[0] * st + w[1] * ct])
            s = C + z * wr
            dw = math.hypot(*wr)
            rv = R * (1.2 + 2.4 * smoother((t - 6.8) / 2.4))
            a = nk * min(max((rv - dw) / (0.7 * R), 0), 1) * math.sin(math.pi * u) * 0.9
            xs.append(s[0]); ys.append(s[1]); vals.append((IVORY * 0.5 + GOLD * 0.5) * a)
        light += splat(np.array(xs), np.array(ys), np.array(vals), 1.3, 1.0)

    img = img + light

    # --- bloom
    small = img.reshape(H // 4, 4, W // 4, 4, 3).mean((1, 3))
    bright = np.clip(small - 0.62, 0, None)
    bloom = gaussian_filter(bright, (4, 4, 0)) * 0.55 + gaussian_filter(bright, (16, 16, 0)) * 0.45
    img = img + upsample(bloom, (W, H)) * 0.9

    # --- tone, vignette, grain
    k = 0.78
    over = np.clip(img - k, 0, None)
    img = np.where(img > k, k + (1 - k) * (1 - np.exp(-over / (1 - k))), img)
    img *= VIGN[..., None]
    if WITH_TEXT:
        img = draw_text(img, t)
    grain = np.random.default_rng(fi).normal(0, 0.011, (H, W, 1)).astype(np.float32)
    img = np.clip(img + grain, 0, 1)
    return (img * 255 + 0.5).astype(np.uint8)


def main():
    args = [a for a in sys.argv[1:] if a != '--no-text']
    if args and args[0] == '--preview':
        for t in [float(s) for s in args[1].split(',')]:
            Image.fromarray(render(int(round(t * FPS)))).save(f'preview_{t:04.1f}.png')
        return
    out = args[0] if args else 'coastar_mid_autumn_2026.mp4'
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-vf', 'hqdn3d=1.2:1.2:5:5', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for i, frame in enumerate(pool.imap(render, range(NF), chunksize=2)):
            proc.stdin.write(frame.tobytes())
            if i % 30 == 0:
                print(f'frame {i}/{NF}', flush=True)
    proc.stdin.close()
    proc.wait()
    Image.fromarray(render(NF - 1)).save(out.rsplit('.', 1)[0] + '_final_frame.png')


if __name__ == '__main__':
    main()
