// Forecast a launch day's Twitch viewers, hour by hour, from the first hour.
//
// The shape of a launch day is not flat: the first hour (00:00-01:00 UTC) only
// holds the regions that already unlocked at midnight their time, then the
// United States wakes the curve up in two waves (east coast, then west coast),
// and the evening brings the day's peak. How far the day climbs above hour 1
// depends, above all, on how big hour 1 already is: a launch whose first hour
// is already huge has little headroom left, while a small first hour can grow
// many times over. Across past launches the log of the peak tracks the log of
// hour 1 with a slope near zero, which is why this program never forecasts a
// fixed multiple of hour 1.
//
// Method, per forecast hour h = 2..24:
//   - Take every past launch in the case ("comparables"), read its hourly
//     viewers on its own release day, and express each hour as a log count.
//   - Fit log(viewers at h) against log(hour-1 viewers) by weighted least
//     squares, with a Gaussian kernel on log hour-1 (width 1.5) so launches
//     the size of this one dominate the fit and tiny or gigantic ones fade.
//   - Shift each past launch's hours so its US-east-midnight unlock lands on
//     the same hour as this launch's (daylight saving moves that hour between
//     04:00 and 05:00 UTC). Hour 1 itself is the predictor, never an outcome.
//   - The fitted value is the median; the spread is the weighted residual
//     scatter around the fit, shrunk by 0.9 and floored, turned into the five
//     quantiles with normal z-scores.
//   - If too few past launches cover an hour, fall back to the median
//     log-ratio across launches, and then to a flat forecast.
// A top-heavy live-streams list (one stream holding over half the hour-1
// audience) widens the spread a little: a concentrated audience is a more
// volatile one. Everything here is deterministic: same input, same bytes out.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Normal quantiles for the five scored levels 0.05, 0.25, 0.5, 0.75, 0.95. */
const Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Kernel width, in natural-log viewer units, around this launch's hour 1. */
const KERNEL_WIDTH = 1.5;
/** The residual scatter is shrunk this much before it becomes the spread. */
const SPREAD_SHRINK = 0.9;
/** The spread never goes below this, in log units, nor above this. */
const SPREAD_FLOOR = 0.1;
const SPREAD_CEIL = 0.6;

interface HourPoint {
  from: string;
  viewers: number;
  channels?: number;
}

interface KnownRow {
  hour: number;
  viewers: number;
  channels?: number;
}

interface Comparable {
  day_start: string;
  hourly: HourPoint[];
}

interface LiveStream {
  viewers?: number;
}

interface CaseFile {
  id: string;
  day_start: string;
  known: KnownRow[];
  forecast: Array<{ hour: number }>;
  history?: HourPoint[];
  history_hourly?: HourPoint[];
  streams?: LiveStream[] | null;
  comparables: Comparable[];
}

/** The middle value of a list (upper median for even counts). */
function medianOf(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/** Which hour of the release day (1..24) a timestamp falls in, or -1. */
function hourInDay(from: string, dayStart: string): number {
  const t = Date.parse(from);
  const start = Date.parse(dayStart);
  if (!Number.isFinite(t) || !Number.isFinite(start)) return -1;
  return Math.floor((t - start) / 3600000) + 1;
}

/**
 * The release-day hour when midnight falls on the US east coast.
 * Eastern daylight time puts it at 04:00 UTC (hour 5); eastern standard time
 * at 05:00 UTC (hour 6). US daylight saving runs from the second Sunday of
 * March to the first Sunday of November.
 */
function eastUnlockHour(dayStart: string): number {
  const d = new Date(dayStart);
  const year = d.getUTCFullYear();
  const month = d.getUTCMonth() + 1;
  const day = d.getUTCDate();
  const marchWeekday = new Date(Date.UTC(year, 2, 1)).getUTCDay();
  const dstStarts = 1 + ((7 - marchWeekday) % 7) + 7;
  const novemberWeekday = new Date(Date.UTC(year, 10, 1)).getUTCDay();
  const dstEnds = 1 + ((7 - novemberWeekday) % 7);
  const inDst =
    (month > 3 || (month === 3 && day >= dstStarts)) &&
    (month < 11 || (month === 11 && day < dstEnds));
  return inDst ? 5 : 6;
}

interface FitPoint {
  x: number;
  y: number;
  w: number;
}

/** Weighted straight-line fit of y on x, plus the weighted residual scatter. */
function weightedLine(points: FitPoint[]): { intercept: number; slope: number; scatter: number } {
  let weightSum = 0;
  let xSum = 0;
  let ySum = 0;
  for (const p of points) {
    weightSum += p.w;
    xSum += p.w * p.x;
    ySum += p.w * p.y;
  }
  const xMean = xSum / weightSum;
  const yMean = ySum / weightSum;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    sxx += p.w * (p.x - xMean) * (p.x - xMean);
    sxy += p.w * (p.x - xMean) * (p.y - yMean);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  const intercept = yMean - slope * xMean;
  let squared = 0;
  for (const p of points) {
    const miss = p.y - (intercept + slope * p.x);
    squared += p.w * miss * miss;
  }
  const scatter = Math.sqrt(squared / Math.max(1, weightSum - 2));
  return { intercept, slope, scatter };
}

/** A past launch reduced to its release-day hourly counts and unlock hour. */
interface PastLaunch {
  firstHour: number;
  unlock: number;
  hours: Map<number, number>;
}

function pastLaunches(c: CaseFile): PastLaunch[] {
  const out: PastLaunch[] = [];
  for (const comp of c.comparables ?? []) {
    const totals = new Map<number, number>();
    const counts = new Map<number, number>();
    for (const p of comp.hourly ?? []) {
      const h = hourInDay(p.from, comp.day_start);
      if (h < 1 || h > 24 || !(p.viewers > 0)) continue;
      totals.set(h, (totals.get(h) ?? 0) + p.viewers);
      counts.set(h, (counts.get(h) ?? 0) + 1);
    }
    const hours = new Map<number, number>();
    for (const [h, total] of totals) hours.set(h, total / (counts.get(h) ?? 1));
    const firstHour = hours.get(1);
    if (!firstHour || firstHour <= 0) continue;
    out.push({ firstHour, unlock: eastUnlockHour(comp.day_start), hours });
  }
  return out;
}

/** The five quantiles around a log-median with a log spread, rounded. */
function quantilesAround(logMedian: number, spread: number): number[] {
  const q = Z_SCORES.map((z) => {
    const value = Math.exp(logMedian + z * spread);
    return Math.min(1e9, Math.max(0.01, Math.round(value * 100) / 100));
  });
  for (let i = 1; i < q.length; i++) q[i] = Math.max(q[i], q[i - 1]);
  return q;
}

/**
 * Which hour of a past launch's own day lines up with one of this launch's
 * hours, once both are measured from their US-east-midnight unlock. Hour 1
 * is the predictor and is never an outcome, so a mapping that lands on it
 * yields nothing.
 */
function alignedHour(hour: number, targetUnlock: number, launchUnlock: number): number {
  return hour - (targetUnlock - launchUnlock);
}

function forecastCase(c: CaseFile): { hours: Array<{ hour: number; q: number[] }> } {
  const first = (c.known ?? []).find((k) => k.hour === 1);
  let firstHour = first && first.viewers > 0 ? first.viewers : NaN;
  if (!Number.isFinite(firstHour)) {
    const recent = (c.history_hourly ?? c.history ?? [])
      .filter((p) => p.viewers > 0)
      .map((p) => p.viewers);
    firstHour = recent.length > 0 ? medianOf(recent.slice(-48)) : 1000;
  }
  firstHour = Math.max(1, firstHour);
  const logFirst = Math.log(firstHour);
  const targetUnlock = eastUnlockHour(c.day_start);
  const launches = pastLaunches(c);

  // A single stream holding most of hour 1 makes the day less predictable.
  let concentration = 1;
  if (Array.isArray(c.streams) && c.streams.length > 0) {
    const counts = c.streams.map((s) => s.viewers ?? 0).filter((v) => v > 0);
    const total = counts.reduce((a, b) => a + b, 0);
    if (total > 0 && Math.max(...counts) / total > 0.5) concentration = 1.1;
  }

  const wanted = (c.forecast ?? []).map((f) => f.hour).filter((h) => h >= 2 && h <= 24);
  const hoursToDo = wanted.length > 0 ? wanted : Array.from({ length: 23 }, (_, i) => i + 2);
  const rows: Array<{ hour: number; q: number[] }> = [];
  for (const hour of hoursToDo) {
    const points: FitPoint[] = [];
    for (const launch of launches) {
      const shifted = alignedHour(hour, targetUnlock, launch.unlock);
      if (shifted === 1) continue;
      const viewers = launch.hours.get(shifted);
      if (!viewers || viewers <= 0) continue;
      const distance = (Math.log(launch.firstHour) - logFirst) / KERNEL_WIDTH;
      points.push({
        x: Math.log(launch.firstHour),
        y: Math.log(viewers),
        w: Math.exp(-0.5 * distance * distance),
      });
    }
    const weightTotal = points.reduce((a, p) => a + p.w, 0);
    let logMedian: number;
    let spread: number;
    if (points.length >= 8 && weightTotal >= 3) {
      const fit = weightedLine(points);
      logMedian = fit.intercept + fit.slope * logFirst;
      spread = Math.min(SPREAD_CEIL, Math.max(SPREAD_FLOOR, SPREAD_SHRINK * fit.scatter));
    } else {
      const ratios: number[] = [];
      for (const launch of launches) {
        const shifted = alignedHour(hour, targetUnlock, launch.unlock);
        if (shifted === 1) continue;
        const viewers = launch.hours.get(shifted);
        if (viewers && viewers > 0) ratios.push(Math.log(viewers / launch.firstHour));
      }
      if (ratios.length >= 5) {
        ratios.sort((a, b) => a - b);
        const middle = ratios[Math.floor(ratios.length / 2)];
        const lower = ratios[Math.floor(ratios.length / 4)];
        const upper = ratios[Math.floor((3 * ratios.length) / 4)];
        logMedian = logFirst + middle;
        spread = Math.min(SPREAD_CEIL, Math.max(0.15, (SPREAD_SHRINK * (upper - lower)) / 1.349));
      } else {
        logMedian = logFirst;
        spread = 0.5;
      }
    }
    rows.push({ hour, q: quantilesAround(logMedian, spread * concentration) });
  }
  rows.sort((a, b) => a.hour - b.hour);
  return { hours: rows };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const result = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(result)}\n`);
}
