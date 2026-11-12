// photo-geolocator-01 entry: a vision-model-first agent with a pixel-heuristic
// fallback, so it answers every photo whether or not a model key was pushed.
//
// For each of the instance's 25 photos it asks claude-sonnet-5-5, once, where
// the photo was taken, and reads the latitude and longitude out of the answer.
// A photo whose model call is refused (no provider key pushed: HTTP 401),
// whose answer holds no coordinates, or whose call fails, is placed by the
// fallback instead: a baseline-JPEG DC decoder reads the photo's coarse
// colour map (no library, the sandbox has only the standard library), a
// handful of features are measured from it (vegetation, dry ground, sky,
// snow, road, brightness), and the nearest of 14 region centroids -- fitted
// on the kit's public practice photos, as aggregate statistics, never a
// table of photos and their places -- names the anchor it is posted at.
// Every photo gets an answer either way: an unanswered photo counts the
// maximum error, so guessing is always better than leaving it blank.
//
// Deterministic given the model's answers: every request is a function of
// the photo's bytes and the fixed prompt, the calls run four at a time and
// their answers are posted in photo order, so a replay sends the same calls
// and leaves the same answers. The fallback is a pure function of the photo
// bytes: no clock, no unseeded randomness anywhere.

const MODEL = 'claude-sonnet-5-5';
const MAX_TOKENS = 1024;
const CONCURRENCY = 4;
const PROMPT =
  'Where was this photo taken? Look at the landscape, vegetation, road, buildings, signs and vehicles, and give your best estimate of the latitude and longitude, as one line of JSON: {"lat": <degrees north>, "lon": <degrees east>}. If you are unsure, guess anyway: never leave it blank.';
/** Retried with a fixed backoff: a provider's transient errors. A gateway refusal (4xx) is final. */
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 3;

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

let envUrl = '';
let anthropicUrl = '';
let apiKey = '';

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** The model's answer to one photo, as text; null when the gateway refused the call (final) or no key is set. */
async function ask(jpeg: Buffer): Promise<string | null> {
  if (anthropicUrl === '' || apiKey === '') return null;
  const body = JSON.stringify({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') } },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    const res = await fetch(`${anthropicUrl}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
      body,
    });
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      return (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
    }
    const text = await res.text();
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`geo: model call refused (HTTP ${res.status}): ${text.slice(0, 200)}\n`);
    return null;
  }
}

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** The latitude and longitude an answer gives: its first JSON object with lat and lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: the next one, or the numbers below.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

// ---------- Baseline JPEG, DC coefficients only ----------
// A baseline (SOF0) JPEG's scan is entropy-coded 8x8 blocks. The DC
// coefficient of a block is 16x its mean level-shifted sample value, so
// decoding just the DC term of every block gives a coarse colour map of
// the photo -- one RGB value per 8x8 block, about 114x64 for a 512-high
// photo -- which is all the fallback's features need. AC coefficients are
// decoded only far enough to skip them (their run/size symbols). No
// library: the sandbox has only the standard library.

interface HuffTable {
  /** counts[i] = how many codes of length i+1; symbols in code order. */
  counts: number[];
  symbols: number[];
}

interface JpegComponent {
  id: number;
  h: number;
  v: number;
  qtable: number;
  dcTable: number;
  acTable: number;
}

interface BlockGrid {
  cols: number;
  rows: number;
  /** RGB per block, row-major, values 0..255. */
  rgb: Float32Array;
}

function buildHuff(counts: number[], symbols: number[]): HuffTable {
  return { counts, symbols };
}

/** Decode one Huffman symbol from the bit reader; -1 on end of data. */
function huffDecode(br: BitReader, t: HuffTable): number {
  let code = 0;
  for (let len = 1; len <= 16; len += 1) {
    const bit = br.bit();
    if (bit < 0) return -1;
    code = (code << 1) | bit;
    let first = 0;
    let idx = 0;
    for (let l = 1; l < len; l += 1) {
      first = (first + t.counts[l - 1]!) << 1;
      idx += t.counts[l - 1]!;
    }
    const count = t.counts[len - 1]!;
    if (code - first < count) return t.symbols[idx + (code - first)]!;
  }
  return -1;
}

class BitReader {
  private data: Buffer;
  private pos: number;
  private bitBuf = 0;
  private bitCnt = 0;

  constructor(data: Buffer, start: number) {
    this.data = data;
    this.pos = start;
  }

  /** Next bit, 0/1, or -1 at end of scan. Handles 0xFF00 stuffing; a
   * restart marker is consumed only by restart(), at an MCU boundary. */
  bit(): number {
    if (this.bitCnt === 0) {
      if (this.pos >= this.data.length) return -1;
      const b = this.data[this.pos]!;
      if (b === 0xff) {
        const m = this.data[this.pos + 1];
        if (m === 0x00) {
          this.pos += 2;
        } else {
          return -1; // RSTn (handled by restart()), EOI, or another marker.
        }
      } else {
        this.pos += 1;
      }
      this.bitBuf = b;
      this.bitCnt = 8;
    }
    this.bitCnt -= 1;
    return (this.bitBuf >> this.bitCnt) & 1;
  }

  /** At an MCU restart boundary: drop padding bits and consume the RSTn marker, if present. */
  restart(): void {
    this.bitCnt = 0;
    if (this.data[this.pos] === 0xff) {
      const m = this.data[this.pos + 1];
      if (m !== undefined && m >= 0xd0 && m <= 0xd7) this.pos += 2;
    }
  }

  /** Read n bits as an unsigned integer; -1 if the scan ends first. */
  bits(n: number): number {
    let v = 0;
    for (let i = 0; i < n; i += 1) {
      const b = this.bit();
      if (b < 0) return -1;
      v = (v << 1) | b;
    }
    return v;
  }

  /** JPEG value extension for a Huffman-coded difference of `size` bits. */
  receiveExtend(size: number): number {
    if (size === 0) return 0;
    const v = this.bits(size);
    if (v < 0) return 0;
    if (v < 1 << (size - 1)) return v - ((1 << size) - 1);
    return v;
  }
}

/** Decode a baseline JPEG's DC coefficients into a coarse RGB block grid. Null when the file is not a baseline JPEG this reader handles. */
function decodeJpegDc(buf: Buffer): BlockGrid | null {
  if (buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8) return null;
  const quant: number[][] = [];
  const dcTables: (HuffTable | null)[] = [null, null, null, null];
  const acTables: (HuffTable | null)[] = [null, null, null, null];
  let width = 0;
  let height = 0;
  let comps: JpegComponent[] = [];
  let restartInterval = 0;
  let pos = 2;
  let scanStart = -1;
  let scanComps: { id: number; dc: number; ac: number }[] = [];

  while (pos + 3 < buf.length) {
    if (buf[pos] !== 0xff) {
      pos += 1;
      continue;
    }
    const marker = buf[pos + 1]!;
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      pos += 2;
      continue;
    }
    if (marker === 0xd9) break;
    const len = (buf[pos + 2]! << 8) | buf[pos + 3]!;
    const seg = pos + 4;
    if (marker === 0xdb) {
      // DQT: one or more tables; precision bit 4 of the first byte.
      let p = seg;
      const end = seg + len - 2;
      while (p < end) {
        const pq = buf[p]! >> 4;
        const tq = buf[p]! & 15;
        p += 1;
        const table: number[] = [];
        for (let i = 0; i < 64; i += 1) {
          if (pq === 0) {
            table.push(buf[p]!);
            p += 1;
          } else {
            table.push((buf[p]! << 8) | buf[p + 1]!);
            p += 2;
          }
        }
        quant[tq] = table;
      }
    } else if (marker === 0xc0) {
      // SOF0: precision, height, width, component count.
      height = (buf[seg + 1]! << 8) | buf[seg + 2]!;
      width = (buf[seg + 3]! << 8) | buf[seg + 4]!;
      const n = buf[seg + 5]!;
      comps = [];
      for (let i = 0; i < n; i += 1) {
        const o = seg + 6 + i * 3;
        comps.push({ id: buf[o]!, h: buf[o + 1]! >> 4, v: buf[o + 1]! & 15, qtable: buf[o + 2]!, dcTable: 0, acTable: 0 });
      }
    } else if (marker === 0xc2) {
      return null; // Progressive: not handled; the caller falls back.
    } else if (marker === 0xc4) {
      // DHT: one or more tables.
      let p = seg;
      const end = seg + len - 2;
      while (p < end) {
        const tc = buf[p]! >> 4;
        const th = buf[p]! & 15;
        p += 1;
        const counts: number[] = [];
        let total = 0;
        for (let i = 0; i < 16; i += 1) {
          counts.push(buf[p + i]!);
          total += buf[p + i]!;
        }
        p += 16;
        const symbols: number[] = [];
        for (let i = 0; i < total; i += 1) symbols.push(buf[p + i]!);
        p += total;
        const table = buildHuff(counts, symbols);
        if (tc === 0) dcTables[th] = table;
        else acTables[th] = table;
      }
    } else if (marker === 0xdd) {
      restartInterval = (buf[seg]! << 8) | buf[seg + 1]!;
    } else if (marker === 0xda) {
      // SOS: component count, then per component id + table ids.
      const n = buf[seg]!;
      scanComps = [];
      for (let i = 0; i < n; i += 1) {
        const o = seg + 1 + i * 2;
        scanComps.push({ id: buf[o]!, dc: buf[o + 1]! >> 4, ac: buf[o + 1]! & 15 });
      }
      scanStart = seg + len - 2;
      break;
    }
    pos += 2 + len;
  }

  if (scanStart < 0 || comps.length === 0 || width === 0 || height === 0) return null;
  for (const c of comps) {
    const s = scanComps.find((x) => x.id === c.id);
    if (!s) return null;
    c.dcTable = s.dc;
    c.acTable = s.ac;
    if (!dcTables[c.dcTable] || !acTables[c.acTable] || !quant[c.qtable]) return null;
  }

  const hMax = Math.max(...comps.map((c) => c.h));
  const vMax = Math.max(...comps.map((c) => c.v));
  const mcuCols = Math.ceil(width / (8 * hMax));
  const mcuRows = Math.ceil(height / (8 * vMax));
  // DC value per block, per component, in raster order over the MCU grid.
  const dcGrids: Float32Array[] = comps.map((c) => new Float32Array(mcuCols * c.h * mcuRows * c.v));
  const br = new BitReader(buf, scanStart);
  const dcPred = new Array<number>(comps.length).fill(0);
  let mcusSinceRestart = 0;
  let completedMcus = 0;
  let failed = false;

  outer: for (let my = 0; my < mcuRows; my += 1) {
    for (let mx = 0; mx < mcuCols; mx += 1) {
      if (restartInterval > 0 && mcusSinceRestart === restartInterval) {
        br.restart();
        dcPred.fill(0);
        mcusSinceRestart = 0;
      }
      for (let ci = 0; ci < comps.length; ci += 1) {
        const c = comps[ci]!;
        const dcT = dcTables[c.dcTable]!;
        const acT = acTables[c.acTable]!;
        const q0 = quant[c.qtable]![0]!;
        for (let by = 0; by < c.v; by += 1) {
          for (let bx = 0; bx < c.h; bx += 1) {
            const cat = huffDecode(br, dcT);
            if (cat < 0) {
              failed = true;
              break outer;
            }
            const diff = br.receiveExtend(cat);
            dcPred[ci] = dcPred[ci]! + diff;
            const blockIdx = (my * c.v + by) * (mcuCols * c.h) + mx * c.h + bx;
            dcGrids[ci]![blockIdx] = dcPred[ci]! * q0;
            // Skip the AC coefficients: decode their symbols, discard values.
            for (;;) {
              const sym = huffDecode(br, acT);
              if (sym < 0) {
                failed = true;
                break outer;
              }
              const run = sym >> 4;
              const size = sym & 15;
              if (size === 0) {
                if (run === 15) continue; // ZRL: 16 zeroes, keep going.
                break; // EOB: end of this block.
              }
              if (br.bits(size) < 0) {
                failed = true;
                break outer;
              }
            }
          }
        }
      }
      mcusSinceRestart += 1;
      completedMcus += 1;
    }
  }

  // A photo whose scan ends early (a truncated file: the last MCUs never
  // arrive) still yields a usable grid from the blocks that did decode;
  // unwritten blocks stay at DC 0, neutral grey. Refuse only a grid that
  // is mostly missing.
  if (failed && completedMcus < (mcuCols * mcuRows) / 2) return null;

  // Build the luma-resolution RGB grid. Y is component 0; chroma is
  // upsampled from the block covering each luma block. Grayscale: Cb=Cr=128.
  const yComp = comps[0]!;
  const cols = mcuCols * yComp.h;
  const rows = mcuRows * yComp.v;
  const rgb = new Float32Array(cols * rows * 3);
  const cbComp = comps.length > 1 ? comps[1]! : null;
  const crComp = comps.length > 2 ? comps[2]! : null;
  for (let by = 0; by < rows; by += 1) {
    for (let bx = 0; bx < cols; bx += 1) {
      const y = 128 + dcGrids[0]![by * cols + bx]! / 8;
      let cb = 128;
      let cr = 128;
      if (cbComp) {
        const cc = Math.min(mcuCols * cbComp.h - 1, Math.floor((bx * cbComp.h) / yComp.h));
        const rr = Math.min(mcuRows * cbComp.v - 1, Math.floor((by * cbComp.v) / yComp.v));
        cb = 128 + dcGrids[1]![rr * (mcuCols * cbComp.h) + cc]! / 8;
      }
      if (crComp) {
        const cc = Math.min(mcuCols * crComp.h - 1, Math.floor((bx * crComp.h) / yComp.h));
        const rr = Math.min(mcuRows * crComp.v - 1, Math.floor((by * crComp.v) / yComp.v));
        cr = 128 + dcGrids[2]![rr * (mcuCols * crComp.h) + cc]! / 8;
      }
      const o = (by * cols + bx) * 3;
      rgb[o] = y + 1.402 * (cr - 128);
      rgb[o + 1] = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128);
      rgb[o + 2] = y + 1.772 * (cb - 128);
    }
  }
  return { cols, rows, rgb };
}

// ---------- Features from the block grid ----------
// Eight continuous indices in 0..1, measured from the coarse RGB map.
// A DC block is an 8x8 average, so colours are muted: hard per-block
// classes catch almost nothing, but the continuous colour differences
// survive averaging. Sky is taken as the bluer of the top and bottom
// bands: some photos are upside down, so neither is assumed to be up.

export interface Features {
  green: number;
  blue: number;
  bright: number;
  sat: number;
  sky: number;
  vegLow: number;
  dry: number;
  botBright: number;
}

export function extractFeatures(g: BlockGrid): Features {
  const { cols, rows, rgb } = g;
  const n = cols * rows;
  let sumGR = 0;
  let sumBR = 0;
  let sumBright = 0;
  let sumSat = 0;
  let topBR = 0;
  let botBR = 0;
  let lowGR = 0;
  let midRB = 0;
  let botBright = 0;
  const topRows = Math.max(1, Math.floor(rows / 3));
  const lowStart = Math.floor(rows / 3);
  const midStart = Math.floor(rows / 3);
  const midEnd = Math.floor((2 * rows) / 3);
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const o = (y * cols + x) * 3;
      const r = Math.max(0, Math.min(255, rgb[o]!));
      const gg = Math.max(0, Math.min(255, rgb[o + 1]!));
      const b = Math.max(0, Math.min(255, rgb[o + 2]!));
      const mean = (r + gg + b) / 3;
      sumGR += gg - r;
      sumBR += b - r;
      sumBright += mean;
      sumSat += Math.max(r, gg, b) - Math.min(r, gg, b);
      if (y < topRows) topBR += b - r;
      else if (y >= rows - topRows) botBR += b - r;
      if (y >= lowStart) lowGR += gg - r;
      if (y >= midStart && y < midEnd) midRB += r - b;
      if (y >= rows - topRows) botBright += mean;
    }
  }
  const band = topRows * cols;
  const lowN = (rows - lowStart) * cols;
  const midN = (midEnd - midStart) * cols;
  const clamp01 = (v: number): number => Math.max(0, Math.min(1, v));
  return {
    green: clamp01(0.5 + sumGR / n / 40),
    blue: clamp01(0.5 + sumBR / n / 40),
    bright: sumBright / n / 255,
    sat: clamp01(sumSat / n / 128),
    sky: clamp01(0.5 + Math.max(topBR, botBR) / band / 40),
    vegLow: clamp01(0.5 + lowGR / lowN / 40),
    dry: clamp01(0.5 + midRB / midN / 40),
    botBright: botBright / band / 255,
  };
}

const FEATURE_KEYS: (keyof Features)[] = ['green', 'blue', 'bright', 'sat', 'sky', 'vegLow', 'dry', 'botBright'];
// The indices are already scaled to comparable 0..1 ranges.
// Feature weights by between-region / within-region variance (F-ratio,
// normalised), fitted on the 75 practice photos: sky is the most
// discriminative, dry/botBright the least. Equal weights scored the same
// within noise on the old centroids; with the corrected centroids the
// F-weights win the proper LOO (6,958 vs 6,986 km).
const FEATURE_WEIGHTS = [0.875464, 0.870493, 0.860609, 0.891708, 1.0, 0.700344, 0.595236, 0.481127];

// ---------- Regions ----------
// Anchor points (population centres) and the feature centroid of the
// practice photos that fall in each region. The centroids are aggregate
// statistics fitted on the kit's public practice photos: no photo id, no
// per-photo place, nothing that looks a photo up -- just what each part
// of the world tends to look like in these coarse colours.
interface Region {
  name: string;
  lat: number;
  lon: number;
  centroid: number[];
}

export const REGIONS: Region[] = [
  { name: 'US West', lat: 39.0, lon: -105.0, centroid: [0.854430, 0.840480, 0.410295, 0.285240, 0.936214, 0.698066, 0.246036, 0.327874] },
  { name: 'US East', lat: 42.0, lon: -78.0, centroid: [0.890921, 0.892579, 0.524723, 0.340280, 1.000000, 0.810360, 0.181275, 0.590079] },
  { name: 'Canada North', lat: 55.0, lon: -105.0, centroid: [0.912078, 0.961750, 0.568183, 0.486519, 1.000000, 0.848607, 0.115510, 0.402085] },
  { name: 'Mexico', lat: 21.0, lon: -100.0, centroid: [0.742848, 0.798597, 0.554387, 0.159036, 0.881562, 0.650419, 0.190307, 0.413421] },
  { name: 'Brazil', lat: -10.0, lon: -55.0, centroid: [0.540909, 0.483746, 0.445079, 0.274889, 0.727763, 0.484531, 0.617446, 0.306849] },
  { name: 'Southern Cone', lat: -34.0, lon: -64.0, centroid: [0.732575, 0.621214, 0.547838, 0.464262, 0.911634, 0.667435, 0.391703, 0.532473] },
  { name: 'Western Europe', lat: 49.0, lon: 4.0, centroid: [1.000000, 1.000000, 0.442630, 0.371373, 1.000000, 0.839246, 0.205933, 0.417696] },
  { name: 'Scandinavia', lat: 62.0, lon: 14.0, centroid: [0.891677, 0.872302, 0.503757, 0.375326, 0.896967, 0.893607, 0.173461, 0.408742] },
  { name: 'Eastern Europe', lat: 52.0, lon: 28.0, centroid: [0.671374, 0.761352, 0.546542, 0.183994, 0.922189, 0.570908, 0.387142, 0.479225] },
  { name: 'Russia East', lat: 55.0, lon: 60.0, centroid: [0.720218, 0.644025, 0.422601, 0.217618, 0.838022, 0.611196, 0.396508, 0.259548] },
  { name: 'Middle East', lat: 27.0, lon: 45.0, centroid: [0.543478, 0.636448, 0.553551, 0.292535, 0.937363, 0.435113, 0.431137, 0.434373] },
  { name: 'Sub-Saharan Africa', lat: -1.0, lon: 28.0, centroid: [0.913771, 0.875000, 0.487240, 0.445638, 0.929786, 0.840119, 0.175557, 0.414987] },
  { name: 'South Asia', lat: 24.0, lon: 80.0, centroid: [0.766585, 0.776782, 0.443390, 0.149396, 0.941230, 0.720629, 0.233415, 0.379271] },
  { name: 'East Asia', lat: 34.0, lon: 128.0, centroid: [0.624448, 0.432301, 0.429356, 0.147807, 0.564206, 0.575352, 0.596644, 0.416377] },
  { name: 'SE Asia', lat: 12.0, lon: 103.0, centroid: [0.722223, 0.674390, 0.422961, 0.227944, 0.775223, 0.604402, 0.384879, 0.349992] },
  { name: 'Oceania', lat: -27.0, lon: 140.0, centroid: [0.855739, 0.884916, 0.399900, 0.314703, 1.000000, 0.715659, 0.419287, 0.298703] },
];

/** The region guess for a photo's features: the inverse-distance-weighted
 * mean of the three nearest region anchors, blended 60% toward the
 * practice photos' centre of mass (55, 30). The features are noisy -- a
 * hard nearest-centroid pick is worse than a fixed point on the practice
 * photos -- so the guess stays soft and central unless the evidence is
 * strong. Constants tuned by proper leave-one-out on the 75 practice
 * photos (LOO mean 6,986 km vs 7,394 km for the old 50%/(28,60)/eps 0.02
 * constants). A photo left unanswered would count the maximum error, so
 * the fallback always answers. */
export function guessFromFeatures(f: Features): { lat: number; lon: number; region: string } {
  const v = FEATURE_KEYS.map((k) => f[k]);
  const scored = REGIONS.map((r) => {
    let d = 0;
    for (let i = 0; i < v.length; i += 1) {
      const diff = v[i]! - r.centroid[i]!;
      d += FEATURE_WEIGHTS[i]! * diff * diff;
    }
    return { r, d };
  }).sort((a, b) => a.d - b.d);
  const top = scored.slice(0, 3);
  let wsum = 0;
  let lat = 0;
  let lon = 0;
  for (const t of top) {
    const w = 1 / (t.d + 0.1);
    wsum += w;
    lat += w * t.r.lat;
    lon += w * t.r.lon;
  }
  lat = (lat / wsum) * 0.4 + BLIND_LAT * 0.6;
  lon = (lon / wsum) * 0.4 + BLIND_LON * 0.6;
  return { lat, lon, region: top[0]!.r.name };
}

/** The fallback guess for one photo's JPEG bytes; null when the bytes do not decode. */
export function guessFromJpeg(jpeg: Buffer): { lat: number; lon: number; region: string } | null {
  const grid = decodeJpegDc(jpeg);
  if (grid === null) return null;
  return guessFromFeatures(extractFeatures(grid));
}

// ---------- Main ----------
// Blind default when a photo will not decode at all: the practice photos'
// centre of mass, which beats any unanswered photo by ~10,000 km.
const BLIND_LAT = 55.0;
const BLIND_LON = 30.0;

async function main(): Promise<number> {
  envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
  anthropicUrl = (process.env.ANTHROPIC_BASE_URL ?? '').replace(/\/+$/, '');
  apiKey = process.env.ANTHROPIC_API_KEY ?? '';
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);
  let refused = 0;
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      const jpeg = Buffer.from(await (await fetch(`${envUrl}${p.url}`)).arrayBuffer());
      const text = await ask(jpeg);
      const c = text === null ? null : coordinates(text);
      if (c !== null) {
        guesses[i] = { id: p.id, lat: c.lat, lon: c.lon };
        continue;
      }
      if (text === null) refused += 1;
      const g = guessFromJpeg(jpeg);
      if (g !== null) {
        process.stderr.write(`geo: ${p.id} fell back to pixels (${g.region})\n`);
        guesses[i] = { id: p.id, lat: g.lat, lon: g.lon };
      } else {
        process.stderr.write(`geo: ${p.id} undecodable, blind default\n`);
        guesses[i] = { id: p.id, lat: BLIND_LAT, lon: BLIND_LON };
      }
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  const answers = guesses.filter((g): g is Guess => g !== null);
  if (answers.length === 0) return refused > 0 ? 1 : 0;
  const res = await fetch(`${envUrl}/answers`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ answers }) });
  process.stdout.write(`geo: ${answers.length} of ${photos.length} photos answered: HTTP ${res.status} ${await res.text()}`);
  return res.ok ? 0 : 1;
}

export { decodeJpegDc };

// Run only when executed directly, so a calibration script can import the
// decoder and the feature code without starting the agent.
import { pathToFileURL } from 'node:url';
if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    process.exitCode = await main();
  } catch (err) {
    process.stderr.write(`geo: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
