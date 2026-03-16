/* global Papa, Chart */

// =============================================================================
// Utilities
// =============================================================================
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const round4 = (x) => (Math.round(x * 10000) / 10000).toFixed(4);
const softmax2 = (a, b) => {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const s = ea + eb;
  return [ea / s, eb / s];
};

function tokenize(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]+/g, " ")
    .split(/\s+/g)
    .filter((t) => t.length >= 2);
}

function ngrams(tokens, nMin, nMax) {
  const out = [];
  for (let n = nMin; n <= nMax; n++) {
    for (let i = 0; i + n <= tokens.length; i++) {
      out.push(tokens.slice(i, i + n).join(" "));
    }
  }
  return out;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function stratifiedSplit(messages, labels, testSize = 0.25, seed = 42) {
  const rng = mulberry32(seed);
  const idx0 = [];
  const idx1 = [];
  labels.forEach((y, i) => (y === 1 ? idx1 : idx0).push(i));

  const shuffle = (arr) => {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  };
  shuffle(idx0);
  shuffle(idx1);

  const n0Test = Math.max(1, Math.floor(idx0.length * testSize));
  const n1Test = Math.max(1, Math.floor(idx1.length * testSize));
  const testIdx = idx0.slice(0, n0Test).concat(idx1.slice(0, n1Test));
  const trainIdx = idx0.slice(n0Test).concat(idx1.slice(n1Test));
  shuffle(testIdx);
  shuffle(trainIdx);

  return { trainIdx, testIdx };
}

// =============================================================================
// Vectorizer (TF-IDF with limited vocabulary)
// =============================================================================
function buildVocab(docs, { ngramMin = 1, ngramMax = 2, minDf = 2, maxFeatures = 4000 } = {}) {
  const df = new Map(); // term -> doc frequency
  const N = docs.length;
  for (const doc of docs) {
    const toks = tokenize(doc);
    const terms = new Set(ngrams(toks, ngramMin, ngramMax));
    for (const t of terms) df.set(t, (df.get(t) || 0) + 1);
  }
  const terms = [];
  for (const [t, c] of df.entries()) {
    if (c >= minDf) terms.push([t, c]);
  }
  terms.sort((a, b) => b[1] - a[1]);
  const top = terms.slice(0, maxFeatures);
  const vocab = new Map();
  top.forEach(([t], i) => vocab.set(t, i));
  const idf = new Float64Array(top.length);
  top.forEach(([_, c], i) => {
    idf[i] = Math.log((N + 1) / (c + 1)) + 1;
  });
  const featureNames = top.map(([t]) => t);
  return { vocab, idf, featureNames, ngramMin, ngramMax };
}

function vectorize(doc, vec) {
  const toks = tokenize(doc);
  const terms = ngrams(toks, vec.ngramMin, vec.ngramMax);
  const tf = new Map(); // idx -> count
  for (const t of terms) {
    const idx = vec.vocab.get(t);
    if (idx === undefined) continue;
    tf.set(idx, (tf.get(idx) || 0) + 1);
  }
  if (tf.size === 0) return { idx: [], val: [] };
  const idx = Array.from(tf.keys()).sort((a, b) => a - b);
  const val = idx.map((i) => (tf.get(i) || 0) * vec.idf[i]);
  // L2 normalize
  let norm = 0;
  for (const v of val) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < val.length; i++) val[i] /= norm;
  return { idx, val };
}

function dotSparse(a, dense) {
  let s = 0;
  for (let k = 0; k < a.idx.length; k++) s += a.val[k] * dense[a.idx[k]];
  return s;
}

function cosineSparse(a, b) {
  // both are L2-normalized already
  let i = 0;
  let j = 0;
  let s = 0;
  while (i < a.idx.length && j < b.idx.length) {
    const ia = a.idx[i];
    const ib = b.idx[j];
    if (ia === ib) {
      s += a.val[i] * b.val[j];
      i++;
      j++;
    } else if (ia < ib) i++;
    else j++;
  }
  return s;
}

// =============================================================================
// Naive Bayes (multinomial-like on TF-IDF weights)
// =============================================================================
function trainNaiveBayes(X, y, nFeatures, alpha = 1.0) {
  const classCount = [0, 0];
  const featSum = [new Float64Array(nFeatures), new Float64Array(nFeatures)];
  const totalSum = [0, 0];

  for (let i = 0; i < X.length; i++) {
    const c = y[i];
    classCount[c] += 1;
    const xi = X[i];
    for (let k = 0; k < xi.idx.length; k++) {
      const j = xi.idx[k];
      const v = xi.val[k];
      featSum[c][j] += v;
      totalSum[c] += v;
    }
  }

  const classLogPrior = [Math.log(classCount[0] / X.length), Math.log(classCount[1] / X.length)];
  const featureLogProb = [new Float64Array(nFeatures), new Float64Array(nFeatures)];
  for (let c = 0; c <= 1; c++) {
    const denom = totalSum[c] + alpha * nFeatures;
    for (let j = 0; j < nFeatures; j++) {
      featureLogProb[c][j] = Math.log((featSum[c][j] + alpha) / denom);
    }
  }
  return { classLogPrior, featureLogProb };
}

function nbLogLikelihood(x, nb) {
  const ll0 = dotSparse(x, nb.featureLogProb[0]);
  const ll1 = dotSparse(x, nb.featureLogProb[1]);
  return [ll0, ll1];
}

function nbPredictProba(x, nb, priorsOverride /* [p0,p1] or null */) {
  const [ll0, ll1] = nbLogLikelihood(x, nb);
  const lp0 = ll0 + (priorsOverride ? Math.log(priorsOverride[0] + 1e-12) : nb.classLogPrior[0]);
  const lp1 = ll1 + (priorsOverride ? Math.log(priorsOverride[1] + 1e-12) : nb.classLogPrior[1]);
  return softmax2(lp0, lp1);
}

// =============================================================================
// Logistic Regression (simple SGD on sparse TF-IDF)
// =============================================================================
function trainLogRegSGD(X, y, nFeatures, { epochs = 10, lr = 0.15, l2 = 1e-4, seed = 7 } = {}) {
  const rng = mulberry32(seed);
  const w = new Float64Array(nFeatures);
  let b = 0;
  const idxs = Array.from({ length: X.length }, (_, i) => i);

  const shuffle = () => {
    for (let i = idxs.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idxs[i], idxs[j]] = [idxs[j], idxs[i]];
    }
  };

  for (let e = 0; e < epochs; e++) {
    shuffle();
    for (const i of idxs) {
      const xi = X[i];
      const yi = y[i];
      let z = b;
      for (let k = 0; k < xi.idx.length; k++) z += w[xi.idx[k]] * xi.val[k];
      const p = 1 / (1 + Math.exp(-z));
      const g = p - yi;
      // update weights
      for (let k = 0; k < xi.idx.length; k++) {
        const j = xi.idx[k];
        w[j] -= lr * (g * xi.val[k] + l2 * w[j]);
      }
      b -= lr * g;
    }
  }
  return { w, b };
}

function lrPredictProba(x, lrModel) {
  let z = lrModel.b;
  for (let k = 0; k < x.idx.length; k++) z += lrModel.w[x.idx[k]] * x.val[k];
  const p1 = 1 / (1 + Math.exp(-z));
  return [1 - p1, p1];
}

// =============================================================================
// KNN (cosine similarity on sparse TF-IDF)
// =============================================================================
function knnPredict(x, Xtrain, ytrain, k = 5) {
  const sims = [];
  for (let i = 0; i < Xtrain.length; i++) {
    sims.push([cosineSparse(x, Xtrain[i]), i]);
  }
  sims.sort((a, b) => b[0] - a[0]);
  const top = sims.slice(0, k);
  let votes1 = 0;
  for (const [, idx] of top) votes1 += ytrain[idx];
  const votes0 = k - votes1;
  const pred = votes1 > votes0 ? 1 : 0;
  return { pred, neighbors: top.map(([, idx]) => idx), votes0, votes1 };
}

// =============================================================================
// Metrics
// =============================================================================
function computeMetrics(yTrue, yPred) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const yt = yTrue[i], yp = yPred[i];
    if (yt === 1 && yp === 1) tp++;
    else if (yt === 0 && yp === 0) tn++;
    else if (yt === 0 && yp === 1) fp++;
    else if (yt === 1 && yp === 0) fn++;
  }
  const acc = (tp + tn) / yTrue.length;
  const prec = tp + fp === 0 ? 0 : tp / (tp + fp);
  const rec = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = prec + rec === 0 ? 0 : (2 * prec * rec) / (prec + rec);
  return { acc, prec, rec, f1 };
}

// =============================================================================
// UI State + Charts
// =============================================================================
const els = {};
let DATA = null;
let VEC = null;
let SPLIT = null;
let Xtrain = null, Xtest = null, ytrain = null, ytest = null;

let NB = null;
let NB_LL_TEST = null; // [ [ll0,ll1], ... ] for test
let LR = null;
let KNN_K = 5;

let charts = {
  c1Pie: null,
  c1Post: null,
  c2Feat: null,
  c2Post: null,
};

function qs(id) { return document.getElementById(id); }

function initEls() {
  [
    "dataStatus", "reloadBtn",
    "tab-priors", "tab-features", "tab-compare",
    "panel-priors", "panel-features", "panel-compare",
    "priorHam", "priorSpam", "priorHamVal", "priorSpamVal", "resetPriorsBtn",
    "c1PHam", "c1PSpam", "c1Acc", "c1Pred",
    "c2Preset", "c2TopN", "c2TopNVal", "c2Custom", "c2AnalyseBtn",
    "c2PHam", "c2PSpam", "c2Pred",
    "c3TrainBtn", "c3TrainStatus", "c3MetricsWrap", "c3Hint",
    "c3Sample", "c3CompareBtn", "c3LR", "c3KNN", "c3NB"
  ].forEach((k) => (els[k] = qs(k)));
}

function setStatus(text, kind = "normal") {
  els.dataStatus.textContent = text;
  els.dataStatus.style.borderColor =
    kind === "ok" ? "rgba(22,163,74,.5)" :
    kind === "err" ? "rgba(220,38,38,.5)" :
    "rgba(15,23,42,0.14)";
}

function initTabs() {
  const tabs = [
    { tab: els["tab-priors"], panel: els["panel-priors"] },
    { tab: els["tab-features"], panel: els["panel-features"] },
    { tab: els["tab-compare"], panel: els["panel-compare"] },
  ];
  const activate = (i) => {
    tabs.forEach((t, j) => {
      t.tab.classList.toggle("is-active", j === i);
      t.panel.classList.toggle("is-active", j === i);
      t.tab.setAttribute("aria-selected", j === i ? "true" : "false");
    });
  };
  tabs.forEach((t, i) => t.tab.addEventListener("click", () => activate(i)));
}

function destroyChart(ch) {
  if (ch) ch.destroy();
  return null;
}

function chartDefaults() {
  Chart.defaults.color = "#64748b";
  Chart.defaults.borderColor = "rgba(15,23,42,0.08)";
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
}

function renderC1Charts(origCounts) {
  chartDefaults();
  charts.c1Pie = destroyChart(charts.c1Pie);
  charts.c1Post = destroyChart(charts.c1Post);

  const pieCtx = qs("c1Pie").getContext("2d");
  charts.c1Pie = new Chart(pieCtx, {
    type: "pie",
    data: {
      labels: ["Ham (0)", "Spam (1)"],
      datasets: [{
        data: [origCounts.ham, origCounts.spam],
        backgroundColor: ["#4ade80", "#f87171"],
        borderColor: "rgba(255,255,255,0.7)",
      }],
    },
    options: {
      plugins: {
        legend: { position: "bottom" },
      },
    },
  });

  const barCtx = qs("c1PosteriorBar").getContext("2d");
  charts.c1Post = new Chart(barCtx, {
    type: "bar",
    data: {
      labels: ["Ham (0)", "Spam (1)"],
      datasets: [{
        label: "Mean posterior",
        data: [0.5, 0.5],
        backgroundColor: ["#4ade80", "#f87171"],
      }],
    },
    options: {
      scales: {
        y: { min: 0, max: 1, ticks: { stepSize: 0.25 } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => ` ${round4(ctx.raw)}` } },
      },
    },
  });
}

function updateC1FromPriors(p0, p1) {
  // Use precomputed log-likelihoods on test set (Tab 1 behavior)
  const preds = new Int8Array(ytest.length);
  let mean0 = 0, mean1 = 0, correct = 0;
  const lp0 = Math.log(p0 + 1e-12);
  const lp1 = Math.log(p1 + 1e-12);

  for (let i = 0; i < NB_LL_TEST.length; i++) {
    const ll = NB_LL_TEST[i];
    const [pHam, pSpam] = softmax2(ll[0] + lp0, ll[1] + lp1);
    mean0 += pHam; mean1 += pSpam;
    preds[i] = pSpam >= 0.5 ? 1 : 0;
    if (preds[i] === ytest[i]) correct++;
  }

  mean0 /= NB_LL_TEST.length;
  mean1 /= NB_LL_TEST.length;
  const acc = correct / NB_LL_TEST.length;
  const predDom = mean1 >= 0.5 ? "Spam" : "Ham";

  qs("c1PHam").textContent = round4(p0);
  qs("c1PSpam").textContent = round4(p1);
  qs("c1Acc").textContent = round4(acc);
  qs("c1Pred").textContent = predDom;
  qs("c1Pred").style.color = predDom === "Spam" ? "var(--spam)" : "var(--ham)";

  charts.c1Post.data.datasets[0].data = [mean0, mean1];
  charts.c1Post.update("none");
}

function initC1PriorsControls(origP0, origP1) {
  let lock = false;

  const setReadouts = () => {
    els.priorHamVal.textContent = Number(els.priorHam.value).toFixed(2);
    els.priorSpamVal.textContent = Number(els.priorSpam.value).toFixed(2);
  };

  const onHam = () => {
    if (lock) return;
    lock = true;
    const ham = Number(els.priorHam.value);
    const spam = clamp(1 - ham, 0.01, 0.99);
    els.priorSpam.value = spam.toFixed(2);
    setReadouts();
    updateC1FromPriors(Number(els.priorHam.value), Number(els.priorSpam.value));
    lock = false;
  };
  const onSpam = () => {
    if (lock) return;
    lock = true;
    const spam = Number(els.priorSpam.value);
    const ham = clamp(1 - spam, 0.01, 0.99);
    els.priorHam.value = ham.toFixed(2);
    setReadouts();
    updateC1FromPriors(Number(els.priorHam.value), Number(els.priorSpam.value));
    lock = false;
  };

  els.priorHam.addEventListener("input", onHam);
  els.priorSpam.addEventListener("input", onSpam);
  els.resetPriorsBtn.addEventListener("click", () => {
    lock = true;
    els.priorHam.value = origP0.toFixed(2);
    els.priorSpam.value = origP1.toFixed(2);
    setReadouts();
    updateC1FromPriors(origP0, origP1);
    lock = false;
  });

  els.priorHam.value = origP0.toFixed(2);
  els.priorSpam.value = origP1.toFixed(2);
  setReadouts();
}

function initC2() {
  chartDefaults();
  charts.c2Feat = destroyChart(charts.c2Feat);
  charts.c2Post = destroyChart(charts.c2Post);

  const featCtx = qs("c2FeatBar").getContext("2d");
  charts.c2Feat = new Chart(featCtx, {
    type: "bar",
    data: { labels: [], datasets: [
      { label: "P(word | Ham)", data: [], backgroundColor: "#28B31E" },
      { label: "P(word | Spam)", data: [], backgroundColor: "#D43E24" },
    ]},
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true } },
      plugins: { legend: { position: "top" } },
    },
  });

  const postCtx = qs("c2Posterior").getContext("2d");
  charts.c2Post = new Chart(postCtx, {
    type: "bar",
    data: {
      labels: ["Ham", "Spam"],
      datasets: [{ data: [0.5, 0.5], backgroundColor: ["#28B31E", "#D43E24"] }],
    },
    options: {
      scales: { y: { min: 0, max: 1, ticks: { stepSize: 0.25 } } },
      plugins: { legend: { display: false } },
    },
  });

  const presets = makeC2Presets();
  els.c2Preset.innerHTML = "";
  for (const p of presets) {
    const opt = document.createElement("option");
    opt.value = p.key;
    opt.textContent = p.label;
    els.c2Preset.appendChild(opt);
  }

  const updateTopNReadout = () => (els.c2TopNVal.textContent = String(els.c2TopN.value));
  updateTopNReadout();

  const onPreset = () => {
    const sel = els.c2Preset.value;
    const isCustom = sel === "__custom__";
    els.c2Custom.disabled = !isCustom;
    if (!isCustom) runC2Analyse();
  };
  els.c2Preset.addEventListener("change", onPreset);
  els.c2TopN.addEventListener("input", () => {
    updateTopNReadout();
  });
  els.c2TopN.addEventListener("change", () => runC2Analyse());
  els.c2AnalyseBtn.addEventListener("click", runC2Analyse);

  onPreset();
}

function makeC2Presets() {
  // Use a couple of real samples from test set for authenticity
  const hamIdx = [];
  const spamIdx = [];
  for (let i = 0; i < ytest.length; i++) (ytest[i] === 1 ? spamIdx : hamIdx).push(i);
  const pickHam = hamIdx[0] ?? 0;
  const pickSpam = spamIdx[0] ?? 0;
  const msgHam = DATA.testMessages[pickHam] || "Good morning, see you at 3.";
  const msgSpam = DATA.testMessages[pickSpam] || "Congratulations! You have won a FREE gift. Claim now!";

  return [
    { key: "ham1", label: 'Ham – Hi, did you have a good day?..', text: "Hi, did you have a good day? Have you spoken to anyone since the weekend?" },
    { key: "ham2", label: 'Ham –  Sir, Good Morning. I hope you had a good weekend.', text: " Sir, Good Morning. I hope you had a good weekend." },
    { key: "spam1", label: 'Spam – Do you want a New Nokia 3510i...', text: "Do you want a New Nokia 3510i Colour Phone Delivered Tomorrow? With 200 FREE minutes to any mobile + 100 FREE text + FREE camcorder Reply or Call 08000930705" },
    { key: "spam2", label: 'Spam – Congratulations ur awarded 500 of CD vouchers ....', text: "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066" },
    { key: "spam3", label: "Spam – Urgent! Your mobile number 077xxx has won a £2,000 Bonus Caller Prize on 02/06/03.", text: "Urgent! Your mobile number 077xxx has won a £2,000 Bonus Caller Prize on 02/06/03. This is the second attempt to contact you. Call 09066362206 as soon as possible. BOX97N7QP, 150ppm." },
    { key: "__custom__", label: "✏️  Custom — type your message below", text: "" },
  ];
}

function runC2Analyse() {
  const presets = makeC2Presets();
  const sel = els.c2Preset.value;
  const isCustom = sel === "__custom__";
  const preset = presets.find((p) => p.key === sel);
  const msg = isCustom ? (els.c2Custom.value || "").trim() : (preset?.text || "");
  const topN = Number(els.c2TopN.value);

  if (!msg) {
    qs("c2Pred").textContent = "—";
    qs("c2Pred").style.color = "var(--muted)";
    qs("c2PHam").textContent = "—";
    qs("c2PSpam").textContent = "—";
    charts.c2Feat.data.labels = [];
    charts.c2Feat.data.datasets.forEach((d) => (d.data = []));
    charts.c2Feat.update("none");
    return;
  }

  const x = vectorize(msg, VEC);
  if (x.idx.length === 0) {
    qs("c2Pred").textContent = "No recognizable vocabulary";
    qs("c2Pred").style.color = "var(--muted)";
    return;
  }

  // Feature likelihoods from NB
  const words = x.idx.map((j) => VEC.featureNames[j]);
  const lkHam = x.idx.map((j) => Math.exp(NB.featureLogProb[0][j]));
  const lkSpam = x.idx.map((j) => Math.exp(NB.featureLogProb[1][j]));
  const importance = x.idx.map((j) => Math.abs(NB.featureLogProb[1][j] - NB.featureLogProb[0][j]));

  const topIdxs = importance
    .map((imp, i) => [imp, i])
    .sort((a, b) => b[0] - a[0])
    .slice(0, topN)
    .map((p) => p[1]);

  const topWords = topIdxs.map((i) => words[i]);
  const topHam = topIdxs.map((i) => lkHam[i]);
  const topSpam = topIdxs.map((i) => lkSpam[i]);

  const post = nbPredictProba(x, NB, null);
  const pred = post[1] >= 0.5 ? "Spam" : "Ham";
  qs("c2PHam").textContent = round4(post[0]);
  qs("c2PSpam").textContent = round4(post[1]);
  qs("c2Pred").textContent = pred;
  qs("c2Pred").style.color = pred === "Spam" ? "var(--spam)" : "var(--ham)";

  charts.c2Feat.data.labels = topWords;
  charts.c2Feat.data.datasets[0].data = topHam;
  charts.c2Feat.data.datasets[1].data = topSpam;
  charts.c2Feat.update("none");

  charts.c2Post.data.datasets[0].data = [post[0], post[1]];
  charts.c2Post.update("none");
}

function initC3() {
  els.c3TrainBtn.addEventListener("click", async () => {
    els.c3TrainBtn.disabled = true;
    els.c3TrainStatus.textContent = "Training…";
    els.c3TrainStatus.classList.add("pill--muted");
    await new Promise((r) => setTimeout(r, 25));

    // NB is already trained for tabs 1/2; train LR and prep KNN.
    LR = trainLogRegSGD(Xtrain, ytrain, VEC.featureNames.length, { epochs: 12, lr: 0.18, l2: 1e-4 });

    // Metrics on test
    const yPredNB = Xtest.map((x) => (nbPredictProba(x, NB, null)[1] >= 0.5 ? 1 : 0));
    const yPredLR = Xtest.map((x) => (lrPredictProba(x, LR)[1] >= 0.5 ? 1 : 0));
    const yPredKNN = Xtest.map((x) => knnPredict(x, Xtrain, ytrain, KNN_K).pred);

    const mNB = computeMetrics(ytest, yPredNB);
    const mLR = computeMetrics(ytest, yPredLR);
    const mKNN = computeMetrics(ytest, yPredKNN);

    renderMetricsTable([
      { name: "Logistic Regression", color: "var(--lr)", ...mLR },
      { name: "KNN", color: "var(--knn)", ...mKNN },
      { name: "Naive Bayes", color: "var(--nb)", ...mNB },
    ]);

    els.c3TrainStatus.textContent = "Trained";
    els.c3TrainBtn.disabled = false;
    els.c3Hint.textContent = "Select a sample and click “Compare the Classifiers” to understand their working.";
  });

  // Populate sample dropdown from fixed user-provided examples
  const samples = makeC3Samples();
  els.c3Sample.innerHTML = "";
  for (const s of samples) {
    const opt = document.createElement("option");
    opt.value = s.key;
    opt.textContent = s.label;
    els.c3Sample.appendChild(opt);
  }

  els.c3CompareBtn.addEventListener("click", () => {
    if (!LR) {
      els.c3Hint.textContent = "Click the training button first to train the models.";
      return;
    }
    const sel = els.c3Sample.value;
    const sample = samples.find((s) => s.key === sel);
    const msg = sample?.text || "";
    const x = vectorize(msg, VEC);
    renderC3Panels(msg, x);
  });
}

function makeC3Samples() {
  const fixed = [
    {
      label: "Spam",
      text: "Do you want a New Nokia 3510i Colour Phone Delivered Tomorrow? With 200 FREE minutes to any mobile + 100 FREE text + FREE camcorder Reply or Call 08000930705",
    },
    {
      label: "Spam",
      text: "Urgent! Your mobile number 077xxx has won a \u00A32,000 Bonus Caller Prize on 02/06/03. This is the second attempt to contact you. Call 09066362206 as soon as possible. BOX97N7QP, 150ppm.",
    },
    {
      label: "Spam",
      text: "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066",
    },
    {
      label: "Ham",
      text: "Hi, did you have a good day? Have you spoken to anyone since the weekend?",
    },
    {
      label: "Ham",
      text: "Sir, Good Morning. I hope you had a good weekend.",
    },
  ];

  return fixed.map((item, k) => {
    const short = item.text.length > 78 ? `${item.text.slice(0, 78)}...` : item.text;
    return { key: `s${k}`, label: `[${item.label}] ${short}`, text: item.text };
  });
}

function renderMetricsTable(rows) {
  const best = {
    acc: Math.max(...rows.map((r) => r.acc)),
    prec: Math.max(...rows.map((r) => r.prec)),
    rec: Math.max(...rows.map((r) => r.rec)),
    f1: Math.max(...rows.map((r) => r.f1)),
  };

  const fmt = (x) => Number(x).toFixed(4);
  const td = (val, isBest) => `<td class="${isBest ? "best" : ""}">${val}</td>`;
  const tr = rows.map((r) => {
    return `<tr>
      <td class="cls" style="color:${r.color}">${r.name}</td>
      ${td(fmt(r.acc), r.acc === best.acc)}
      ${td(fmt(r.prec), r.prec === best.prec)}
      ${td(fmt(r.rec), r.rec === best.rec)}
      ${td(fmt(r.f1), r.f1 === best.f1)}
    </tr>`;
  }).join("");

  els.c3MetricsWrap.innerHTML = `
    <table class="metrics">
      <thead>
        <tr>
          <th style="text-align:left">Classifier</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1 Score</th>
        </tr>
      </thead>
      <tbody>${tr}</tbody>
    </table>
    <div class="hint"><span style="display:inline-block;width:10px;height:10px;background:var(--ham);border-radius:50%;margin-right:6px"></span>best value</div>
  `;
}

function renderC3Panels(msgText, msgVec) {
  renderPanelLR(msgVec);
  renderPanelKNN(msgText, msgVec);
  renderPanelNB(msgVec);
}

function renderPanelLR(x) {
  const coef = LR.w;
  const weights = x.idx.map((j, i) => coef[j] * x.val[i]);
  const top = weights
    .map((w, i) => [Math.abs(w), i])
    .sort((a, b) => b[0] - a[0])
    .slice(0, 6)
    .map((p) => p[1]);

  const rows = top.map((i) => {
    const word = VEC.featureNames[x.idx[i]];
    const w = weights[i];
    const sign = w >= 0 ? "+" : "";
    const col = w >= 0 ? "var(--spam)" : "var(--text)";
    return `<tr>
      <td class="cpWord" style="color:${col}">${word}</td>
      <td class="cpNum"  style="color:${col}">${sign}${w.toFixed(3)}</td>
    </tr>`;
  }).join("");

  const prob = lrPredictProba(x, LR);
  const pred = prob[1] >= 0.5 ? "Spam" : "Ham";
  const boxCls = pred === "Spam" ? "cpPredBox cpPredBox--spam" : "cpPredBox cpPredBox--ham";

  els.c3LR.innerHTML = `
    <div class="cpTitle" style="color:var(--lr)">Logistic Regression</div>
    <div class="cpSubtitle">Top Word Weights</div>
    <table class="cpTable">
      <thead><tr><th>Word</th><th class="cpNum">Weight</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="2" style="color:var(--muted);font-size:12px;padding:8px">No recognizable vocabulary.</td></tr>`}</tbody>
    </table>
    <div class="cpDivider"></div>
    <div class="${boxCls}">
      <span>Predicted Class:&nbsp;${pred}</span>
      <span>Probability:&nbsp;${round4(prob[1])}</span>
    </div>
    <div class="cpHint">Logistic Regression uses weighted word features for classification.</div>
  `;
}

function renderPanelKNN(msgText, x) {
  const res = knnPredict(x, Xtrain, ytrain, KNN_K);
  const pred = res.pred === 1 ? "Spam" : "Ham";
  const boxCls = pred === "Spam" ? "cpPredBox cpPredBox--spam" : "cpPredBox cpPredBox--ham";

  const items = res.neighbors.map((idx, rank) => {
    const txt = DATA.trainMessages[idx] || "";
    const short = txt.length > 55 ? `${txt.slice(0, 55)}…` : txt;
    const lbl = ytrain[idx] === 1 ? "Spam" : "Ham";
    const lblCls = ytrain[idx] === 1 ? "nbrLbl nbrLbl--spam" : "nbrLbl nbrLbl--ham";
    return `<li>
      <div class="nbrLine">
        <div class="nbrTxt">"${short}"</div>
        <div class="${lblCls}">&rarr; ${lbl}</div>
      </div>
    </li>`;
  }).join("");

  els.c3KNN.innerHTML = `
    <div class="cpTitle" style="color:var(--knn)">K-Nearest Neighbors</div>
    <div class="cpSubtitle">Top ${KNN_K} Nearest Documents</div>
    <ol class="neighbors">${items || `<li><span class="hint">No neighbors found.</span></li>`}</ol>
    <div class="cpDivider"></div>
    <div class="cpVotes">Neighbor Votes &rarr; Spam:&nbsp;<strong style="color:var(--spam)">${res.votes1}</strong>&nbsp;&nbsp;Ham:&nbsp;<strong style="color:var(--ham)">${res.votes0}</strong></div>
    <div class="${boxCls}">Predicted Class:&nbsp;${pred}</div>
    <div class="cpHint">KNN classifies based on similarity to training samples.</div>
  `;
}

function renderPanelNB(x) {
  const importance = x.idx.map((j) => Math.abs(NB.featureLogProb[1][j] - NB.featureLogProb[0][j]));
  const top = importance
    .map((imp, i) => [imp, i])
    .sort((a, b) => b[0] - a[0])
    .slice(0, 6)
    .map((p) => p[1]);

  const rows = top.map((i) => {
    const j = x.idx[i];
    const w = VEC.featureNames[j];
    const ps = Math.exp(NB.featureLogProb[1][j]);
    const ph = Math.exp(NB.featureLogProb[0][j]);
    const col = ps > ph ? "var(--spam)" : "var(--text)";
    return `<tr>
      <td class="cpWord" style="color:${col}">${w}</td>
      <td class="cpNum"  style="color:${ps > ph ? 'var(--spam)' : 'var(--text)'}">${ps.toFixed(5)}</td>
      <td class="cpNum">${ph.toFixed(5)}</td>
    </tr>`;
  }).join("");

  const post = nbPredictProba(x, NB, null);
  const pred = post[1] >= 0.5 ? "Spam" : "Ham";
  const boxCls = pred === "Spam" ? "cpPredBox cpPredBox--spam" : "cpPredBox cpPredBox--ham";

  els.c3NB.innerHTML = `
    <div class="cpTitle" style="color:var(--nb)">Naive Bayes</div>
    <div class="cpSubtitle">Word Probabilities</div>
    <table class="cpTable">
      <thead><tr><th>Word</th><th class="cpNum">P(w|Spam)</th><th class="cpNum">P(w|Ham)</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="3" style="color:var(--muted);font-size:12px;padding:8px">No recognizable vocabulary features found.</td></tr>`}</tbody>
    </table>
    <div class="cpDivider"></div>
    <div class="cpPosterior">Posterior &rarr; Spam:&nbsp;<strong style="color:var(--spam)">${round4(post[1])}</strong>&nbsp;&nbsp;Ham:&nbsp;<strong style="color:var(--ham)">${round4(post[0])}</strong></div>
    <div class="${boxCls}">Predicted Class:&nbsp;${pred}</div>
    <div class="cpHint">Naive Bayes uses probabilities to drive classification.</div>
  `;
}

// =============================================================================
// Dataset load + pipeline
// =============================================================================
async function loadCSV() {
  return new Promise((resolve, reject) => {
    Papa.parse("./Spam_Detection.csv", {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data),
      error: (err) => reject(err),
    });
  });
}

function normalizeRow(row) {
  // Common variants in spam datasets: Category/Message or v1/v2
  const cat = (row.Category ?? row.v1 ?? row.label ?? "").toString().trim();
  const msg = (row.Message ?? row.v2 ?? row.text ?? "").toString();
  if (!cat || !msg) return null;
  const y = cat.toLowerCase() === "spam" ? 1 : 0;
  return { y, msg };
}

async function bootstrap() {
  setStatus("Loading dataset…");

  const rows = await loadCSV();
  const data = [];
  for (const r of rows) {
    const rr = normalizeRow(r);
    if (rr) data.push(rr);
  }
  if (data.length < 100) throw new Error("Dataset parsed but has too few rows.");

  const messages = data.map((d) => d.msg);
  const labels = data.map((d) => d.y);

  SPLIT = stratifiedSplit(messages, labels, 0.25, 42);
  const trainMessages = SPLIT.trainIdx.map((i) => messages[i]);
  const testMessages = SPLIT.testIdx.map((i) => messages[i]);
  const trainLabels = SPLIT.trainIdx.map((i) => labels[i]);
  const testLabels = SPLIT.testIdx.map((i) => labels[i]);

  DATA = { trainMessages, testMessages };
  ytrain = trainLabels;
  ytest = testLabels;

  // Build vectorizer similar to Cell 3 (1,2 ngrams) but keep modest features
  VEC = buildVocab(trainMessages, { ngramMin: 1, ngramMax: 2, minDf: 2, maxFeatures: 3500 });
  Xtrain = trainMessages.map((m) => vectorize(m, VEC));
  Xtest = testMessages.map((m) => vectorize(m, VEC));

  NB = trainNaiveBayes(Xtrain, ytrain, VEC.featureNames.length, 1.0);
  NB_LL_TEST = Xtest.map((x) => nbLogLikelihood(x, NB));

  const nHam = ytest.reduce((s, y) => s + (y === 0 ? 1 : 0), 0);
  const nSpam = ytest.length - nHam;
  const origP0 = nHam / ytest.length;
  const origP1 = nSpam / ytest.length;

  renderC1Charts({ ham: nHam, spam: nSpam });
  initC1PriorsControls(origP0, origP1);
  updateC1FromPriors(origP0, origP1);

  initC2();
  initC3();

  setStatus(`Dataset loaded • Train: ${ytrain.length} • Test: ${ytest.length}`, "ok");
}

function wireReload() {
  els.reloadBtn.addEventListener("click", async () => {
    // reset charts and state
    charts.c1Pie = destroyChart(charts.c1Pie);
    charts.c1Post = destroyChart(charts.c1Post);
    charts.c2Feat = destroyChart(charts.c2Feat);
    charts.c2Post = destroyChart(charts.c2Post);
    LR = null;
    els.c3MetricsWrap.innerHTML = "";
    els.c3TrainStatus.textContent = "Not trained";
    els.c3Hint.textContent = "Click the button to train models and view metrics.";

    try {
      await bootstrap();
    } catch (e) {
      console.error(e);
      setStatus(`Failed to load: ${e.message}`, "err");
    }
  });
}

// =============================================================================
// Boot
// =============================================================================
initEls();
initTabs();
wireReload();

bootstrap().catch((e) => {
  console.error(e);
  setStatus(
    "Could not load CSV. Run with a local server (not file://) and ensure Spam_Detection.csv is alongside index.html.",
    "err"
  );
});

