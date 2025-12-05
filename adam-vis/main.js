import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'lil-gui';
import { shaders } from './shaders.js';
import { Arrow } from './arrow.js';
import {_plasma_data, _viridis_data} from "./color_map.js"

let controlCanvas, controlCtx;
const cameraCenter = new THREE.Vector3(0, 0, 0.25);
const scene = new THREE.Scene();
const clock = new THREE.Clock();
const camera = new THREE.PerspectiveCamera(25, innerWidth / innerHeight, 0.1, 1000);
camera.position.set(0, 4, 2);
camera.up.set(0, 0, 1);
camera.lookAt(cameraCenter);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.style.margin = '0';
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.autoRotate = true;
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = false;

let stepArrows = [];

let params = {
    optimizer: "adam",
    warmup: 10000,
    steps: 100000,
    lr: 1.0,
    b1: 0.92,
    b2: 0.44,
    weight_decay: 0.01,
    epsilon: 0,

    
};

let vizParams =  {
    palette: "plasma",
    animationSpeed : 1.0,
    point_size: 2,
    alpha: 1,
    n_arrows: 12,
};

let colorCache = {
  steps: null,
  palette: null,
  array: null  // Float32Array length = steps * 3
};

const presets = {
    "Preset 1": () => { Object.assign(params, { warmup: 10000, steps: 100000, lr: 1.0, b1: 0.92, b2: 0.44, weight_decay: 0.01, epsilon: 0}); Object.assign(vizParams, {point_size: 2, alpha: 1, n_arrows: 12}); updateVisualization(); updateBox(); },
    "Preset 2": () => { Object.assign(params, { warmup: 10000, steps: 1000000, lr: 1.0, b1: 0.51, b2: 0.55, weight_decay: 0.01, epsilon: 0}); Object.assign(vizParams, {point_size: 2, alpha: 0.1, n_arrows: 0}); updateVisualization(); updateBox(); },
    "Preset 3": () => { Object.assign(params, { warmup: 10000, steps: 1000000, lr: 1.0, b1: 0.9, b2: 0.14, weight_decay: 0.01, epsilon: 0}); Object.assign(vizParams, {point_size: 2, alpha: 0.1, n_arrows: 16}); updateVisualization(); updateBox(); },
    "Preset 4": () => { Object.assign(params, { warmup: 10000, steps: 1000000, lr: 1.0, b1: 0.22, b2: 0.55, weight_decay: 0.01, epsilon: 0}); Object.assign(vizParams, {point_size: 2, alpha: 0.05, n_arrows: 0}); updateVisualization(); updateBox(); },
};

const rs = { s: 1, box: 300, guiW: 320, guiFS: 13, titleFS: 18, labelFS: 36, spriteFS: 32 };

function setupGUI() {
    const gui = new GUI({ title: 'Limit Cycles & Chaos in Adam' });
    const info = gui.addFolder('📖 About');
    const el = document.createElement('div');
    el.innerHTML = `
        <div style="padding: 10px; font-size: 13px; line-height: 1.4; color: #fff !important; width: 100%; background: rgba(0,0,0,0.1);">
            <strong style="color: #6cf;">Visualization of Adam on f(x) = ½x²</strong><br/>
            <ul style="margin: 8px 0; padding-left: 20px;">
                <li>First run Adam from x=1 for "Warmup Iterations" to reach a steady state</li>
                <li>Then show a scatter plot of the next "Visualization Points" iterations</li>
            </ul>

            <strong style="color: #6cf;">What you're seeing:</strong><br/>
            <ul style="margin: 8px 0; padding-left: 20px;">
                <li>White points: optimization path (x, m, v)</li>
                <li>Red arrows: recent steps</li>
                <li>Axes: the three Adam parameters
                    <ul style="margin: 4px 0; padding-left: 20px;">
                        <li>x (param)</li>
                        <li>m (momentum)</li>
                        <li>v (velocity)</li>
                    </ul>
                </li>
            </ul>

            <strong style="color: #6cf;">(β₁,β₂) Box (right):</strong><br/>
            <ul style="margin: 8px 0; padding-left: 20px;">
                <li>Click/drag to set parameters</li>
                <li>Green: stable zone</li>
                <li>Red: unstable zone</li>
            </ul>

            <strong style="color: #6cf;">Try the presets or drag the (β₁,β₂) dot!</strong>
        </div>
    `;
    info.domElement.querySelector('.children').appendChild(el);

    const ctrls = gui.addFolder('🎛️ Controls');
    ctrls.add(params, 'optimizer').name('Optimizer').options(["adam", "adamw"]).onChange(updateVisualization).listen();
    ctrls.add(params, 'warmup').name('Warmup Iterations').step(1).onChange(updateVisualization).listen();
    ctrls.add(params, 'steps').name('Visualization Points').min(0).step(1).onChange(updateVisualization).listen();
    ctrls.add(params, 'b1', 0, 1).name('B1').onChange(() => { updateVisualization(); updateBox(); }).listen();
    ctrls.add(params, 'b2', 0, 1).name('B2').onChange(() => { updateVisualization(); updateBox(); }).listen();
    ctrls.add(params, 'lr').name('Learning Rate').min(0).onChange(updateVisualization).listen();
    ctrls.add(params, 'epsilon').name('epsilon').min(0).onChange(updateVisualization).listen();
    ctrls.add(params, 'weight_decay', 0, 1).name('weight_decay').onChange(updateVisualization).listen();

    const viewCtrls = gui.addFolder('🎛️ Vizualization');

    viewCtrls.add(vizParams, 'palette').name('Palette').options(["plasma", "viridis", "white"]).onChange(() => { ensureColorCache(params.steps, vizParams.palette); applyColorsToGeometry(); }).listen();
    viewCtrls.add(vizParams, 'animationSpeed', 0.0, 2.0).name('Animation Speed').listen();

    viewCtrls.add(vizParams, 'point_size').name('Point Size').min(0).onChange(() => { material.uniforms.size.value = vizParams.point_size; }).listen();
    viewCtrls.add(vizParams, 'alpha', 0, 1).name('Alpha').onChange(() => { material.uniforms.alpha.value = vizParams.alpha; }).listen();
    viewCtrls.add(vizParams, 'n_arrows').name('Number of Arrows').min(0).step(1).onChange(updateArrows).listen();

    Object.keys(presets).forEach(k => ctrls.add(presets, k).name(k));

    const style = document.createElement('style');
    style.textContent = `
    .lil-gui{font-size:${rs.guiFS}px!important;width:${rs.guiW}px!important;z-index:10;max-height:80vh;overflow:auto}
    .lil-gui .folder>.title{font-size:${rs.titleFS}px!important;font-weight:bold!important;padding:10px 8px!important}
    .lil-gui.root>.title::before{display:none!important}
  `;
    style.setAttribute('data-responsive-gui', '');
    document.head.appendChild(style);
    gui.domElement.style.top = '10px';
    gui.domElement.style.left = '10px';
}


function applyColorsToGeometry() {
  if (!geometry) return;
  const steps = (geometry.attributes.position.array.length / 3) | 0;
  const colors = ensureColorCache(steps, vizParams.palette);

  if (!geometry.getAttribute("aColor") || geometry.getAttribute("aColor").array.length !== colors.length) {
    // create new attribute
    geometry.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(colors), 3));
  } else {
    // update in-place
    const colArr = geometry.getAttribute("aColor").array;
    colArr.set(colors);
    geometry.getAttribute("aColor").needsUpdate = true;
  }
}


function ensureColorCache(steps, palette = 'plasma') {
  if (colorCache.steps === steps && colorCache.palette === palette && colorCache.array) {
    return colorCache.array;
  }

  const arr = new Float32Array(steps * 3);


  if(palette == "white"){
    arr.fill(1.0);
    return arr;
  }

  // sampled stops for viridis and plasma (8 stops each) - linear interpolation between stops
  // (values taken from common matplotlib samples; we use a small set and linearly interpolate)
  const palettes = {
    viridis: _viridis_data,
    plasma: _plasma_data
  };

  const stops = palettes[palette] || palettes.plasma;
  const nStops = stops.length;

  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? 0.0 : i / (steps - 1); // normalized 0..1
    // map t into stops
    const scaled = t * (nStops - 1);
    const idx = Math.floor(scaled);
    const frac = Math.min(Math.max(scaled - idx, 0), 1);
    const c0 = stops[idx];
    const c1 = stops[Math.min(idx + 1, nStops - 1)];
    const r = c0[0] * (1 - frac) + c1[0] * frac;
    const g = c0[1] * (1 - frac) + c1[1] * frac;
    const b = c0[2] * (1 - frac) + c1[2] * frac;
    arr[i * 3] = r;
    arr[i * 3 + 1] = g;
    arr[i * 3 + 2] = b;
  }

  colorCache.steps = steps;
  colorCache.palette = palette;
  colorCache.array = arr;
  return arr;
}


let workerRunning = false, pendingUpdate = null, adamWorker = null;
function updateVisualization() {
    material.uniforms.size.value = vizParams.point_size;
    material.uniforms.alpha.value = vizParams.alpha;

    ensureColorCache(params.steps, vizParams.palette);
    pendingUpdate = { params, steps: params.steps, warmup: params.warmup };
    if (!workerRunning) processNextUpdate();
}
function processNextUpdate() {
    if (!pendingUpdate) return;
    const req = pendingUpdate; pendingUpdate = null; workerRunning = true;
    if (adamWorker) { adamWorker.terminate(); adamWorker = null; }
    adamWorker = new Worker(new URL('./adamWorker.js', import.meta.url), { type: 'module' });
    adamWorker.onmessage = e => {
        const { vertices } = e.data;
        const arr = geometry.attributes.position.array;
        if (vertices.length !== arr.length) {
            geometry.dispose();
            geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));

            const steps = vertices.length / 3;
            const colors = ensureColorCache(steps, vizParams.palette);
            geometry.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(colors), 3));


            mesh.geometry = geometry;
        } else {
            arr.set(vertices);
            geometry.attributes.position.needsUpdate = true;

            applyColorsToGeometry();
        }
        workerRunning = false;
        processNextUpdate();
        updateArrows();
    };
    adamWorker.onerror = (err) => { console.error('Adam worker error:', err); workerRunning = false; };
    adamWorker.onmessageerror = (err) => { console.error('Adam worker messageerror:', err); workerRunning = false; };


    adamWorker.postMessage(req);
}

let geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(params.steps * 3), 3));

const initialColors = ensureColorCache(params.steps, vizParams.palette);
geometry.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(initialColors), 3));

const material = new THREE.ShaderMaterial({
    uniforms: { color: { value: new THREE.Color(1, 1, 1) }, size: { value: vizParams.point_size }, alpha: { value: vizParams.alpha } },
    vertexShader: shaders.vertex,
    fragmentShader: shaders.fragment,
    transparent: true,
    depthWrite: false,
    vertexColors: true
});
updateVisualization();
const mesh = new THREE.Points(geometry, material);
scene.add(mesh);

function addAxes() {
    const L = 1.0, t = 0.005, c = 'rgb(75,75,150)';
    scene.add(Arrow(L, 0, 0, 0, 0, 0, t, c));
    scene.add(Arrow(-L, 0, 0, 0, 0, 0, t, c));
    scene.add(Arrow(0, L, 0, 0, 0, 0, t, c));
    scene.add(Arrow(0, -L, 0, 0, 0, 0, t, c));
    scene.add(Arrow(0, 0, L, 0, 0, 0, t, c));
    const x = createTextSprite('x', { fontsize: rs.spriteFS, scale: 0.5, fillStyle: 'rgb(75,75,150)' }); x.position.set(1.1, 0, 0); scene.add(x);
    const y = createTextSprite('m', { fontsize: rs.spriteFS, scale: 0.5, fillStyle: 'rgb(75,75,150)' }); y.position.set(0, 1.1, 0); scene.add(y);
    const z = createTextSprite('v', { fontsize: rs.spriteFS, scale: 0.5, fillStyle: 'rgb(75,75,150)' }); z.position.set(0, 0, 1.1); scene.add(z);
}
function createTextSprite(text, { fontsize = 24, fontface = null, fillStyle = 'white', scale = 1 } = {}) {
    const dpr = Math.min(2, devicePixelRatio || 1);
    const w = fontsize * 4, h = fontsize * 2;
    const canvas = document.createElement('canvas');
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
    ctx.font = fontsize + 'px ' + (fontface || 'sans-serif');
    ctx.fillStyle = fillStyle; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, w / 2, h / 2);
    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(mat); sprite.scale.set(scale, scale / 2, 1);
    return sprite;
}
addAxes();

function animate() {
    const dt = clock.getDelta() * vizParams.animationSpeed;
    controls.update(dt);
    camera.lookAt(cameraCenter);
    renderer.render(scene, camera);
}
renderer.setAnimationLoop(animate);

function setupSimpleBox() {
    const container = document.createElement('div');
    container.style.position = 'absolute';
    container.style.top = '12px';
    container.style.right = '12px';
    container.style.display = 'flex';
    container.style.touchAction = 'none';
    container.style.zIndex = 9;

    controlCanvas = document.createElement('canvas');
    const dpr = Math.min(2, devicePixelRatio || 1);
    controlCanvas.width = rs.box * dpr;
    controlCanvas.height = rs.box * dpr;
    controlCanvas.style.width = rs.box + 'px';
    controlCanvas.style.height = rs.box + 'px';
    controlCtx = controlCanvas.getContext('2d');
    controlCtx.scale(dpr, dpr);

    const beta1Label = document.createElement('div');
    Object.assign(beta1Label.style, { position: 'absolute', textAlign: 'center', top: '100%', left: '0', transform: 'translateY(8px)', color: 'white', fontSize: rs.labelFS + 'px', width: rs.box + 'px' });

    const beta2Label = document.createElement('div');
    Object.assign(beta2Label.style, { position: 'absolute', textAlign: 'center', top: '100%', left: '0', transform: 'translateY(-100%) rotate(-90deg) translateY(-8px)', transformOrigin: 'left bottom', color: 'white', fontSize: rs.labelFS + 'px', width: rs.box + 'px' });

    container.appendChild(controlCanvas);
    container.appendChild(beta1Label);
    container.appendChild(beta2Label);
    document.body.appendChild(container);

    function updateLabels() {
        beta1Label.textContent = `β₁ = ${params.b1.toFixed(3)}`;
        beta2Label.textContent = `β₂ = ${params.b2.toFixed(3)}`;
    }
    controlCanvas.updateLabels = updateLabels;

    let active = false;
    const pick = (cx, cy) => {
        const r = controlCanvas.getBoundingClientRect();
        let x = cx - r.left, y = cy - r.top;
        let b1 = x / rs.box, b2 = 1 - (y / rs.box);
        params.b1 = Math.min(Math.max(b1, 1e-4), 1 - 1e-4);
        params.b2 = Math.min(Math.max(b2, 1e-4), 1 - 1e-4);
        updateBox(); updateLabels(); updateVisualization();
    };

    controlCanvas.addEventListener('mousedown', e => { active = true; pick(e.clientX, e.clientY); });
    addEventListener('mousemove', e => { if (active) pick(e.clientX, e.clientY); });
    addEventListener('mouseup', () => { active = false; });

    controlCanvas.addEventListener('touchstart', e => { e.preventDefault(); active = true; const t = e.changedTouches[0]; pick(t.clientX, t.clientY); }, { passive: false });
    addEventListener('touchmove', e => { if (!active) return; e.preventDefault(); const t = e.changedTouches[0]; pick(t.clientX, t.clientY); }, { passive: false });
    addEventListener('touchend', () => { active = false; });

    updateLabels();
    updateBox();
}

function updateBox() {
    const s = rs.box;
    controlCtx.clearRect(0, 0, s, s);
    controlCtx.fillStyle = 'red';
    controlCtx.fillRect(0, 0, s, s);

    controlCtx.save();
    controlCtx.beginPath();
    const x0 = 3 - 2 * Math.sqrt(2), x1 = 1 / 3, N = 100;
    controlCtx.moveTo(x0 * s, 0);
    for (let i = 1; i <= N; i++) {
        const a = i / N, x = x0 + (x1 - x0) * a, y = (1 - 3 * x) / (x * (3 - x));
        controlCtx.lineTo(x * s, (1 - y) * s);
    }
    controlCtx.lineTo(0, s);
    controlCtx.lineTo(0, 0);
    controlCtx.closePath();
    controlCtx.fillStyle = 'green';
    controlCtx.fill();
    controlCtx.restore();

    const dotX = params.b1 * s, dotY = s - params.b2 * s;
    controlCtx.fillStyle = 'black';
    controlCtx.beginPath(); controlCtx.arc(dotX, dotY, 6, 0, Math.PI * 2); controlCtx.fill();
    controlCtx.strokeStyle = 'white';
    controlCtx.lineWidth = 1;
    controlCtx.beginPath(); controlCtx.arc(dotX, dotY, 6, 0, Math.PI * 2); controlCtx.stroke();

    controlCanvas.updateLabels();
}

function updateArrows() {
    stepArrows.forEach(a => scene.remove(a));
    stepArrows = [];
    const p = geometry.attributes.position.array;
    const n = Math.min(vizParams.n_arrows, p.length / 3 - 1);
    for (let i = 0; i < n; i++) {
        const ax = p[i * 3], ay = p[i * 3 + 1], az = p[i * 3 + 2];
        const bx = p[(i + 1) * 3], by = p[(i + 1) * 3 + 1], bz = p[(i + 1) * 3 + 2];
        const arrow = Arrow(bx, by, bz, ax, ay, az, 0.005, 0xFF0000);
        scene.add(arrow);
        stepArrows.push(arrow);
    }
}

function onResize() {
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.setSize(innerWidth, innerHeight);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();

    if (controlCanvas) {
        const dpr = Math.min(2, devicePixelRatio || 1);
        controlCanvas.width = rs.box * dpr;
        controlCanvas.height = rs.box * dpr;
        controlCanvas.style.width = rs.box + 'px';
        controlCanvas.style.height = rs.box + 'px';
        controlCtx.setTransform(1, 0, 0, 1, 0, 0);
        controlCtx.scale(dpr, dpr);
        updateBox();
        controlCanvas.updateLabels();
    }

    const style = document.querySelector('style[data-responsive-gui]');
    if (style) {
        style.textContent = `.lil-gui{font-size:${rs.guiFS}px!important;width:${rs.guiW}px!important;max-height:80vh;overflow:auto}
                         .lil-gui .folder>.title{font-size:${rs.titleFS}px!important}`;
    }
}
addEventListener('resize', onResize, { passive: true });

document.addEventListener('visibilitychange', () => {
    renderer.setAnimationLoop(document.hidden ? null : animate);
});

setupSimpleBox();
setupGUI();
onResize();