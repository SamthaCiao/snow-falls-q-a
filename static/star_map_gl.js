/**
 * 探索星图 WebGL2：星云底图渐进揭示 + 弧段内星点 + hover 才显示连线
 */
(function (global) {
    const TYPE_COLOR = {
        character: [1.0, 0.92, 0.72],
        event: [0.55, 0.82, 1.0],
        imagery: [0.78, 0.58, 1.0],
    };

    let gl = null;
    let progStar = null;
    let progLine = null;
    let starVao = null;
    let catalogCache = null;
    let animId = null;
    let startTime = 0;
    let currentPayload = null;
    let canvasEl = null;
    let wrapEl = null;
    let detailEl = null;
    let dimBgEl = null;
    let brightBgEl = null;
    let hoveredNodeId = null;

    const VS_STAR = `#version 300 es
in vec2 aPos;
in float aMag;
in float aImp;
in vec3 aColor;
in float aPhase;
uniform vec2 uRes;
uniform float uReveal;
out vec3 vColor;
out float vImp;
void main(){
  float reveal = clamp((uReveal - aPhase) / max(1.0 - aPhase, 0.01), 0.0, 1.0);
  if(reveal <= 0.001){ gl_Position = vec4(2.0); gl_PointSize = 0.0; return; }
  vec2 p = aPos * 2.0 - 1.0;
  p.x *= uRes.x / uRes.y;
  gl_Position = vec4(p, 0.0, 1.0);
  float flux = pow(10.0, -0.4 * (aMag - 2.0));
  float sz = 3.5 + flux * 12.0 * aImp * reveal;
  gl_PointSize = clamp(sz, 3.0, 18.0);
  vColor = aColor;
  vImp = aImp * reveal;
}`;

    const FS_STAR = `#version 300 es
precision highp float;
in vec3 vColor;
in float vImp;
out vec4 o;
void main(){
  vec2 c = gl_PointCoord - 0.5;
  float r2 = dot(c, c);
  float psf = exp(-r2 * 14.0);
  float I = vImp * psf * 1.35;
  if(I < 0.004) discard;
  o = vec4(vColor * I, min(I * 0.85, 0.92));
}`;

    const VS_LINE = `#version 300 es
in vec2 aPos;
uniform vec2 uRes;
void main(){
  vec2 p = aPos * 2.0 - 1.0;
  p.x *= uRes.x / uRes.y;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

    const FS_LINE = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 o;
void main(){ o = uColor; }`;

    function compile(type, src) {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
            throw new Error(gl.getShaderInfoLog(s) || 'shader compile');
        }
        return s;
    }

    function link(vs, fs) {
        const p = gl.createProgram();
        gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
        gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(p) || 'link');
        }
        return p;
    }

    function initGl(canvas) {
        gl = canvas.getContext('webgl2', { alpha: true, antialias: true, premultipliedAlpha: true });
        if (!gl) throw new Error('WebGL2 不可用');
        progStar = link(VS_STAR, FS_STAR);
        progLine = link(VS_LINE, FS_LINE);
        starVao = gl.createVertexArray();
    }

    function isNebulaMode(payload) {
        return payload.background_mode === 'nebula_subject'
            || (payload.background && payload.background.mode === 'nebula_subject')
            || (catalogCache && catalogCache.background_mode === 'nebula_subject')
            || (catalogCache && catalogCache.background && catalogCache.background.mode === 'nebula_subject');
    }

    function pathRevealIndex(progress, pathLen) {
        if (pathLen <= 1) return 0;
        const p = Math.max(0, Math.min(1, progress));
        if (p <= 0) return 0;
        return Math.round(p * (pathLen - 1));
    }

    function updateNebulaReveal(path, progress, cssW, cssH) {
        if (!brightBgEl || !path || path.length < 2) return;
        const p = Math.max(0, Math.min(1, progress));
        if (p <= 0) {
            applyMaskToBright(blankMaskDataUrl(cssW, cssH));
            return;
        }
        const cap = pathRevealIndex(p, path.length);
        const nReveal = Math.max(1, cap + 1);
        const mc = document.createElement('canvas');
        mc.width = Math.max(64, Math.floor(cssW));
        mc.height = Math.max(64, Math.floor(cssH));
        const ctx = mc.getContext('2d');
        ctx.clearRect(0, 0, mc.width, mc.height);

        const toXY = (pt) => [pt[0] * mc.width, (1 - pt[1]) * mc.height];

        // 全路径淡轮廓（未揭示区 hint）
        ctx.strokeStyle = 'rgba(255,255,255,0.14)';
        ctx.lineWidth = Math.max(10, mc.width * 0.014);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.filter = 'blur(5px)';
        ctx.beginPath();
        path.forEach((pt, i) => {
            const [px, py] = toXY(pt);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.stroke();

        // 已揭示弧段：渐变亮区（白=mask 可见）
        ctx.filter = 'blur(12px)';
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.lineWidth = Math.max(26, mc.width * 0.036);
        ctx.beginPath();
        for (let i = 0; i < nReveal; i++) {
            const [px, py] = toXY(path[i]);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();

        const maskUrl = mc.toDataURL('image/png');
        applyMaskToBright(maskUrl);
    }

    function ensureNebulaLayers(wrap, bgUrl, cssW, cssH) {
        if (!wrap) return;
        let dim = wrap.querySelector('.star-map-bg-dim');
        let bright = wrap.querySelector('.star-map-bg-bright');
        if (!dim) {
            dim = document.createElement('div');
            dim.className = 'star-map-bg-dim';
            bright = document.createElement('div');
            bright.className = 'star-map-bg-bright';
            wrap.insertBefore(dim, wrap.firstChild);
            wrap.insertBefore(bright, wrap.firstChild);
        }
        const bg = `url("${bgUrl}")`;
        dim.style.backgroundImage = bg;
        bright.style.backgroundImage = bg;
        dimBgEl = dim;
        brightBgEl = bright;
        const mw = cssW || wrap.clientWidth || 640;
        const mh = cssH || wrap.clientHeight || 360;
        applyMaskToBright(blankMaskDataUrl(mw, mh));
    }

    function setBackground(wrap, url, nebulaMode, cssW, cssH) {
        if (!wrap) return;
        if (nebulaMode) {
            ensureNebulaLayers(wrap, url, cssW, cssH);
            wrap.style.backgroundImage = '';
        } else {
            wrap.style.backgroundImage = `url("${url}")`;
            wrap.style.backgroundSize = 'cover';
            wrap.style.backgroundPosition = 'center';
            wrap.style.backgroundColor = '#050508';
        }
    }

    let displayProgress = 0;
    let targetProgress = 0;

    function blankMaskDataUrl(w, h) {
        const mc = document.createElement('canvas');
        mc.width = Math.max(8, w);
        mc.height = Math.max(8, h);
        mc.getContext('2d').clearRect(0, 0, mc.width, mc.height);
        return mc.toDataURL('image/png');
    }

    function applyMaskToBright(url) {
        if (!brightBgEl) return;
        const v = `url("${url}")`;
        brightBgEl.style.webkitMaskImage = v;
        brightBgEl.style.maskImage = v;
    }

    let markerCanvas = null;
    let markerCtx = null;
    let starCount = 0;
    let starLitKey = '';
    let pointerBound = false;

    function getMobiusPath(payload) {
        const p = payload || currentPayload || {};
        return p.mobius_full_points
            || (p.mobius && p.mobius.path)
            || (catalogCache && catalogCache.mobius && catalogCache.mobius.path)
            || [];
    }

    function snapNodeToPath(node, payload) {
        const path = getMobiusPath(payload);
        const pi = node.path_index != null ? node.path_index : 0;
        const i = Math.max(0, Math.min(pi, path.length - 1));
        if (node.x != null && node.y != null) {
            return node;
        }
        if (path[i]) return { ...node, x: path[i][0], y: path[i][1] };
        return node;
    }

    function ensureMarkerCanvas(wrap) {
        if (!wrap) return;
        let mc = wrap.querySelector('#starMapMarkers, .star-map-markers');
        if (!mc) {
            mc = document.createElement('canvas');
            mc.className = 'star-map-markers';
            mc.id = 'starMapMarkers';
            wrap.appendChild(mc);
        }
        markerCanvas = mc;
        markerCtx = mc.getContext('2d');
    }

    function catalogNodes(payload) {
        return (payload.nodes || (catalogCache && catalogCache.nodes) || []);
    }

    function litCharacters(payload) {
        const nebulaMode = isNebulaMode(payload);
        const nodes = nebulaMode ? visibleLitNodes(payload, true) : allLitNodes(payload);
        return nodes
            .filter((n) => n.type === 'character')
            .map((n) => snapNodeToPath(n, payload));
    }

    function getNodeParents(node) {
        if (Array.isArray(node.parents) && node.parents.length) return node.parents;
        if (node.parent) return [node.parent];
        return [];
    }

    function eventMatchesCharacter(eventNode, charLabel) {
        return getNodeParents(eventNode).includes(charLabel);
    }

    function pathStepDistance(a, b) {
        return Math.abs((a ?? 0) - (b ?? 0));
    }

    function normDist(a, b) {
        return Math.hypot((a.x ?? a.drawX) - (b.x ?? b.drawX), (a.y ?? a.drawY) - (b.y ?? b.drawY));
    }

    function inMainNebulaBand(x, y) {
        return x >= 0.24 && x <= 0.76 && y >= 0.20 && y <= 0.82;
    }

    function satellitesForCharacter(charNode, payload) {
        if (!charNode || charNode.type !== 'character') return [];
        const label = charNode.label || '';
        const nebulaMode = isNebulaMode(payload);
        const cap = nebulaMode
            ? pathRevealIndex(payload.progress || 0, (getMobiusPath(payload).length || 1))
            : Infinity;
        const charPi = charNode.path_index ?? 0;
        const maxPathSteps = 9;
        const maxNormDist = 0.17;

        const sats = catalogNodes(payload)
            .filter((n) => (n.type === 'event' || n.type === 'imagery') && eventMatchesCharacter(n, label))
            .sort((a, b) => (a.path_index ?? 0) - (b.path_index ?? 0) || (b.importance || 0) - (a.importance || 0));
        const byId = new Map();
        sats.forEach((n) => { if (!byId.has(n.id)) byId.set(n.id, n); });

        return Array.from(byId.values())
            .map((n) => snapNodeToPath(n, payload))
            .filter((n) => {
                const pi = n.path_index ?? 0;
                if (pi > cap) return false;
                if (pathStepDistance(pi, charPi) > maxPathSteps) return false;
                if (normDist(n, charNode) > maxNormDist) return false;
                if (!inMainNebulaBand(n.x, n.y)) return false;
                return true;
            })
            .slice(0, 8);
    }

    function layoutSatelliteOnPath(sat, index, total, charNode, path, payload) {
        const snapped = snapNodeToPath(sat, payload);
        return { drawX: snapped.x, drawY: snapped.y };
    }

    function findActiveCharacter(hoveredId, chars, payload) {
        if (!hoveredId) return null;
        const direct = chars.find((c) => c.id === hoveredId);
        if (direct) return direct;
        const hoveredNode = catalogNodes(payload).find((n) => n.id === hoveredId)
            || (payload.lit_nodes || []).find((n) => n.id === hoveredId);
        if (hoveredNode && (hoveredNode.type === 'event' || hoveredNode.type === 'imagery')) {
            for (const c of chars) {
                if (eventMatchesCharacter(hoveredNode, c.label)) return c;
            }
        }
        for (const c of chars) {
            if (satellitesForCharacter(c, payload).some((s) => s.id === hoveredId)) return c;
        }
        return null;
    }

    function linkedCharactersForNode(node, chars) {
        if (!node || node.type === 'character') return [];
        const labels = getNodeParents(node);
        return chars.filter((c) => labels.includes(c.label));
    }

    function buildMarkerLayout(payload, hoveredId) {
        const path = getMobiusPath(payload);
        const chars = litCharacters(payload).map((n) => ({
            ...n,
            drawX: n.x,
            drawY: n.y,
            isSatellite: false,
            parentId: null,
        }));
        const layout = [...chars];
        const activeChar = findActiveCharacter(hoveredId, chars, payload);
        const hoveredNode = catalogNodes(payload).find((n) => n.id === hoveredId);

        if (activeChar) {
            const sats = satellitesForCharacter(activeChar, payload);
            sats.forEach((sat, i) => {
                const { drawX, drawY } = layoutSatelliteOnPath(sat, i, sats.length, activeChar, path, payload);
                layout.push({
                    ...sat,
                    drawX,
                    drawY,
                    isSatellite: true,
                    parentId: activeChar.id,
                    linkedCharIds: linkedCharactersForNode(sat, chars).map((c) => c.id),
                });
            });
        } else if (hoveredNode && (hoveredNode.type === 'event' || hoveredNode.type === 'imagery')) {
            const snapped = snapNodeToPath(hoveredNode, payload);
            layout.push({
                ...hoveredNode,
                drawX: snapped.x,
                drawY: snapped.y,
                isSatellite: true,
                parentId: null,
                linkedCharIds: linkedCharactersForNode(hoveredNode, chars).map((c) => c.id),
            });
        }
        return { layout, activeCharId: activeChar ? activeChar.id : null };
    }

    function drawLitMarkers(layout, cssW, cssH, hoveredId, activeCharId) {
        if (!markerCtx || !markerCanvas) return;
        if (markerCanvas.width !== Math.floor(cssW) || markerCanvas.height !== Math.floor(cssH)) {
            markerCanvas.width = Math.max(1, Math.floor(cssW));
            markerCanvas.height = Math.max(1, Math.floor(cssH));
        }
        markerCtx.clearRect(0, 0, markerCanvas.width, markerCanvas.height);
        const w = markerCanvas.width;
        const h = markerCanvas.height;

        layout.forEach((node) => {
            if (!node.isSatellite || !activeCharId) return;
            const parent = layout.find((n) => n.id === activeCharId);
            if (!parent) return;
            const px0 = (parent.drawX ?? parent.x) * w;
            const py0 = (1 - (parent.drawY ?? parent.y)) * h;
            const px1 = node.drawX * w;
            const py1 = (1 - node.drawY) * h;
            const dist = Math.hypot(px1 - px0, py1 - py0);
            if (dist > w * 0.22) return;
            if (!inMainNebulaBand(node.drawX, node.drawY)) return;
            markerCtx.strokeStyle = 'rgba(140, 200, 255, 0.42)';
            markerCtx.lineWidth = 1;
            markerCtx.beginPath();
            markerCtx.moveTo(px0, py0);
            markerCtx.lineTo(px1, py1);
            markerCtx.stroke();
        });

        layout.forEach((node) => {
            const px = node.drawX * w;
            const py = (1 - node.drawY) * h;
            const col = TYPE_COLOR[node.type] || TYPE_COLOR.character;
            const r = node.isSatellite ? 3.5 : 6;
            const hovered = node.id === hoveredId;
            markerCtx.beginPath();
            markerCtx.arc(px, py, hovered ? r + 2 : r, 0, Math.PI * 2);
            markerCtx.fillStyle = `rgba(${Math.round(col[0] * 255)},${Math.round(col[1] * 255)},${Math.round(col[2] * 255)},${node.isSatellite ? 0.75 : 0.88})`;
            markerCtx.fill();
            if (hovered) {
                markerCtx.strokeStyle = 'rgba(255,255,255,0.9)';
                markerCtx.lineWidth = 1.5;
                markerCtx.stroke();
            }
            const showLabel = hovered || !node.isSatellite;
            if (node.label && showLabel) {
                markerCtx.font = node.isSatellite ? '10px "Microsoft YaHei", sans-serif' : '11px "Microsoft YaHei", sans-serif';
                markerCtx.fillStyle = node.isSatellite ? 'rgba(190,220,255,0.9)' : 'rgba(255,252,240,0.92)';
                const label = node.label.length > 16 ? `${node.label.slice(0, 15)}…` : node.label;
                markerCtx.fillText(label, px + r + 3, py + 3);
            }
        });
    }

    function pickFromLayout(layout, clientX, clientY, canvas) {
        const rect = canvas.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) return null;
        const nx = (clientX - rect.left) / rect.width;
        const ny = 1 - (clientY - rect.top) / rect.height;
        let best = null;
        let bestScore = Infinity;
        layout.forEach((node) => {
            const d = Math.hypot(node.drawX - nx, node.drawY - ny);
            const bias = node.isSatellite ? 0 : -0.012;
            const score = d + bias;
            if (score < bestScore && d < 0.075) {
                bestScore = score;
                best = node;
            }
        });
        return best;
    }

    function resolveHoveredNode(clientX, clientY) {
        if (!markerCanvas || !currentPayload) return null;
        const payload = currentPayload;
        const nebulaMode = isNebulaMode(payload);
        const progressForView = payload.ready ? displayProgress : 0;
        const viewPayload = { ...payload, progress: progressForView };
        let { layout } = buildMarkerLayout(viewPayload, null);
        let node = pickFromLayout(layout, clientX, clientY, markerCanvas);
        if (node && node.type === 'character') {
            ({ layout } = buildMarkerLayout(viewPayload, node.id));
            const node2 = pickFromLayout(layout, clientX, clientY, markerCanvas);
            if (node2) node = node2;
        }
        return node;
    }

    function showNodeDetail(node) {
        if (!detailEl) return;
        if (node) {
            const typeLabel = { character: '人物', event: '事件', imagery: '意象' }[node.type] || '';
            const parents = getNodeParents(node);
            const parentHint = parents.length > 1
                ? `<span>关联：${parents.join(' · ')}</span>`
                : `<span>${typeLabel}</span>`;
            detailEl.innerHTML = `<strong>${node.label}</strong>${parentHint}`;
            detailEl.classList.add('visible');
        } else {
            detailEl.classList.remove('visible');
        }
    }

    function bindMarkerPointerEvents() {
        if (pointerBound || !markerCanvas) return;
        pointerBound = true;
        const onMove = (ev) => {
            const node = resolveHoveredNode(ev.clientX, ev.clientY);
            hoveredNodeId = node ? node.id : null;
            markerCanvas.style.cursor = node ? 'pointer' : 'default';
            showNodeDetail(node);
        };
        markerCanvas.addEventListener('mousemove', onMove);
        markerCanvas.addEventListener('mouseleave', () => {
            hoveredNodeId = null;
            markerCanvas.style.cursor = 'default';
            showNodeDetail(null);
        });
        markerCanvas.addEventListener('click', (ev) => {
            showNodeDetail(resolveHoveredNode(ev.clientX, ev.clientY));
        });
    }

    function mergeStarMapPayload(catalog, sessionPayload) {
        const session = sessionPayload || {};
        const ready = !!session.ready;
        const base = { ...(catalog || {}), ...session };
        base.ready = ready;
        base.progress = ready ? Math.max(0, Math.min(1, session.progress || 0)) : 0;
        base.lit_node_ids = ready ? (session.lit_node_ids || []) : [];
        base.lit_nodes = ready ? (session.lit_nodes || []) : [];
        base.lit_count = ready ? (session.lit_count || 0) : 0;
        if (ready && base.lit_nodes.length) {
            const idSet = new Set(base.lit_node_ids);
            base.lit_nodes = base.lit_nodes.map((n) => ({
                ...n,
                type: n.type || (String(n.id || '').startsWith('char:') ? 'character'
                    : String(n.id || '').startsWith('event:') ? 'event' : 'imagery'),
            })).filter((n) => idSet.has(n.id));
        }
        base.constellation_lines = ready ? (session.constellation_lines || []) : [];
        base.mobius_arc_points = ready ? (session.mobius_arc_points || []) : [];
        base.mobius_full_points = session.mobius_full_points
            || (session.mobius && session.mobius.path)
            || (catalog && catalog.mobius && catalog.mobius.path)
            || [];
        return base;
    }

    function buildStarVao(litNodes) {
        const key = litNodes.map((n) => n.id).join(',');
        if (key === starLitKey && starCount) return starCount;
        starLitKey = key;
        const n = litNodes.length;
        if (!n) return 0;
        const stride = 8;
        const data = new Float32Array(n * stride);
        litNodes.forEach((node, i) => {
            const o = i * stride;
            const c = TYPE_COLOR[node.type] || TYPE_COLOR.character;
            data[o + 0] = node.x;
            data[o + 1] = node.y;
            data[o + 2] = node.magnitude || 5;
            data[o + 3] = Math.max(node.importance || 0.5, 0.55);
            data[o + 4] = c[0];
            data[o + 5] = c[1];
            data[o + 6] = c[2];
            data[o + 7] = 0;
        });

        const buf = gl.createBuffer();
        gl.bindVertexArray(starVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

        const attrs = [
            ['aPos', 2, 0],
            ['aMag', 1, 2],
            ['aImp', 1, 3],
            ['aColor', 3, 4],
            ['aPhase', 1, 7],
        ];
        attrs.forEach(([name, size, off]) => {
            const loc = gl.getAttribLocation(progStar, name);
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride * 4, off * 4);
        });
        gl.bindVertexArray(null);
        starCount = n;
        return n;
    }

    function drawPolyline(points, color, alpha, vw, vh) {
        if (!points || points.length < 2) return;
        gl.useProgram(progLine);
        gl.uniform2f(gl.getUniformLocation(progLine, 'uRes'), vw, vh);
        gl.uniform4f(gl.getUniformLocation(progLine, 'uColor'), color[0], color[1], color[2], alpha);
        const flat = new Float32Array(points.length * 2);
        points.forEach((p, i) => { flat[i * 2] = p[0]; flat[i * 2 + 1] = p[1]; });
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, flat, gl.STREAM_DRAW);
        const loc = gl.getAttribLocation(progLine, 'aPos');
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINE_STRIP, 0, points.length);
    }

    function allLitNodes(payload) {
        const litSet = new Set(payload.lit_node_ids || []);
        return (payload.lit_nodes || []).filter((n) => litSet.has(n.id));
    }

    function visibleLitNodes(payload, nebulaMode) {
        const nodes = allLitNodes(payload);
        if (!nebulaMode) return nodes;
        const path = payload.mobius_full_points
            || (payload.mobius && payload.mobius.path)
            || (catalogCache && catalogCache.mobius && catalogCache.mobius.path)
            || [];
        const cap = pathRevealIndex(payload.progress || 0, path.length);
        return nodes.filter((n) => (n.path_index != null ? n.path_index : 0) <= cap);
    }

    function frame() {
        if (!gl || !canvasEl) return;
        const dpr = window.devicePixelRatio || 1;
        const cssW = canvasEl.clientWidth;
        const cssH = canvasEl.clientHeight;
        if (cssW < 10 || cssH < 10) {
            animId = requestAnimationFrame(frame);
            return;
        }
        const vw = cssW * dpr;
        const vh = cssH * dpr;
        if (canvasEl.width !== vw || canvasEl.height !== vh) {
            canvasEl.width = vw;
            canvasEl.height = vh;
        }
        gl.viewport(0, 0, vw, vh);

        const t = (performance.now() - startTime) / 1000;
        const reveal = Math.min(1, t / 0.8);
        const fade = Math.min(1, t / 0.5);

        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        const payload = currentPayload || {};
        const nebulaMode = isNebulaMode(payload);
        if (Math.abs(displayProgress - targetProgress) > 0.001) {
            displayProgress += (targetProgress - displayProgress) * 0.12;
        } else {
            displayProgress = targetProgress;
        }
        const progressForView = payload.ready ? displayProgress : 0;
        const viewPayload = { ...payload, progress: progressForView };
        const litOnArc = visibleLitNodes(viewPayload, nebulaMode);
        const { layout, activeCharId } = buildMarkerLayout(viewPayload, hoveredNodeId);
        const path = getMobiusPath(viewPayload);

        if (nebulaMode) {
            updateNebulaReveal(path, progressForView, cssW, cssH);
        }

        ensureMarkerCanvas(wrapEl);
        if (markerCanvas) {
            markerCanvas.width = Math.max(1, Math.floor(cssW));
            markerCanvas.height = Math.max(1, Math.floor(cssH));
        }
        drawLitMarkers(layout, cssW, cssH, hoveredNodeId, activeCharId);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        if (!nebulaMode) {
            drawPolyline(payload.mobius_full_points, [0.48, 0.38, 0.78], 0.42 * fade, vw, vh);
            drawPolyline(payload.mobius_arc_points, [0.68, 0.55, 0.98], 0.88 * fade, vw, vh);
        }

        /* hover 连线仅由 marker 层绘制；星云模式下不画跨环星座线 */
        if (hoveredNodeId && !activeCharId && !nebulaMode) {
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
            (payload.constellation_lines || []).forEach((ln) => {
                if (ln.from !== hoveredNodeId && ln.to !== hoveredNodeId) return;
                drawPolyline(ln.points, [0.72, 0.86, 1.0], 0.55 * fade, vw, vh);
            });
        }

        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
        const count = buildStarVao(litOnArc);
        if (count > 0) {
            gl.useProgram(progStar);
            gl.bindVertexArray(starVao);
            gl.uniform2f(gl.getUniformLocation(progStar, 'uRes'), vw, vh);
            gl.uniform1f(gl.getUniformLocation(progStar, 'uReveal'), reveal);
            gl.drawArrays(gl.POINTS, 0, count);
            gl.bindVertexArray(null);
        }

        if (detailEl && payload.ready) {
            detailEl.dataset.progress = String(Math.round(progressForView * 100));
        }
        animId = requestAnimationFrame(frame);
    }

    function pickNode(payload, clientX, clientY) {
        if (!markerCanvas) return null;
        const { layout } = buildMarkerLayout(payload, hoveredNodeId);
        return pickFromLayout(layout, clientX, clientY, markerCanvas);
    }

    async function preloadCatalog(apiBase) {
        if (catalogCache) return catalogCache;
        const resp = await fetch((apiBase || '') + '/api/observation/catalog');
        if (resp.ok) catalogCache = await resp.json();
        return catalogCache;
    }

    async function renderStarMap(canvas, sessionPayload, apiBase) {
        canvasEl = canvas;
        wrapEl = canvas.closest('.star-map-canvas-wrap');
        detailEl = document.getElementById('starMapDetailPanel');
        hoveredNodeId = null;
        if (animId) cancelAnimationFrame(animId);

        const catalog = await preloadCatalog(apiBase);
        const merged = mergeStarMapPayload(catalog, sessionPayload);
        if (sessionPayload && sessionPayload.nodes && sessionPayload.nodes.length) {
            catalogCache = { ...(catalogCache || {}), nodes: sessionPayload.nodes };
        }
        const nebulaMode = isNebulaMode(merged);
        const bgUrl = merged.background_url
            || (merged.background && merged.background.url)
            || '/static/images/exploration_nebula_bg.webp';

        const cssW = canvas.clientWidth || 640;
        const cssH = canvas.clientHeight || 360;
        setBackground(wrapEl, bgUrl, nebulaMode, cssW, cssH);

        if (!gl) initGl(canvas);

        ensureMarkerCanvas(wrapEl);
        if (markerCanvas) {
            markerCanvas.width = Math.max(1, Math.floor(cssW));
            markerCanvas.height = Math.max(1, Math.floor(cssH));
        }

        targetProgress = merged.progress || 0;
        displayProgress = 0;
        currentPayload = merged;
        starLitKey = '';
        startTime = performance.now();

        bindMarkerPointerEvents();

        frame();
        return merged;
    }

    async function refreshStarMap(sessionId, apiBase) {
        if (!canvasEl || !sessionId) return null;
        const resp = await fetch(
            (apiBase || '') + `/api/observation/${encodeURIComponent(sessionId)}/starmap`,
            { credentials: 'include', headers: (typeof getAuthHeaders === 'function' ? getAuthHeaders() : {}) },
        );
        if (!resp.ok) return null;
        const sessionPayload = await resp.json();
        const merged = mergeStarMapPayload(catalogCache, sessionPayload);
        if (sessionPayload && sessionPayload.nodes && sessionPayload.nodes.length) {
            catalogCache = { ...(catalogCache || {}), ...sessionPayload, nodes: sessionPayload.nodes };
        }
        targetProgress = merged.progress || 0;
        currentPayload = merged;
        starLitKey = '';
        return merged;
    }

    function disposeStarMap() {
        if (animId) cancelAnimationFrame(animId);
        animId = null;
        currentPayload = null;
        hoveredNodeId = null;
        pointerBound = false;
    }

    global.preloadStarMapCatalog = preloadCatalog;
    global.renderStarMap = renderStarMap;
    global.refreshStarMap = refreshStarMap;
    global.disposeStarMap = disposeStarMap;
})(typeof window !== 'undefined' ? window : globalThis);
