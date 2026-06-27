/**
 * 探索星图：银河背景 + 物理感光照（1/r² 衰减），点亮即周围变亮
 */
(function (global) {
    const BG_URL = '/static/images/galaxy_bg.jpg';
    let bgImage = null;
    let bgLoading = null;

    const TYPE_RGB = {
        character: [255, 220, 160],
        event: [120, 190, 255],
        imagery: [190, 140, 255],
    };

    function loadBackground(url) {
        const src = url || BG_URL;
        if (bgImage && bgImage.src.indexOf('galaxy_bg') >= 0) {
            return Promise.resolve(bgImage);
        }
        if (bgLoading) return bgLoading;
        bgLoading = new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => { bgImage = img; resolve(img); };
            img.onerror = reject;
            img.src = src;
        });
        return bgLoading;
    }

    /** 光照强度：importance × 类型权重，点亮时才有值 */
    function lightIntensity(node) {
        if (!node.lit) return 0;
        const imp = node.importance != null ? node.importance : 0.5;
        const typeScale = { character: 1.0, event: 0.72, imagery: 0.55 };
        return Math.min(0.35 + imp * (typeScale[node.type] || 0.6) * 0.75, 1);
    }

    /** 多层径向叠加，近似点光源 1/r² 衰减（无十字星芒） */
    function drawPointLight(ctx, x, y, intensity, rgb, maxRadius) {
        if (intensity <= 0) return;
        const [r, g, b] = rgb;
        const steps = 14;
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        for (let i = steps; i >= 1; i--) {
            const t = i / steps;
            const radius = maxRadius * t;
            const alpha = (intensity * 0.11) / (t * t + 0.06);
            const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
            grad.addColorStop(0, `rgba(${r},${g},${b},${Math.min(alpha * 1.4, 0.85)})`);
            grad.addColorStop(0.35, `rgba(${r},${g},${b},${alpha * 0.4})`);
            grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();
    }

    function drawCore(ctx, x, y, intensity, rgb, coreR) {
        const [r, g, b] = rgb;
        const cr = coreR * (0.7 + intensity * 0.5);
        const grad = ctx.createRadialGradient(x, y, 0, x, y, cr);
        grad.addColorStop(0, `rgba(255,255,255,${0.55 + intensity * 0.45})`);
        grad.addColorStop(0.45, `rgba(${r},${g},${b},${0.85})`);
        grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, cr, 0, Math.PI * 2);
        ctx.fill();
    }

    function truncateLabel(text, maxLen) {
        if (!text) return '';
        const t = String(text).trim();
        if (t.length <= maxLen) return t;
        return t.slice(0, maxLen - 1) + '…';
    }

    function drawNodeLabel(ctx, x, y, label, type, glowR) {
        if (!label) return;
        const maxLen = type === 'character' ? 6 : type === 'event' ? 10 : 8;
        const text = truncateLabel(label, maxLen);
        const fontSize = type === 'character' ? 11 : 9;
        const offsetY = glowR + (type === 'character' ? 10 : 8);
        ctx.save();
        ctx.font = `${type === 'character' ? 600 : 500} ${fontSize}px "Segoe UI", "Microsoft YaHei", sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.shadowColor = 'rgba(0,0,0,0.95)';
        ctx.shadowBlur = 5;
        ctx.fillStyle = type === 'imagery'
            ? 'rgba(220, 190, 255, 0.92)'
            : type === 'event'
                ? 'rgba(180, 220, 255, 0.92)'
                : 'rgba(255, 255, 255, 0.95)';
        ctx.fillText(text, x, y - offsetY);
        ctx.restore();
    }

    function drawLitNode(ctx, x, y, node) {
        const type = node.type || 'character';
        const rgb = TYPE_RGB[type] || TYPE_RGB.character;
        const I = lightIntensity(node);
        const imp = node.importance != null ? node.importance : 0.5;
        const baseR = node.star_radius || (2 + imp * 3);

        const maxGlow = type === 'character'
            ? 18 + I * 42
            : type === 'event'
                ? 10 + I * 28
                : 8 + I * 22;

        drawPointLight(ctx, x, y, I, rgb, maxGlow);
        drawCore(ctx, x, y, I, rgb, baseR * 0.55);
        drawNodeLabel(ctx, x, y, node.label, type, maxGlow * 0.35);
    }

    function drawDimNode(ctx, x, y, node) {
        const type = node.type || 'character';
        if (type === 'imagery') return;
        ctx.globalAlpha = 0.05;
        ctx.beginPath();
        ctx.arc(x, y, type === 'character' ? 0.8 : 0.45, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.globalAlpha = 1;
    }

    function drawOrbit(ctx, x1, y1, x2, y2, lit) {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = lit ? 'rgba(100, 180, 255, 0.18)' : 'rgba(255,255,255,0.025)';
        ctx.lineWidth = lit ? 0.6 : 0.35;
        ctx.stroke();
    }

    async function drawStarMap(canvas, mapData) {
        if (!canvas || !mapData) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth || canvas.width || 440;
        const h = canvas.clientHeight || canvas.height || 300;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        try {
            const img = await loadBackground(mapData.background_url || BG_URL);
            const scale = Math.max(w / img.width, h / img.height);
            const iw = img.width * scale;
            const ih = img.height * scale;
            ctx.drawImage(img, (w - iw) / 2, (h - ih) / 2, iw, ih);
        } catch (e) {
            const grd = ctx.createRadialGradient(w * 0.5, h * 0.5, 10, w * 0.5, h * 0.5, w * 0.55);
            grd.addColorStop(0, '#1a1035');
            grd.addColorStop(1, '#050510');
            ctx.fillStyle = grd;
            ctx.fillRect(0, 0, w, h);
        }

        ctx.fillStyle = 'rgba(0,0,0,0.58)';
        ctx.fillRect(0, 0, w, h);

        const pad = 12;
        const toX = (x) => pad + x * (w - pad * 2);
        const toY = (y) => pad + y * (h - pad * 2);
        const nodes = mapData.nodes || [];
        const nodeById = {};
        nodes.forEach((n) => { nodeById[n.id] = n; });

        (mapData.edges || []).forEach((e) => {
            const a = nodeById[e.from];
            const b = nodeById[e.to];
            if (!a || !b) return;
            drawOrbit(ctx, toX(a.x), toY(a.y), toX(b.x), toY(b.y), a.lit && b.lit);
        });

        nodes.filter((n) => !n.lit).forEach((n) => {
            drawDimNode(ctx, toX(n.x), toY(n.y), n);
        });

        // 先画光晕（暗→亮），再画核心与文字
        const litNodes = nodes.filter((n) => n.lit);
        litNodes.sort((a, b) => lightIntensity(a) - lightIntensity(b));
        litNodes.forEach((n) => {
            drawLitNode(ctx, toX(n.x), toY(n.y), n);
        });
    }

    global.renderStarMap = function (canvas, mapData) {
        drawStarMap(canvas, mapData).catch(() => {});
    };
})(typeof window !== 'undefined' ? window : globalThis);
