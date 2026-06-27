// DOM元素引用
let chatContainer, messageInput, sendButton, emptyState, historySidebar, overlay, historyContent;
let historyButton, closeButton, newChatButton;
// =========================
// 对话树模型（支持编辑分叉/版本切换）
// =========================
// node: { id, role, content, parentId, children: [], activeChildId, createdAt }
// userGroup: { id, nodeIds: [], activeNodeId }
let conversationTree = null; // { id, nodesById, rootId, leafId, userGroupsById, nodeToUserGroupId }
let currentPathNodeIds = []; // 从 root 到 leaf 的路径（用于渲染/构建conversation_history）
let isProcessing = false;
let currentConversationId = null; // 当前加载的对话ID

// 历史对话存储键名
const HISTORY_STORAGE_KEY = 'novel_rag_chat_history';
const ANON_USER_ID_KEY = 'novel_rag_anon_user_id';
const OBSERVATION_SESSION_KEY = 'novel_rag_observation_session_id';
const OBSERVATION_TURN_COUNT_KEY = 'novel_rag_observation_turn_count';
const OBSERVATION_TRIGGER_EVERY_N = 10;

let observationSessionId = null;
let observationTurnCount = 0;
let currentUserProfile = null;
// API 基地址：同源留空；部署到 GitHub Pages 时在 HTML 中设置 window.__API_BASE__ 为后端地址
function getApiBase() { return (typeof window !== 'undefined' && window.__API_BASE__) ? window.__API_BASE__.replace(/\/$/, '') : ''; }

function ensureAnonymousUserId() {
    let id = localStorage.getItem(ANON_USER_ID_KEY);
    if (!id) {
        id = `anon_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
        localStorage.setItem(ANON_USER_ID_KEY, id);
    }
    return id;
}

function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (!currentUserProfile || !currentUserProfile.authenticated) {
        headers['X-Anonymous-User-Id'] = ensureAnonymousUserId().replace(/^anon_/, '');
    }
    return headers;
}

function loadObservationSessionFromStorage() {
    observationSessionId = localStorage.getItem(OBSERVATION_SESSION_KEY);
    observationTurnCount = parseInt(localStorage.getItem(OBSERVATION_TURN_COUNT_KEY) || '0', 10) || 0;
}

function persistObservationSession() {
    if (observationSessionId) {
        localStorage.setItem(OBSERVATION_SESSION_KEY, observationSessionId);
    }
    localStorage.setItem(OBSERVATION_TURN_COUNT_KEY, String(observationTurnCount));
}

async function ensureObservationSession() {
    if (observationSessionId) return observationSessionId;
    try {
        const resp = await fetch(getApiBase() + '/api/observation/session', {
            method: 'POST',
            headers: getAuthHeaders(),
            credentials: 'include',
            body: JSON.stringify({ conversation_id: currentConversationId || null })
        });
        if (resp.ok) {
            const data = await resp.json();
            observationSessionId = data.session_id;
            persistObservationSession();
        }
    } catch (e) {
        // 观测会话创建失败不影响主对话
    }
    return observationSessionId;
}

async function refreshAuthProfile() {
    try {
        const resp = await fetch(getApiBase() + '/api/auth/me', {
            credentials: 'include',
            headers: getAuthHeaders()
        });
        if (resp.ok) {
            currentUserProfile = await resp.json();
            updateAuthUI();
        }
    } catch (e) {
        currentUserProfile = { authenticated: false, auth_provider: 'anonymous' };
    }
}

function updateAuthUI() {
    const badge = document.getElementById('userBadge');
    const authBtn = document.getElementById('authButton');
    if (!badge || !authBtn) return;

    if (currentUserProfile && currentUserProfile.authenticated) {
        badge.textContent = currentUserProfile.display_name || currentUserProfile.email || '已登录';
        authBtn.style.display = 'inline-block';
        authBtn.textContent = '退出';
        authBtn.onclick = () => {
            window.location.href = getApiBase() + '/auth/logout';
        };
    } else if (currentUserProfile && currentUserProfile.google_oauth_enabled) {
        badge.textContent = '访客（未登录）';
        authBtn.style.display = 'inline-block';
        authBtn.textContent = 'Google 登录';
        authBtn.onclick = () => {
            window.location.href = getApiBase() + '/auth/login';
        };
    } else {
        badge.textContent = '本地访客';
        authBtn.style.display = 'none';
    }
}

function toggleTrajectorySidebar() {
    const sidebar = document.getElementById('trajectorySidebar');
    if (!sidebar) return;
    sidebar.classList.toggle('open');
    if (sidebar.classList.contains('open')) {
        loadObservationTrajectory();
    }
}

function formatTrajectoryUpdatedAt(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getMonth() + 1}月${d.getDate()}日 ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function isStaleTrajectoryData(data) {
    if (!data) return false;
    const s = data.visible_summary || '';
    if (/展开了\s*\d+\s*轮|待继续深入|么故事|键节点/.test(s)) return true;
    const md = data.exploration_coordinates_md || '';
    if (/第7章沉默|第七集沉默|逃避责任|当时在场其他人|沉默作为逃避|倾向「逃避责任」/.test(md)) return true;
    return false;
}

function renderTrajectoryPanel(data) {
    const container = document.getElementById('trajectoryContent');
    if (!container) return;
    if (!data || !data.ready) {
        container.innerHTML = '<p class="trajectory-empty">尚无阅读轨迹。继续对话，或点击「重新生成轨迹」。</p>';
        return;
    }
    const coordsMd = data.exploration_coordinates_md || '';
    const staleHint = isStaleTrajectoryData(data)
        ? '<p class="trajectory-stale">检测到旧版或模板化轨迹，请点击「重新生成轨迹」。</p>'
        : '';
    const updatedLabel = formatTrajectoryUpdatedAt(data.updated_at);
    const footer = updatedLabel
        ? `<p class="trajectory-empty trajectory-updated">更新 ${escapeHtml(updatedLabel)}</p>`
        : '';
    container.innerHTML = `
        <div class="trajectory-section">
            <h3>阅读旅程</h3>
            <p class="trajectory-journey">${escapeHtml(data.visible_summary || '')}</p>
        </div>
        <div class="trajectory-section interest-block">
            <h3>当前探索坐标</h3>
            ${staleHint}
            <div class="markdown-body trajectory-md">${renderMarkdown(coordsMd)}</div>
        </div>
        ${footer}
    `;
}

async function openStarMapFullPage() {
    await ensureObservationSession();
    const page = document.getElementById('starMapFullPage');
    const canvas = document.getElementById('starMapFullCanvas');
    const legend = document.getElementById('starMapFullLegend');
    const detail = document.getElementById('starMapDetailPanel');
    if (!page || !canvas) return;
    page.classList.add('open');
    if (legend) legend.textContent = '加载星图…';
    if (detail) detail.classList.remove('visible');
    try {
        if (typeof preloadStarMapCatalog === 'function') {
            preloadStarMapCatalog(getApiBase()).catch(() => {});
        }
        const resp = await fetch(getApiBase() + `/api/observation/${encodeURIComponent(observationSessionId)}/starmap`, {
            credentials: 'include',
            headers: getAuthHeaders()
        });
        if (!resp.ok) {
            if (legend) legend.textContent = '星图加载失败';
            return;
        }
        const mapData = await resp.json();
        if (typeof renderStarMap === 'function') {
            await renderStarMap(canvas, mapData, getApiBase());
        }
        if (legend) {
            const pct = Math.round((mapData.progress || 0) * 100);
            legend.textContent = mapData.ready
                ? `已点亮 ${mapData.lit_count || 0} 颗 · 探索 ${pct}%`
                : '尚无探索记录，继续对话后重新生成轨迹';
        }
    } catch (e) {
        if (legend) legend.textContent = '星图加载失败';
    }
}

function closeStarMapFullPage() {
    const page = document.getElementById('starMapFullPage');
    if (page) page.classList.remove('open');
    if (typeof disposeStarMap === 'function') disposeStarMap();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

async function loadObservationTrajectory() {
    if (!observationSessionId) {
        renderTrajectoryPanel(null);
        return;
    }
    try {
        const resp = await fetch(getApiBase() + `/api/observation/${encodeURIComponent(observationSessionId)}`, {
            credentials: 'include',
            headers: getAuthHeaders()
        });
        if (resp.ok) {
            renderTrajectoryPanel(await resp.json());
        }
    } catch (e) {
        renderTrajectoryPanel(null);
    }
}

async function triggerObservationExtract(showPanel = false) {
    await ensureObservationSession();
    if (!observationSessionId) return false;
    const btn = document.getElementById('regenerateTrajectoryButton');
    if (btn) { btn.disabled = true; btn.textContent = '生成中…'; }
    try {
        const resp = await fetch(getApiBase() + '/api/observation/trigger', {
            method: 'POST',
            credentials: 'include',
            headers: getAuthHeaders(),
            body: JSON.stringify({ session_id: observationSessionId })
        });
        if (resp.ok) {
            await loadObservationTrajectory();
            const page = document.getElementById('starMapFullPage');
            if (page && page.classList.contains('open') && typeof refreshStarMap === 'function') {
                const mapData = await refreshStarMap(observationSessionId, getApiBase());
                const legend = document.getElementById('starMapFullLegend');
                if (legend && mapData) {
                    const pct = Math.round((mapData.progress || 0) * 100);
                    legend.textContent = mapData.ready
                        ? `已点亮 ${mapData.lit_count || 0} 颗 · 探索 ${pct}%`
                        : '尚无探索记录，继续对话后重新生成轨迹';
                }
            }
            if (showPanel) {
                const sidebar = document.getElementById('trajectorySidebar');
                if (sidebar && !sidebar.classList.contains('open')) {
                    sidebar.classList.add('open');
                }
            }
            const data = await resp.json();
            return !!(data && data.ready && data.visible_summary);
        }
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '重新生成轨迹'; }
    }
    return false;
}

async function pollObservationUntilReady(maxAttempts = 15, intervalMs = 2000) {
    for (let i = 0; i < maxAttempts; i++) {
        await loadObservationTrajectory();
        const container = document.getElementById('trajectoryContent');
        const hasSummary = container && container.textContent && !container.textContent.includes('尚无阅读轨迹');
        if (hasSummary) {
            const sidebar = document.getElementById('trajectorySidebar');
            if (sidebar && !sidebar.classList.contains('open')) {
                sidebar.classList.add('open');
            }
            return true;
        }
        await new Promise(r => setTimeout(r, intervalMs));
    }
    return false;
}

async function maybeAutoTriggerObservation() {
    if (observationTurnCount > 0 && observationTurnCount % OBSERVATION_TRIGGER_EVERY_N === 0) {
        await triggerObservationExtract(true);
    }
}

function onObservationScheduledFromServer(turn) {
    if (typeof turn === 'number') {
        observationTurnCount = Math.max(observationTurnCount, turn);
        persistObservationSession();
    }
    pollObservationUntilReady();
}

function onAssistantTurnComplete(finalData) {
    if (finalData && finalData.observation_session_id) {
        observationSessionId = finalData.observation_session_id;
        persistObservationSession();
    }
    if (finalData && typeof finalData.observation_turn === 'number') {
        observationTurnCount = finalData.observation_turn;
    } else {
        observationTurnCount += 1;
    }
    persistObservationSession();
    // 抽取由服务端在 turn=N 时后台触发；前端仅轮询结果（见 observation SSE）
}
// 将“会话内容（对话树）”按会话ID单独存储，历史列表仅存元信息，加载更快
const CONVERSATION_STORAGE_PREFIX = 'novel_rag_conversation_tree_';
// 服务端共享历史缓存（所有用户可见，不写盘；重启清空）
let sharedHistoryCache = [];
// 本机隐藏的共享对话 ID（仅不在此设备侧栏显示，不删服务端）
const hiddenSharedIds = new Set();

function conversationTreeStorageKey(conversationId) {
    return `${CONVERSATION_STORAGE_PREFIX}${conversationId}`;
}

function persistConversationTree(conversationId, tree, currentPath) {
    if (!conversationId || !tree) return;
    const payload = { tree, current_path: currentPath, savedAt: nowIso() };
    localStorage.setItem(conversationTreeStorageKey(conversationId), JSON.stringify(payload));
}

function loadPersistedConversationTree(conversationId) {
    const raw = localStorage.getItem(conversationTreeStorageKey(conversationId));
    if (!raw) return null;
    try {
        return JSON.parse(raw);
    } catch (e) {
        // 已删除日志
        return null;
    }
}

function deletePersistedConversationTree(conversationId) {
    localStorage.removeItem(conversationTreeStorageKey(conversationId));
}

function nowIso() {
    return new Date().toISOString();
}

function genId(prefix) {
    return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function ensureTreeInitialized() {
    if (conversationTree) return;
    const rootId = genId('root');
    conversationTree = {
        id: currentConversationId || Date.now().toString(),
        nodesById: {
            [rootId]: { id: rootId, role: 'system', content: '', parentId: null, children: [], activeChildId: null, createdAt: nowIso() }
        },
        rootId,
        leafId: rootId,
        userGroupsById: {},
        nodeToUserGroupId: {}
    };
    currentPathNodeIds = [rootId];
}

function rebuildCurrentPathFromRoot() {
    if (!conversationTree) return;
    const path = [];
    let cur = conversationTree.rootId;
    path.push(cur);
    while (true) {
        const node = conversationTree.nodesById[cur];
        const next = node && node.activeChildId ? node.activeChildId : null;
        if (!next) break;
        path.push(next);
        cur = next;
    }
    currentPathNodeIds = path;
    conversationTree.leafId = cur;
}

function buildConversationHistoryForBackend(excludeLeafIfUserWillBeSent = false) {
    // 生成后端需要的 conversation_history（不包含当前要发送的 message）
    // 当前约定：后端接口会收到 request.message + conversation_history（不含当前message）
    if (!conversationTree) return [];
    const ids = currentPathNodeIds.slice(1); // 跳过root
    const msgs = ids.map(id => {
        const n = conversationTree.nodesById[id];
        return { role: n.role, content: n.content };
    });
    if (excludeLeafIfUserWillBeSent && msgs.length > 0) {
        const last = msgs[msgs.length - 1];
        if (last.role === 'user') {
            return msgs.slice(0, -1);
        }
    }
    return msgs;
}

function getUserGroupForNode(nodeId) {
    const gid = conversationTree.nodeToUserGroupId[nodeId];
    return gid ? conversationTree.userGroupsById[gid] : null;
}

function getUserGroupVariantIndex(group, nodeId) {
    if (!group) return 0;
    const idx = group.nodeIds.indexOf(nodeId);
    return idx >= 0 ? idx : 0;
}

function switchUserVariant(groupId, direction) {
    if (!conversationTree) return;
    const group = conversationTree.userGroupsById[groupId];
    if (!group || group.nodeIds.length <= 1) return;
    const curIdx = group.nodeIds.indexOf(group.activeNodeId);
    const nextIdx = Math.max(0, Math.min(group.nodeIds.length - 1, curIdx + direction));
    group.activeNodeId = group.nodeIds[nextIdx];

    // 关键：该用户节点的父节点要把 activeChildId 指向选中的 variant 节点
    const userNode = conversationTree.nodesById[group.activeNodeId];
    const parent = conversationTree.nodesById[userNode.parentId];
    if (parent) {
        parent.activeChildId = userNode.id;
    }

    // 重新沿 activeChildId 回放路径并重绘
    rebuildCurrentPathFromRoot();
    rerenderFromCurrentPath();
    saveConversation();
}

function addNode(role, content, parentId) {
    ensureTreeInitialized();
    const id = genId(role);
    conversationTree.nodesById[id] = { id, role, content, parentId, children: [], activeChildId: null, createdAt: nowIso() };
    const parent = conversationTree.nodesById[parentId];
    if (parent) {
        parent.children.push(id);
        parent.activeChildId = id; // 默认沿新分支继续
    }
    conversationTree.leafId = id;
    rebuildCurrentPathFromRoot();
    return id;
}

function addUserNodeWithGroup(content, parentId, existingGroupId = null) {
    ensureTreeInitialized();
    const nodeId = addNode('user', content, parentId);
    let groupId = existingGroupId;
    if (!groupId) {
        groupId = genId('ug');
        conversationTree.userGroupsById[groupId] = { id: groupId, nodeIds: [], activeNodeId: nodeId };
    }
    const group = conversationTree.userGroupsById[groupId];
    group.nodeIds.push(nodeId);
    group.activeNodeId = nodeId;
    conversationTree.nodeToUserGroupId[nodeId] = groupId;
    return { nodeId, groupId };
}

function convertLegacyMessagesToTreeIfNeeded(messages) {
    // 兼容旧格式：[{role,content},...] 线性对话转成单一路径树
    if (!Array.isArray(messages) || messages.length === 0) return;
    ensureTreeInitialized();
    let parent = conversationTree.rootId;
    for (const m of messages) {
        if (!m || !m.role) continue;
        if (m.role === 'user') {
            const { nodeId } = addUserNodeWithGroup(m.content || '', parent);
            parent = nodeId;
        } else if (m.role === 'assistant') {
            const nodeId = addNode('assistant', m.content || '', parent);
            parent = nodeId;
        }
    }
    rebuildCurrentPathFromRoot();
}

// 初始化DOM元素引用
function initElements() {
    chatContainer = document.getElementById('chatContainer');
    messageInput = document.getElementById('messageInput');
    sendButton = document.getElementById('sendButton');
    emptyState = document.getElementById('emptyState');
    historySidebar = document.getElementById('historySidebar');
    overlay = document.getElementById('overlay');
    historyContent = document.getElementById('historyContent');
    historyButton = document.getElementById('historyButton');
    closeButton = document.getElementById('closeButton');
    newChatButton = document.getElementById('newChatButton');
    
    // 检查关键元素是否存在
    if (!chatContainer || !messageInput || !sendButton) {
        // 已删除日志
        return false;
    }
    return true;
}

// 配置Marked.js
function initMarkdown() {
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,  // 支持换行
            gfm: true,    // 支持GitHub风格的Markdown
            highlight: function(code, lang) {
                if (typeof hljs !== 'undefined' && lang) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (e) {
                        return hljs.highlightAuto(code).value;
                    }
                }
                return code;
            }
        });
    }
}

// Markdown渲染函数
function renderMarkdown(text) {
    if (!text) return '';
    
    if (typeof marked !== 'undefined') {
        try {
            return marked.parse(text);
        } catch (e) {
            // 已删除日志
            // 如果渲染失败，转义HTML并返回
            return escapeHtml(text).replace(/\n/g, '<br>');
        }
    } else {
        // 如果没有marked库，简单处理换行和转义
        return escapeHtml(text).replace(/\n/g, '<br>');
    }
}

// HTML转义函数
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 将Markdown文本转换为纯文本（去除格式符号）
function markdownToPlainText(markdown) {
    if (!markdown) return '';
    
    let text = markdown;
    
    // 移除代码块（```code```）
    text = text.replace(/```[\s\S]*?```/g, '');
    
    // 移除行内代码（`code`）
    text = text.replace(/`[^`]*`/g, '');
    
    // 移除标题标记（# ## ###等）
    text = text.replace(/^#{1,6}\s+/gm, '');
    
    // 移除粗体标记（**text** 或 __text__）
    text = text.replace(/\*\*([^*]+)\*\*/g, '$1');
    text = text.replace(/__([^_]+)__/g, '$1');
    
    // 移除斜体标记（*text* 或 _text_）
    text = text.replace(/\*([^*]+)\*/g, '$1');
    text = text.replace(/_([^_]+)_/g, '$1');
    
    // 移除删除线标记（~~text~~）
    text = text.replace(/~~([^~]+)~~/g, '$1');
    
    // 移除链接标记（[text](url)）
    text = text.replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1');
    
    // 移除图片标记（![alt](url)）
    text = text.replace(/!\[([^\]]*)\]\([^\)]+\)/g, '$1');
    
    // 移除引用标记（> text）
    text = text.replace(/^>\s+/gm, '');
    
    // 移除列表标记（- * + 或数字.）
    text = text.replace(/^[\s]*[-*+]\s+/gm, '');
    text = text.replace(/^\s*\d+\.\s+/gm, '');
    
    // 移除水平线（--- 或 ***）
    text = text.replace(/^[-*]{3,}$/gm, '');
    
    // 清理多余的空行（将多个连续空行替换为两个空行）
    text = text.replace(/\n{3,}/g, '\n\n');
    
    // 移除首尾空白
    text = text.trim();
    
    return text;
}

// 自动调整输入框高度
function setupInputAutoResize() {
    if (messageInput) {
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });
    }
}

// 历史对话功能
function toggleHistory() {
    try {
        if (!historySidebar || !overlay) {
            // 已删除日志
            return;
        }
        historySidebar.classList.toggle('open');
        overlay.classList.toggle('show');
        if (historySidebar.classList.contains('open')) {
            loadHistoryList();
        }
    } catch (error) {
        // 已删除日志
    }
}

function saveConversation() {
    // 保存当前对话到localStorage
    ensureTreeInitialized();
    // 仅保存非空对话
    const hasAnyMessage = currentPathNodeIds.length > 1;
    if (hasAnyMessage) {
        // 先保存重载更快的“tree主体”（按会话ID单独存）
        const conversationId = currentConversationId || conversationTree.id || Date.now().toString();
        currentConversationId = conversationId;
        conversationTree.id = conversationId;
        persistConversationTree(conversationId, conversationTree, currentPathNodeIds);

        const conversationData = {
            id: conversationId,
            timestamp: new Date().toISOString(),
            title: (() => {
                const firstUserId = currentPathNodeIds.find(id => {
                    const n = conversationTree.nodesById[id];
                    return n && n.role === 'user';
                });
                const first = firstUserId ? conversationTree.nodesById[firstUserId].content : '未命名对话';
                return (first || '未命名对话').substring(0, 50) + ((first || '').length > 50 ? '...' : '');
            })()
        };
        
        // 获取现有历史记录
        let history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || '[]');
        
        // 如果当前对话有ID，更新现有记录；否则添加新记录
        if (conversationId) {
            const existingIndex = history.findIndex(h => h.id === conversationId);
            if (existingIndex >= 0) {
                history[existingIndex] = conversationData;
            } else {
                history.unshift(conversationData);
            }
        } else {
            // 新对话，生成新ID
            currentConversationId = conversationData.id;
            history.unshift(conversationData);
        }
        
        // 限制历史记录数量（最多保存50条）
        if (history.length > 50) {
            history = history.slice(0, 50);
        }
        
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));

        // 仅当用户选择「提交到公网」时同步到服务端共享历史
        if (localStorage.getItem(SHARE_HISTORY_CONSENT_KEY) === 'true') {
            const base = (getApiBase() || '').replace(/\/$/, '');
            const url = base ? `${base}/api/shared_history` : '/api/shared_history';
            if (conversationTree && currentPathNodeIds) {
                fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        id: conversationId,
                        title: conversationData.title,
                        timestamp: conversationData.timestamp,
                        tree: conversationTree,
                        current_path: currentPathNodeIds
                    })
                }).catch(() => {});
            }
        }
    }
}

async function loadHistoryList() {
    if (!historyContent) return;
    historyContent.innerHTML = '<div style="color: #8e8ea0; text-align: center; padding: 2rem;">加载中…</div>';

    const localHistory = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || '[]');
    try {
        const base = (getApiBase() || '').replace(/\/$/, '');
        const url = base ? `${base}/api/shared_history` : '/api/shared_history';
        const res = await fetch(url);
        const data = await res.json();
        sharedHistoryCache = data.items || [];
    } catch (e) {
        sharedHistoryCache = [];
    }

    const sharedIds = new Set(sharedHistoryCache.map(x => x.id));
    const merged = sharedHistoryCache
        .filter(x => !hiddenSharedIds.has(x.id))
        .map(x => ({ ...x, source: 'shared' }));
    localHistory.forEach(h => {
        if (!sharedIds.has(h.id)) merged.push({ ...h, source: 'local' });
    });
    merged.sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));

    historyContent.innerHTML = '';
    if (merged.length === 0) {
        historyContent.innerHTML = '<div style="color: #8e8ea0; text-align: center; padding: 2rem;">暂无历史对话</div>';
        return;
    }

    merged.forEach((item) => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        if (item.id === currentConversationId) historyItem.classList.add('active');
        if (item.source === 'shared') historyItem.setAttribute('data-source', 'shared');

        const contentDiv = document.createElement('div');
        contentDiv.className = 'history-item-content';
        contentDiv.onclick = (e) => {
            if (!e.target.closest('.history-item-delete')) loadConversation(item.id, item);
        };

        const title = document.createElement('div');
        title.className = 'history-item-title';
        title.textContent = item.title || '未命名对话';

        const preview = document.createElement('div');
        preview.className = 'history-item-preview';
        preview.textContent = item.source === 'shared' ? '来自全部用户 · 点击查看' : '点击继续该对话';

        const time = document.createElement('div');
        time.className = 'history-item-time';
        time.textContent = item.timestamp ? new Date(item.timestamp).toLocaleString('zh-CN') : '';

        contentDiv.appendChild(title);
        contentDiv.appendChild(preview);
        contentDiv.appendChild(time);

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'history-item-actions';
        const deleteButton = document.createElement('button');
        deleteButton.className = 'history-item-delete';
        deleteButton.textContent = '×';
        deleteButton.onclick = (e) => {
            e.stopPropagation();
            deleteConversation(item.id);
        };
        if (item.source === 'shared') deleteButton.title = '仅删除本机侧边栏显示，不影响其他用户';
        actionsDiv.appendChild(deleteButton);

        historyItem.appendChild(contentDiv);
        historyItem.appendChild(actionsDiv);
        historyContent.appendChild(historyItem);
    });
}

function loadConversation(conversationId, itemFromList) {
    if (!chatContainer) return;
    // 来自“全部用户”的共享项：用服务端下发的 tree/current_path 直接加载
    if (itemFromList && itemFromList.source === 'shared' && itemFromList.tree && itemFromList.tree.nodesById && itemFromList.tree.rootId) {
        if (conversationTree && currentConversationId) saveConversation();
        chatContainer.innerHTML = '';
        if (emptyState) emptyState.style.display = 'none';
        currentConversationId = conversationId;
        conversationTree = itemFromList.tree;
        if (Array.isArray(itemFromList.current_path) && itemFromList.current_path.length > 0) {
            currentPathNodeIds = itemFromList.current_path.slice();
            conversationTree.leafId = currentPathNodeIds[currentPathNodeIds.length - 1];
        } else {
            rebuildCurrentPathFromRoot();
        }
        rerenderFromCurrentPath();
        toggleHistory();
        setTimeout(() => { if (historySidebar && historySidebar.classList.contains('open')) loadHistoryList(); }, 300);
        return;
    }
    const history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || '[]');
    const conversation = history.find(h => h.id === conversationId);
    if (!conversation) return;
    if (conversationTree && currentConversationId) saveConversation();
    chatContainer.innerHTML = '';
    if (emptyState) emptyState.style.display = 'none';
    currentConversationId = conversationId;
    conversationTree = null;
    currentPathNodeIds = [];
    const persisted = loadPersistedConversationTree(conversationId);
    if (persisted && persisted.tree && persisted.tree.nodesById && persisted.tree.rootId) {
        conversationTree = persisted.tree;
        if (Array.isArray(persisted.current_path) && persisted.current_path.length > 0) {
            currentPathNodeIds = persisted.current_path;
            conversationTree.leafId = persisted.current_path[persisted.current_path.length - 1];
        } else {
            rebuildCurrentPathFromRoot();
        }
        rerenderFromCurrentPath();
    } else if (conversation.tree && conversation.tree.nodesById && conversation.tree.rootId) {
        conversationTree = conversation.tree;
        if (Array.isArray(conversation.current_path) && conversation.current_path.length > 0) {
            currentPathNodeIds = conversation.current_path;
            conversationTree.leafId = conversation.current_path[conversation.current_path.length - 1];
        } else {
            rebuildCurrentPathFromRoot();
        }
        rerenderFromCurrentPath();
        persistConversationTree(conversationId, conversationTree, currentPathNodeIds);
    } else {
        convertLegacyMessagesToTreeIfNeeded(conversation.messages || []);
        rerenderFromCurrentPath();
        persistConversationTree(conversationId, conversationTree, currentPathNodeIds);
    }
    toggleHistory();
    setTimeout(() => { if (historySidebar && historySidebar.classList.contains('open')) loadHistoryList(); }, 300);
}

function deleteConversation(conversationId) {
    const isShared = sharedHistoryCache.some(x => x.id === conversationId);
    if (isShared) {
        if (confirm('仅从本机侧边栏隐藏该条（其他用户仍可见），确定？')) {
            hiddenSharedIds.add(conversationId);
            if (conversationId === currentConversationId) startNewConversation();
            loadHistoryList();
        }
        return;
    }
    if (confirm('确定要删除这条对话记录吗？')) {
        const history = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || '[]');
        const filteredHistory = history.filter(h => h.id !== conversationId);
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(filteredHistory));
        deletePersistedConversationTree(conversationId);
        if (conversationId === currentConversationId) startNewConversation();
        loadHistoryList();
    }
}

function startNewConversation() {
    // 保存当前对话（如果有）
    if (conversationTree && currentConversationId && currentPathNodeIds.length > 1) {
        saveConversation();
    }
    
    // 清空当前对话
    conversationTree = null;
    currentPathNodeIds = [];
    currentConversationId = null;
    observationSessionId = null;
    observationTurnCount = 0;
    localStorage.removeItem(OBSERVATION_SESSION_KEY);
    localStorage.removeItem(OBSERVATION_TURN_COUNT_KEY);
    
    // 清空显示
    if (chatContainer) {
        chatContainer.innerHTML = '';
    }
    if (emptyState) {
        emptyState.style.display = 'block';
    }
    
    // 关闭侧边栏并刷新列表
    if (historySidebar && historySidebar.classList.contains('open')) {
        toggleHistory();
    }
    setTimeout(() => {
        if (historySidebar && historySidebar.classList.contains('open')) {
            loadHistoryList();
        }
    }, 300);
}

// 移除clearHistory函数，因为现在使用单独删除功能

function handleKeyDown(event) {
    try {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    } catch (error) {
        // 已删除日志
    }
}

function showMetaAnalysisPage(userQuestion, answer, sources) {
    // 显示元文本分析新页面
    const overlay = document.getElementById('metaAnalysisOverlay');
    const page = document.getElementById('metaAnalysisPage');
    const content = document.getElementById('metaAnalysisContent');
    const closeBtn = document.getElementById('metaAnalysisClose');
    
    if (!overlay || !page || !content) {
        // 已删除日志
        return;
    }
    
    // 查找提示词元素（在page内部）
    const promptDiv = page.querySelector('.meta-analysis-prompt');
    
    // 显示页面（先显示提示词）
    overlay.classList.add('show');
    
    // 绑定关闭按钮
    if (closeBtn) {
        closeBtn.onclick = () => {
            hideMetaAnalysisPage();
        };
    }
    
    // 点击遮罩层关闭
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            hideMetaAnalysisPage();
        }
    };
    
    // 如果只有问题没有答案，只显示提示词
    if (!answer) {
        // 确保提示词显示
        if (promptDiv) {
            promptDiv.style.display = 'block';
        }
        // 清空内容区域
        content.innerHTML = '';
        return;
    }
    
    // 一旦回答加载完成，隐藏提示词并显示内容（不显示参考来源）
    // 构建内容（只包含用户问题和AI回答，不包含提示词和参考来源）
    let contentHTML = '';
    
    // 用户问题
    if (userQuestion) {
        contentHTML += `
            <div class="meta-analysis-question">
                <h3>您的问题</h3>
                <p>${escapeHtml(userQuestion)}</p>
            </div>
        `;
    }
    
    // AI回答
    contentHTML += `
        <div class="meta-analysis-answer">
            ${renderMarkdown(answer)}
        </div>
    `;
    
    content.innerHTML = contentHTML;
    
    // 隐藏提示词（回答加载完成后）
    if (promptDiv) {
        promptDiv.style.display = 'none';
    }
    
    // 高亮代码块
    if (typeof hljs !== 'undefined') {
        setTimeout(() => {
            content.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }, 0);
    }
}

function hideMetaAnalysisPage() {
    const overlay = document.getElementById('metaAnalysisOverlay');
    if (overlay) {
        overlay.classList.remove('show');
    }
    
    // 关闭弹出层后，在原页面显示回答
    // 查找所有被隐藏的元文本分析消息
    const hiddenMessages = document.querySelectorAll('[data-meta-analysis-message="true"]');
    hiddenMessages.forEach(msg => {
        msg.style.display = '';  // 显示消息
        msg.removeAttribute('data-meta-analysis-message');  // 移除标记
    });
}

function addMessage(role, content, sources = null, usedRag = false, messageIndex = null, isMetaAnalysis = false, nodeId = null) {
    if (!chatContainer) {
        // 已删除日志
        return null;
    }
    
    if (emptyState) {
        emptyState.style.display = 'none';
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    if (nodeId) {
        messageDiv.setAttribute('data-node-id', nodeId);
    }
    // 新模式：由调用方设置 data-node-id；messageIndex 仅用于旧兼容（当前已不再使用）
    if (messageIndex !== null) {
        messageDiv.setAttribute('data-message-index', messageIndex);
    }
    
    // 如果是元文本分析，添加特殊样式
    if (isMetaAnalysis) {
        messageDiv.classList.add('meta-analysis-message');
    }
    
    // 创建消息内部容器（包含头像和内容）
    const messageInner = document.createElement('div');
    messageInner.className = 'message-inner';
    
    const avatar = document.createElement('div');
    avatar.className = `avatar ${role}`;
    avatar.textContent = role === 'user' ? '你' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    
    // 对AI助手消息使用Markdown渲染，用户消息保持纯文本
    if (role === 'assistant') {
        textDiv.innerHTML = renderMarkdown(content);
        // 高亮代码块（如果使用了highlight.js）
        if (typeof hljs !== 'undefined') {
            setTimeout(() => {
                textDiv.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }, 0);
        }
    } else {
        // 用户消息不需要Markdown渲染，使用textContent防止XSS
        textDiv.textContent = content;
    }
    
    contentDiv.appendChild(textDiv);
    
    // 不再显示参考来源的详细内容，避免用户对不相关来源产生误解
    
    messageInner.appendChild(avatar);
    messageInner.appendChild(contentDiv);
    messageDiv.appendChild(messageInner);
    
    // 添加操作按钮
    // 用户消息：版本切换 + 编辑
    // Assistant消息：复制
    if (role === 'user') {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';

        // 版本切换UI（只有当该 user 属于一个 group 且 variants>1 时才显示）
        const nodeId = messageDiv.getAttribute('data-node-id');
        if (nodeId && conversationTree) {
            const group = getUserGroupForNode(nodeId);
            if (group && group.nodeIds.length > 1) {
                const nav = document.createElement('div');
                nav.className = 'variant-nav';
                nav.setAttribute('data-user-group-id', group.id);

                const prevBtn = document.createElement('button');
                prevBtn.textContent = '‹';
                const nextBtn = document.createElement('button');
                nextBtn.textContent = '›';

                const idx = getUserGroupVariantIndex(group, group.activeNodeId);
                const label = document.createElement('span');
                label.textContent = `${idx + 1}/${group.nodeIds.length}`;

                prevBtn.disabled = idx <= 0;
                nextBtn.disabled = idx >= group.nodeIds.length - 1;

                prevBtn.onclick = () => switchUserVariant(group.id, -1);
                nextBtn.onclick = () => switchUserVariant(group.id, +1);

                nav.appendChild(prevBtn);
                nav.appendChild(label);
                nav.appendChild(nextBtn);
                actionsDiv.appendChild(nav);
            }
        }
        
        const editButton = document.createElement('button');
        editButton.className = 'message-action-button edit';
        editButton.textContent = '✏️ 编辑';
        editButton.onclick = () => {
            const nid = messageDiv.getAttribute('data-node-id');
            if (nid) {
                editMessage(messageDiv, nid);
            } else {
                // 兜底（旧逻辑）
                editMessage(messageDiv, null);
            }
        };
        
        actionsDiv.appendChild(editButton);
        // 将操作按钮添加到消息容器外部，显示在消息间隔之间
        messageDiv.appendChild(actionsDiv);
    } else if (role === 'assistant') {
        // 为assistant消息添加复制按钮
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        
        const copyButton = document.createElement('button');
        copyButton.className = 'message-action-button copy';
        copyButton.textContent = '📋 复制';
        copyButton.onclick = async () => {
            try {
                // 获取最新的内容：优先从conversationTree获取，其次从DOM获取，最后使用传入的content参数
                let latestContent = content;
                
                // 尝试从conversationTree获取最新内容
                const nodeId = messageDiv.getAttribute('data-node-id');
                if (nodeId && conversationTree && conversationTree.nodesById[nodeId]) {
                    latestContent = conversationTree.nodesById[nodeId].content || latestContent;
                }
                
                // 如果仍然为空，尝试从DOM的textContent获取（去除HTML标签）
                if (!latestContent || latestContent.trim() === '') {
                    const textElement = messageDiv.querySelector('.message-text');
                    if (textElement) {
                        // 从DOM中提取纯文本（去除HTML标签）
                        latestContent = textElement.textContent || textElement.innerText || '';
                    }
                }
                
                // 将Markdown转换为纯文本
                const plainText = markdownToPlainText(latestContent);
                
                if (!plainText || plainText.trim() === '') {
                    throw new Error('没有可复制的内容');
                }
                
                // 使用Clipboard API复制
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    await navigator.clipboard.writeText(plainText);
                    // 临时改变按钮文本提示复制成功
                    const originalText = copyButton.textContent;
                    copyButton.textContent = '✅ 已复制';
                    copyButton.style.color = '#19c37d';
                    setTimeout(() => {
                        copyButton.textContent = originalText;
                        copyButton.style.color = '';
                    }, 2000);
                } else {
                    // 降级方案：使用传统方法
                    const textArea = document.createElement('textarea');
                    textArea.value = plainText;
                    textArea.style.position = 'fixed';
                    textArea.style.left = '-999999px';
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    
                    const originalText = copyButton.textContent;
                    copyButton.textContent = '✅ 已复制';
                    copyButton.style.color = '#19c37d';
                    setTimeout(() => {
                        copyButton.textContent = originalText;
                        copyButton.style.color = '';
                    }, 2000);
                }
            } catch (error) {
                // 已删除日志
                const originalText = copyButton.textContent;
                copyButton.textContent = '❌ 复制失败';
                copyButton.style.color = '#ef4444';
                setTimeout(() => {
                    copyButton.textContent = originalText;
                    copyButton.style.color = '';
                }, 2000);
            }
        };
        
        actionsDiv.appendChild(copyButton);
        messageDiv.appendChild(actionsDiv);
    }
    
    chatContainer.appendChild(messageDiv);
    
    // 滚动到底部
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return messageDiv;
}

function editMessage(messageDiv, nodeId) {
    const textDiv = messageDiv.querySelector('.message-text');
    if (!textDiv) return;
    
    const originalContent = textDiv.textContent;
    
    // 创建编辑输入框
    const editInput = document.createElement('textarea');
    editInput.className = 'message-edit-input';
    editInput.value = originalContent;
    
    // 创建操作按钮
    const editActions = document.createElement('div');
    editActions.className = 'message-edit-actions';
    
    const saveButton = document.createElement('button');
    saveButton.className = 'message-edit-button save';
    saveButton.textContent = '保存并发送';
    saveButton.onclick = async () => {
        const newContent = editInput.value.trim();
        if (newContent) {
            ensureTreeInitialized();
            if (!nodeId || !conversationTree.nodesById[nodeId]) {
                // 兜底：没有 nodeId 就只更新文本，不触发分支
                textDiv.textContent = newContent;
                textDiv.style.display = 'block';
                editInput.remove();
                editActions.remove();
                return;
            }

            // 关键：编辑并不是“覆盖原节点”，而是“同一用户组新增一个 variant 节点（兄弟分支）”
            const oldNode = conversationTree.nodesById[nodeId];
            const group = getUserGroupForNode(nodeId);
            const parentId = oldNode.parentId;
            const groupId = group ? group.id : null;
            const { nodeId: newUserNodeId, groupId: ensuredGroupId } = addUserNodeWithGroup(newContent, parentId, groupId);

            // 让父节点沿新 variant 继续（新分支成为当前显示路径）
            const parent = conversationTree.nodesById[parentId];
            if (parent) parent.activeChildId = newUserNodeId;

            // 恢复原始显示（本节点不再强行改文本，重绘会切到新 variant）
            textDiv.style.display = 'block';
            editInput.remove();
            editActions.remove();

            // 重新回放到新分支并渲染
            rebuildCurrentPathFromRoot();
            rerenderFromCurrentPath();
            saveConversation();

            // 从“新用户节点”继续生成回答（不新增用户气泡）
            awaitSendFromEditedUserNode(newUserNodeId);
        } else {
            // 恢复原始显示
            textDiv.style.display = 'block';
            editInput.remove();
            editActions.remove();
        }
    };
    
    const cancelButton = document.createElement('button');
    cancelButton.className = 'message-edit-button cancel';
    cancelButton.textContent = '取消';
    cancelButton.onclick = () => {
        textDiv.style.display = 'block';
        editInput.remove();
        editActions.remove();
    };
    
    editActions.appendChild(saveButton);
    editActions.appendChild(cancelButton);
    
    // 替换显示
    textDiv.style.display = 'none';
    const contentDiv = messageDiv.querySelector('.message-content');
    contentDiv.insertBefore(editInput, textDiv);
    contentDiv.appendChild(editActions);
    
    // 聚焦输入框并自动调整高度
    editInput.focus();
    editInput.style.height = 'auto';
    editInput.style.height = Math.min(editInput.scrollHeight, 200) + 'px';
    editInput.setSelectionRange(editInput.value.length, editInput.value.length);
    
    // 支持Enter发送，Shift+Enter换行
    editInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            saveButton.click();
        }
    });
}

function showTypingIndicator() {
    if (!chatContainer) return;
    
    // 若已有多跳提示，先移除，避免重复
    removeMultiHopIndicator();
    // 若已有等待动画，不重复创建
    if (document.getElementById('typingIndicator')) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'typingIndicator';
    
    const messageInner = document.createElement('div');
    messageInner.className = 'message-inner';
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar assistant';
    avatar.textContent = 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    
    contentDiv.appendChild(typingDiv);
    messageInner.appendChild(avatar);
    messageInner.appendChild(contentDiv);
    messageDiv.appendChild(messageInner);
    chatContainer.appendChild(messageDiv);
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

function showMultiHopIndicator(message) {
    if (!chatContainer) {
        // 已删除日志
        return;
    }

    // 先移除加载动画（三个点），避免同时显示
    removeTypingIndicator();

    // 准备消息内容：文字 + 内联加载动画（替换省略号）
    const textContent = message || '您的问题需要深度思考，请等待系统推理…';
    // 移除末尾的省略号，替换为内联加载动画
    const textWithoutEllipsis = textContent.replace(/…$/, '').trim();
    const messageWithDots = `${textWithoutEllipsis} <span class="inline-typing-dots"><span>.</span><span>.</span><span>.</span></span>`;

    // 如果已经有多跳提示消息，直接更新其内容
    let existing = document.getElementById('multiHopIndicator');
    if (existing) {
        const textDiv = existing.querySelector('.message-text');
        if (textDiv) {
            // 更新文字内容，并在末尾添加内联加载动画
            textDiv.innerHTML = messageWithDots;
        }
        existing.style.display = 'block';
        existing.style.visibility = 'visible';
        existing.style.opacity = '1';
        
        // 强制浏览器重绘
        existing.offsetHeight; // 触发重排
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        // 已删除日志
        return;
    }

    // 否则，单独插入一条“助手消息气泡”作为多跳提示（不再去复用三个点的占位符）
    const multiHopMessage = addMessage(
        'assistant',
        messageWithDots,
        null,
        false,
        null,
        false,
        null
    );

    if (multiHopMessage) {
        // 标记这个消息用于后续删除/隐藏
        multiHopMessage.id = 'multiHopIndicator';
        multiHopMessage.style.display = 'block';
        multiHopMessage.style.visibility = 'visible';
        multiHopMessage.style.opacity = '1';
        
        // 确保 message-text 使用 innerHTML 来显示带加载动画的内容
        const textDiv = multiHopMessage.querySelector('.message-text');
        if (textDiv) {
            // addMessage 可能已经设置了内容，但我们需要确保是 HTML 格式
            textDiv.innerHTML = messageWithDots;
        }
        
        // 强制浏览器重绘，确保DOM更新可见
        multiHopMessage.offsetHeight; // 触发重排
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        // 使用 requestAnimationFrame 确保在下一帧渲染
        requestAnimationFrame(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
            // 已删除日志
        });
        
        // 已删除日志
    } else {
        // 已删除日志
    }
}

function removeMultiHopIndicator() {
    const indicator = document.getElementById('multiHopIndicator');
    if (indicator) {
        indicator.remove();
    }
}

function rerenderFromCurrentPath() {
    if (!chatContainer) return;
    chatContainer.innerHTML = '';
    if (emptyState) {
        emptyState.style.display = currentPathNodeIds.length > 1 ? 'none' : 'block';
    }
    if (!conversationTree) return;

    const ids = currentPathNodeIds.slice(1); // skip root
    ids.forEach((id) => {
        const n = conversationTree.nodesById[id];
        addMessage(n.role, n.content, null, false, null, false, id);
    });
}

async function awaitSendFromEditedUserNode(userNodeId) {
    // 生成该用户节点之后的assistant回复（不新增用户消息气泡）
    if (!conversationTree || !conversationTree.nodesById[userNodeId]) return;
    // 把 leaf 设置到这个 user 节点（清空其后续 activeChild 链会在新回答后覆盖）
    conversationTree.leafId = userNodeId;
    rebuildCurrentPathFromRoot();
    rerenderFromCurrentPath();
    await requestAssistantAnswer(conversationTree.nodesById[userNodeId].content, /*isEditingResend*/true);
}

// 安全的JSON解析函数，确保正确处理SSE格式
function safeParseSSEJson(line) {
    if (!line || typeof line !== 'string') {
        return null;
    }
    
    let jsonStr = line.trim();
    
    // 如果以 "data: " 开头，去掉前缀
    if (jsonStr.startsWith('data: ')) {
        jsonStr = jsonStr.slice(6).trim();
    }
    
    // 循环处理，确保完全去除所有 "data: " 前缀
    while (jsonStr.startsWith('data: ')) {
        jsonStr = jsonStr.slice(6).trim();
    }
    
    // 验证是否是有效的JSON格式
    if (!jsonStr.startsWith('{') && !jsonStr.startsWith('[')) {
        return null;
    }
    
    // 最后检查：如果仍然包含 "data: "，说明有问题
    if (jsonStr.includes('data: ')) {
        // 已删除日志
        return null;
    }
    
    try {
        return JSON.parse(jsonStr);
    } catch (e) {
        // 已删除日志
        return null;
    }
}

async function requestAssistantAnswer(userMessageText, isEditingResend) {
    // 与 sendMessage 的后端调用逻辑复用
    try {
        isProcessing = true;
        if (sendButton) {
            sendButton.disabled = true;
            sendButton.innerHTML = '<div class="loading"></div>';
        }

        showTypingIndicator();

        // route-check（保持现有体验）
        let isMetaAnalysisDetected = false;
        try {
            const routeResponse = await fetch(getApiBase() + '/api/route-check', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMessageText })
            });
            if (routeResponse.ok) {
                const routeData = await routeResponse.json();
                isMetaAnalysisDetected = routeData.is_meta_analysis || false;
                if (isMetaAnalysisDetected) {
                    showMetaAnalysisPage(userMessageText, null, null);
                }
            }
        } catch (e) {
            // 已删除日志
        }

        const conversation_history = buildConversationHistoryForBackend(true);
        await ensureObservationSession();
        
        // 创建assistant节点和消息占位符（用于流式显示）
        const parentId = conversationTree.leafId;
        let assistantNodeId = null;
        let messagePlaceholder = null;
        let fullAnswer = '';
        let finalData = null;
        let firstChunkReceived = false;
        
        // 使用fetch处理流式响应
        const response = await fetch(getApiBase() + '/api/chat', {
            method: 'POST',
            headers: getAuthHeaders(),
            credentials: 'include',
            body: JSON.stringify({
                message: userMessageText,
                conversation_history,
                session_id: observationSessionId,
                conversation_id: currentConversationId
            })
        });

        if (!response.ok) throw new Error('请求失败');
        
        // 处理流式响应
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let hasReceivedFinal = false;
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                // 保留最后一个不完整的行
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    const trimmedLine = line.trim();
                    if (trimmedLine === '') continue;
                    
                    // 处理SSE格式：data: {...}
                    if (trimmedLine.startsWith('data: ')) {
                        try {
                            // 使用安全的JSON解析函数
                            // 已删除日志
                            const data = safeParseSSEJson(trimmedLine);
                            if (!data) {
                                // 已删除日志
                                continue;
                            }
                            
                            // 已删除日志
                            
                            if (data.type === 'multi_hop_notification') {
                                // 多跳推理通知
                                // 已删除日志
                                // showMultiHopIndicator 内部会先移除 typingIndicator，然后创建或更新 multiHopIndicator
                                // 直接复用当前“三个点”的 message-content 做替换，避免 UI 一直停留在纯 dots
                                try {
                                    // 已删除日志
                                    showMultiHopIndicator(data.message || '您的问题相对复杂，正在执行深度思考推理…');
                                    // 已删除日志
                                    
                                    // 强制浏览器重绘，确保DOM更新可见
                                    if (chatContainer) {
                                        chatContainer.offsetHeight; // 触发重排
                                        requestAnimationFrame(() => {
                                            chatContainer.scrollTop = chatContainer.scrollHeight;
                                        });
                                    }
                                } catch (e) {
                                    // 已删除日志
                                }
                                
                                // 已删除日志
                            } else if (data.type === 'chunk') {
                                // 流式文本chunk
                                // 注意：不要在这里立即移除多跳提示，让用户能看到提示
                                // 多跳提示会在开始显示实际回答内容时移除
                                
                                if (!firstChunkReceived) {
                                    // 第一次收到chunk，移除等待动画并创建assistant节点和消息
                                    removeTypingIndicator();
                                    
                                    assistantNodeId = addNode('assistant', '', parentId);
                                    messagePlaceholder = addMessage('assistant', '', null, false, null, false, assistantNodeId);
                                    firstChunkReceived = true;
                                    
                                    // 如果元文本分析弹窗已显示，隐藏消息占位符
                                    if (isMetaAnalysisDetected) {
                                        if (messagePlaceholder) {
                                            messagePlaceholder.style.display = 'none';
                                            messagePlaceholder.setAttribute('data-meta-analysis-message', 'true');
                                        }
                                    }
                                    
                                    // 确保流式更新时删除sources元素
                                    if (messagePlaceholder) {
                                        const sourcesDiv = messagePlaceholder.querySelector('.message-sources');
                                        if (sourcesDiv) {
                                            sourcesDiv.remove();
                                        }
                                    }
                                }
                                
                                if (data.content) {
                                    fullAnswer += data.content;
                                    
                                    // 当开始收到实际回答内容时，移除多跳提示
                                    // 这样用户能看到多跳提示，直到回答开始显示
                                    // 只要fullAnswer有内容（即使很小），就移除多跳提示，开始显示回答
                                    if (fullAnswer && fullAnswer.trim().length > 0) {
                                        // 已删除日志
                                        removeMultiHopIndicator();
                                    }
                                    
                                    // 如果元文本分析弹窗已显示，实时更新弹窗内容
                                    if (isMetaAnalysisDetected) {
                                        const overlay = document.getElementById('metaAnalysisOverlay');
                                        if (overlay && overlay.classList.contains('show')) {
                                            // 实时更新弹窗内容
                                            showMetaAnalysisPage(userMessageText, fullAnswer, null);
                                        }
                                    }
                                    
                                    // 实时更新消息内容（使用Markdown渲染）
                                    if (messagePlaceholder) {
                                        // 如果元文本分析弹窗已显示，隐藏消息占位符
                                        if (isMetaAnalysisDetected) {
                                            messagePlaceholder.style.display = 'none';
                                        } else {
                                            const textElement = messagePlaceholder.querySelector('.message-text');
                                            if (textElement) {
                                                
                                                // 确保样式正确应用，防止分栏
                                                textElement.style.display = 'block';
                                                textElement.style.width = '100%';
                                                textElement.style.boxSizing = 'border-box';
                                                
                                                textElement.innerHTML = renderMarkdown(fullAnswer);
                                                // 高亮代码块（如果使用了highlight.js）
                                                if (typeof hljs !== 'undefined') {
                                                    setTimeout(() => {
                                                        textElement.querySelectorAll('pre code').forEach((block) => {
                                                            hljs.highlightElement(block);
                                                        });
                                                    }, 0);
                                                }
                                                
                                                // 确保sources在流式更新时删除
                                                const sourcesDiv = messagePlaceholder.querySelector('.message-sources');
                                                if (sourcesDiv) {
                                                    sourcesDiv.remove();
                                                }
                                                
                                                // 滚动到底部
                                                if (chatContainer) {
                                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                                }
                                            }
                                        }
                                    }
                                }
                            } else if (data.type === 'observation') {
                                onObservationScheduledFromServer(data.turn);
                            } else if (data.type === 'final') {
                                hasReceivedFinal = true;
                                // 最终结果
                                hasReceivedFinal = true;
                                finalData = data.data;
                                
                                // 移除所有等待动画和提示
                                removeTypingIndicator();
                                removeMultiHopIndicator();
                                
                                // 如果还没有收到chunk，创建消息占位符
                                if (!firstChunkReceived) {
                                    assistantNodeId = addNode('assistant', '', parentId);
                                    messagePlaceholder = addMessage('assistant', '', null, false, null, false, assistantNodeId);
                                    firstChunkReceived = true;
                                }
                                
                                // 更新完整答案
                                fullAnswer = finalData.answer || fullAnswer;
                                
                                // 如果还没有创建节点，现在创建
                                if (!assistantNodeId) {
                                    assistantNodeId = addNode('assistant', fullAnswer, parentId);
                                } else {
                                    // 更新节点内容
                                    if (conversationTree && conversationTree.nodesById[assistantNodeId]) {
                                        conversationTree.nodesById[assistantNodeId].content = fullAnswer;
                                    }
                                }
                                
                                const isMetaAnalysis = finalData.is_meta_analysis || isMetaAnalysisDetected;
                                
                                // 更新消息显示（如果已经有占位符，更新它；否则创建新消息）
                                if (messagePlaceholder) {
                                    // 更新现有消息的内容（使用Markdown渲染）
                                    const textElement = messagePlaceholder.querySelector('.message-text');
                                    if (textElement) {
                                        // 确保样式正确应用，防止分栏
                                        textElement.style.display = 'block';
                                        textElement.style.width = '100%';
                                        textElement.style.boxSizing = 'border-box';
                                        
                                        textElement.innerHTML = renderMarkdown(fullAnswer);
                                        // 高亮代码块（如果使用了highlight.js）
                                        if (typeof hljs !== 'undefined') {
                                            setTimeout(() => {
                                                textElement.querySelectorAll('pre code').forEach((block) => {
                                                    hljs.highlightElement(block);
                                                });
                                            }, 0);
                                        }
                                    }
                                    
                                    // 流式输出时不显示sources，直接删除sources元素
                                    const sourcesDiv = messagePlaceholder.querySelector('.message-sources');
                                    if (sourcesDiv) {
                                        sourcesDiv.remove();
                                    }
                                    
                                    // 如果是元文本分析，隐藏消息占位符并标记，然后更新弹窗
                                    if (isMetaAnalysis) {
                                        messagePlaceholder.style.display = 'none';
                                        messagePlaceholder.setAttribute('data-meta-analysis-message', 'true');
                                        // 更新弹窗内容（替换提示词）
                                        showMetaAnalysisPage(userMessageText, fullAnswer, finalData.sources);
                                    }
                                } else {
                                    // 创建新消息（元文本分析保持现有隐藏逻辑）
                                    if (isMetaAnalysis) {
                                        const assistantMessage = addMessage('assistant', fullAnswer, finalData.sources, finalData.used_rag, null, true, assistantNodeId);
                                        if (assistantMessage) {
                                            assistantMessage.style.display = 'none';
                                            assistantMessage.setAttribute('data-meta-analysis-message', 'true');
                                        }
                                        // 更新弹窗内容（替换提示词）
                                        showMetaAnalysisPage(userMessageText, fullAnswer, finalData.sources);
                                    } else {
                                        addMessage('assistant', fullAnswer, finalData.sources, finalData.used_rag, null, false, assistantNodeId);
                                    }
                                }
                                
                                saveConversation();
                            } else if (data.type === 'error') {
                                throw new Error(data.error || '服务器返回错误');
                            }
                        } catch (e) {
                            // 已删除日志
                            
                            // 如果是JSON解析错误，记录但继续处理下一行
                            if (e instanceof SyntaxError || e.message.includes('JSON')) {
                                // 已删除日志
                                continue;
                            }
                            // 其他错误记录但不抛出，继续处理
                            // 已删除日志
                            continue;
                        }
                    } else if (trimmedLine.length > 0) {
                        // 如果不是以 "data: " 开头，可能是格式错误，记录但继续
                        // 已删除日志
                    }
                }
            }
            
            // 处理剩余的buffer
            const trimmedBuffer = buffer.trim();
            if (trimmedBuffer && trimmedBuffer.startsWith('data: ')) {
                try {
                    // 使用安全的JSON解析函数
                    const data = safeParseSSEJson(trimmedBuffer);
                    if (data && data.type === 'final' && !hasReceivedFinal) {
                        hasReceivedFinal = true;
                        finalData = data.data;
                        
                        // 移除所有等待动画和提示
                        removeTypingIndicator();
                        removeMultiHopIndicator();
                        
                        // 如果还没有收到chunk，创建消息占位符
                        if (!firstChunkReceived) {
                            assistantNodeId = addNode('assistant', '', parentId);
                            messagePlaceholder = addMessage('assistant', '', null, false, null, false, assistantNodeId);
                            firstChunkReceived = true;
                        }
                        
                        fullAnswer = finalData.answer || fullAnswer;
                        // 处理最终数据（与上面的final处理逻辑相同）
                        if (!assistantNodeId) {
                            assistantNodeId = addNode('assistant', fullAnswer, parentId);
                        } else {
                            if (conversationTree && conversationTree.nodesById[assistantNodeId]) {
                                conversationTree.nodesById[assistantNodeId].content = fullAnswer;
                            }
                        }
                        const isMetaAnalysis = finalData.is_meta_analysis || isMetaAnalysisDetected;
                        if (messagePlaceholder) {
                            // 更新消息内容（使用Markdown渲染）
                            const textElement = messagePlaceholder.querySelector('.message-text');
                            if (textElement) {
                                // 确保样式正确应用，防止分栏
                                textElement.style.display = 'block';
                                textElement.style.width = '100%';
                                textElement.style.boxSizing = 'border-box';
                                
                                textElement.innerHTML = renderMarkdown(fullAnswer);
                                // 高亮代码块（如果使用了highlight.js）
                                if (typeof hljs !== 'undefined') {
                                    setTimeout(() => {
                                        textElement.querySelectorAll('pre code').forEach((block) => {
                                            hljs.highlightElement(block);
                                        });
                                    }, 0);
                                }
                            }
                            
                            // 流式输出时不显示sources，直接删除sources元素
                            const sourcesDiv = messagePlaceholder.querySelector('.message-sources');
                            if (sourcesDiv) {
                                sourcesDiv.remove();
                            }
                            
                            // 如果是元文本分析，隐藏消息占位符并标记，然后更新弹窗
                            if (isMetaAnalysis) {
                                messagePlaceholder.style.display = 'none';
                                messagePlaceholder.setAttribute('data-meta-analysis-message', 'true');
                                // 更新弹窗内容（替换提示词）
                                showMetaAnalysisPage(userMessageText, fullAnswer, finalData.sources);
                            }
                        } else {
                            if (isMetaAnalysis) {
                                const assistantMessage = addMessage('assistant', fullAnswer, finalData.sources, finalData.used_rag, null, true, assistantNodeId);
                                if (assistantMessage) {
                                    assistantMessage.style.display = 'none';
                                    assistantMessage.setAttribute('data-meta-analysis-message', 'true');
                                }
                                // 更新弹窗内容（替换提示词）
                                showMetaAnalysisPage(userMessageText, fullAnswer, finalData.sources);
                            } else {
                                addMessage('assistant', fullAnswer, finalData.sources, finalData.used_rag, null, false, assistantNodeId);
                            }
                        }
                        saveConversation();
                    }
                } catch (e) {
                    // 已删除日志
                }
            } else if (trimmedBuffer) {
                // buffer中有内容但不是以 "data: " 开头，可能是格式问题
                // 已删除日志
            }
            
            // 如果没有收到final消息，但有内容，创建消息
            if (!hasReceivedFinal && fullAnswer) {
                // 已删除日志
                // 移除所有等待动画和提示
                removeTypingIndicator();
                removeMultiHopIndicator();
                if (!assistantNodeId) {
                    assistantNodeId = addNode('assistant', fullAnswer, parentId);
                }
                if (!messagePlaceholder) {
                    messagePlaceholder = addMessage('assistant', fullAnswer, null, false, null, false, assistantNodeId);
                } else {
                    const contentElement = messagePlaceholder.querySelector('.message-content');
                    if (contentElement) {
                        contentElement.textContent = fullAnswer;
                    }
                }
                if (conversationTree && conversationTree.nodesById[assistantNodeId]) {
                    conversationTree.nodesById[assistantNodeId].content = fullAnswer;
                }
                saveConversation();
            } else if (!hasReceivedFinal && !fullAnswer) {
                // 既没有收到final也没有内容，可能是错误
                removeTypingIndicator();
                removeMultiHopIndicator();
                // 已删除日志
                throw new Error('未收到完整的响应数据，请检查网络连接或稍后重试');
            }
        } catch (streamError) {
            // 已删除日志
            throw streamError;
        }
        } catch (error) {
        removeTypingIndicator();
        removeMultiHopIndicator();
        // 已删除日志
        
        // 如果已经有部分内容，显示部分内容并提示错误
        if (fullAnswer && assistantNodeId) {
            const errorMsg = `\n\n⚠️ 注意：响应可能不完整。错误信息：${error.message || '未知错误'}`;
            if (messagePlaceholder) {
                const contentElement = messagePlaceholder.querySelector('.message-content');
                if (contentElement) {
                    contentElement.textContent = fullAnswer + errorMsg;
                }
            } else {
                addMessage('assistant', fullAnswer + errorMsg, null, false, null, false, assistantNodeId);
            }
            saveConversation();
        } else {
            // 如果没有内容，显示错误消息
            addMessage('assistant', `抱歉，处理您的问题时出现错误：${error.message || '未知错误'}。请稍后重试。`, null, false, null, false);
        }
    } finally {
        if (finalData) {
            onAssistantTurnComplete(finalData);
        }
        isProcessing = false;
        if (sendButton) {
            sendButton.disabled = false;
            sendButton.textContent = '发送';
        }
        if (messageInput) {
            messageInput.focus();
        }
    }
}

async function sendMessage() {
    try {
        if (!messageInput) {
            // 已删除日志
            return;
        }
        
        const message = messageInput.value.trim();
        if (!message || isProcessing) return;
        
        // 如果是新消息（不是编辑后的），创建新对话ID
        if (!currentConversationId && (!conversationTree || currentPathNodeIds.length <= 1)) {
            currentConversationId = Date.now().toString();
        }

        ensureTreeInitialized();
        // 新用户消息：作为当前 leaf 的子节点加入（新用户组）
        const parentId = conversationTree.leafId;
        const { nodeId: userNodeId } = addUserNodeWithGroup(message, parentId);

        // 渲染用户消息（不会新增“额外一条”，这里只是正常发送）
        addMessage('user', message, null, false, null, false, userNodeId);
        
        // 清空输入框
        if (messageInput) {
            messageInput.value = '';
            messageInput.style.height = 'auto';
        }

        // 复用统一的后端请求与 assistant 节点落库逻辑
        await requestAssistantAnswer(message, false);
    } catch (error) {
        // 已删除日志
        // 确保即使出错也能恢复按钮状态
        isProcessing = false;
        if (sendButton) {
            sendButton.disabled = false;
            sendButton.textContent = '发送';
        }
    }
}

// 系统说明弹窗相关
const AGREEMENT_STORAGE_KEY = 'novel_rag_agreement_confirmed';
const SHARE_HISTORY_CONSENT_KEY = 'novel_rag_share_history_consent';

function checkAndShowAgreement() {
    const agreementConfirmed = localStorage.getItem(AGREEMENT_STORAGE_KEY);
    const shareConsentSet = localStorage.getItem(SHARE_HISTORY_CONSENT_KEY);
    if (agreementConfirmed && shareConsentSet) return;

    const overlay = document.getElementById('systemAgreementOverlay');
    const button = document.getElementById('systemAgreementButton');
    const consentBlock = document.getElementById('shareHistoryConsentBlock');
    const consentCheckbox = document.getElementById('shareHistoryConsentCheckbox');

    if (overlay) overlay.style.display = 'flex';
    if (consentBlock) {
        consentBlock.style.display = shareConsentSet ? 'none' : 'block';
        if (!shareConsentSet && consentCheckbox) consentCheckbox.checked = true;
    }

    if (button) {
        button.addEventListener('click', function() {
            localStorage.setItem(AGREEMENT_STORAGE_KEY, 'true');
            if (consentBlock && consentBlock.style.display !== 'none' && consentCheckbox) {
                localStorage.setItem(SHARE_HISTORY_CONSENT_KEY, consentCheckbox.checked ? 'true' : 'false');
            }
            if (overlay) overlay.style.display = 'none';
        });
    }
}

// 页面加载时初始化
function initializeApp() {
    if (!initElements()) {
        // 已删除日志
        return;
    }
    
    // 检查并显示系统说明弹窗
    checkAndShowAgreement();
    
    // 初始化Markdown
    initMarkdown();
    
    // 设置输入框自动调整高度
    setupInputAutoResize();
    
    // 绑定事件监听器
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
    
    if (messageInput) {
        messageInput.addEventListener('keydown', handleKeyDown);
        messageInput.focus();
    }
    
    if (historyButton) {
        historyButton.addEventListener('click', toggleHistory);
    }
    
    if (closeButton) {
        closeButton.addEventListener('click', toggleHistory);
    }
    
    if (overlay) {
        overlay.addEventListener('click', toggleHistory);
    }
    
    if (newChatButton) {
        newChatButton.addEventListener('click', startNewConversation);
    }

    const trajectoryButton = document.getElementById('trajectoryButton');
    const trajectoryCloseButton = document.getElementById('trajectoryCloseButton');
    const regenerateTrajectoryButton = document.getElementById('regenerateTrajectoryButton');
    const openStarMapButton = document.getElementById('openStarMapButton');
    const starMapBackButton = document.getElementById('starMapBackButton');
    if (trajectoryButton) trajectoryButton.addEventListener('click', toggleTrajectorySidebar);
    if (trajectoryCloseButton) trajectoryCloseButton.addEventListener('click', toggleTrajectorySidebar);
    if (regenerateTrajectoryButton) regenerateTrajectoryButton.addEventListener('click', () => triggerObservationExtract(false));
    if (openStarMapButton) openStarMapButton.addEventListener('click', openStarMapFullPage);
    if (starMapBackButton) starMapBackButton.addEventListener('click', closeStarMapFullPage);

    loadObservationSessionFromStorage();
    refreshAuthProfile();
    if (typeof preloadStarMapCatalog === 'function') {
        preloadStarMapCatalog(getApiBase()).catch(() => {});
    }
    
    // 启动时默认新对话，不自动加载上次对话
}

// 定期保存当前对话到历史记录
setInterval(() => {
    if (conversationTree && currentPathNodeIds.length > 1) {
        saveConversation();
    }
}, 30000); // 每30秒保存一次

// 页面加载完成后初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}


