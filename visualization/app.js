/**
 * Flux — AI Crowd Management Simulator
 * =============================================
 * Interactive web dashboard for the CrowdManagementEnv.
 * Communicates with the Python backend via REST API.
 */

// ─── State ───────────────────────────────────────────────────────────────────

const state = {
    connected: false,
    running: false,
    autoPlaying: false,
    autoInterval: null,
    speed: 300,
    currentTask: 'easy',
    observation: null,
    stepResult: null,
    gradeResult: null,
    envState: null,
    timelineData: {},   // {zoneId: [density values]}
    maxTimeline: 300,   // Match max steps of Hard mode
    episodeDone: false,
};

const API_BASE = ""; // Use relative paths for iframe compatibility on HF Spaces

const ZONE_IDS = ['A', 'B', 'C', 'D', 'E', 'F'];
const ZONE_NAMES = {
    A: 'Main Entrance', B: 'North Stand', C: 'South Stand',
    D: 'Central Arena', E: 'East Concourse', F: 'West Exit',
};

// Zone topology: neighbors and gate counts for dynamic dropdowns
const ZONE_NEIGHBORS = {
    A: ['B', 'C'],
    B: ['A', 'D', 'E'],
    C: ['A', 'D', 'F'],
    D: ['B', 'C', 'E', 'F'],
    E: ['B', 'D'],
    F: ['C', 'D'],
};

const ZONE_GATE_COUNT = { A: 3, B: 2, C: 2, D: 4, E: 2, F: 3 };

const RISK_COLORS = {
    safe: '#00f5d4',
    elevated: '#ffbe0b',
    critical: '#ff4d6d',
    stampede: '#ff0040',
};

// ─── API ─────────────────────────────────────────────────────────────────────

async function apiCall(endpoint, method = 'GET', body = null) {
    try {
        const opts = {
            method,
            headers: { 'Content-Type': 'application/json' },
        };
        if (body) opts.body = JSON.stringify(body);
        const res = await fetch(`${API_BASE}${endpoint}`, opts);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        console.error(`API error: ${endpoint}`, e);
        addEvent(`❌ API error: ${e.message}`, 'danger');
        return null;
    }
}

async function resetEnv() {
    state.episodeDone = false;
    state.timelineData = {};
    ZONE_IDS.forEach(id => state.timelineData[id] = []);
    clearTimeline();

    const seed = Math.floor(Math.random() * 100000);
    const data = await apiCall('/reset', 'POST', {
        task: state.currentTask,
        seed: seed,
    });
    if (data) {
        state.observation = data.observation;
        state.envState = data.state;
        state.running = true;
        updateUI();
        addEvent(`🔄 Environment reset — Task: ${state.currentTask.toUpperCase()} (seed: ${seed})`, 'success');
        setStatus('Running', true);
    }
}

async function stepEnv(action) {
    if (state.episodeDone) {
        addEvent('⏹ Episode is done. Press Reset to start a new one.', 'warning');
        return;
    }

    const data = await apiCall('/step', 'POST', action);
    if (data) {
        state.stepResult = data;
        state.observation = data.observation;
        state.envState = data.env_state;
        state.gradeResult = data.grade;

        // Track timeline
        if (state.observation && state.observation.zones) {
            state.observation.zones.forEach(z => {
                if (!state.timelineData[z.zone_id]) state.timelineData[z.zone_id] = [];
                state.timelineData[z.zone_id].push(z.density);
                if (state.timelineData[z.zone_id].length > state.maxTimeline) {
                    state.timelineData[z.zone_id].shift();
                }
            });
        }

        updateUI();

        // Log events
        if (data.info && data.info.action_result) {
            addEvent(data.info.action_result, 'action');
        }
        if (data.info && data.info.events) {
            data.info.events.forEach(e => {
                let cls = 'default';
                if (e.includes('STAMPEDE') || e.includes('💀')) cls = 'critical';
                else if (e.includes('CRITICAL') || e.includes('🔴')) cls = 'danger';
                else if (e.includes('ELEVATED') || e.includes('⚠')) cls = 'warning';
                else if (e.includes('SURGE') || e.includes('⚡')) cls = 'warning';
                addEvent(e, cls);
            });
        }

        // Check done
        if (data.terminated || data.truncated) {
            state.episodeDone = true;
            stopAutoPlay();
            if (data.terminated) {
                addEvent(`💀 STAMPEDE — Episode terminated at step ${state.observation.time_step}!`, 'critical');
                setStatus('Stampede!', false, true);
            } else {
                addEvent(`✅ Episode completed — Survived all ${state.observation.time_step} steps!`, 'success');
                setStatus('Complete', false);
            }
            updateGrade();
        }
    }
}

async function getState() {
    const data = await apiCall('/state');
    if (data) {
        state.envState = data;
    }
    return data;
}

// ─── UI Update ───────────────────────────────────────────────────────────────

function updateUI() {
    const obs = state.observation;
    if (!obs) return;

    // Step counter
    document.getElementById('step-counter').textContent = `Step: ${obs.time_step} / ${obs.max_steps}`;

    // Global metrics
    document.getElementById('metric-population').textContent = obs.total_population.toLocaleString();
    document.getElementById('metric-risk').textContent = obs.global_risk_score.toFixed(3);

    if (state.envState) {
        document.getElementById('metric-reward').textContent =
            (state.envState.cumulative_reward || 0).toFixed(1);
        document.getElementById('metric-arrivals').textContent =
            (state.envState.total_arrivals || 0).toLocaleString();
    }

    // Risk gauge
    updateRiskGauge(obs.global_risk_score);

    // Zone map + bars
    updateZones(obs.zones);

    // Timeline
    drawTimeline();

    // Grade (live update)
    if (state.gradeResult) {
        updateGrade();
    }
}

function updateRiskGauge(riskScore) {
    const fill = document.getElementById('gauge-fill');
    const pct = Math.min(riskScore * 100, 100);
    fill.style.width = `${pct}%`;

    fill.className = 'gauge-fill';
    if (riskScore >= 1.0) fill.classList.add('stampede');
    else if (riskScore >= 0.7) fill.classList.add('critical');
    else if (riskScore >= 0.4) fill.classList.add('elevated');
}

function updateZones(zones) {
    if (!zones) return;

    const barsContainer = document.getElementById('zone-bars');
    // Build bars if not yet built
    if (barsContainer.children.length === 0) {
        zones.forEach(z => {
            const row = document.createElement('div');
            row.className = 'zone-bar-row';
            row.innerHTML = `
                <span class="zone-bar-label">${z.zone_id}</span>
                <div class="zone-bar-track">
                    <div class="zone-bar-fill" id="bar-fill-${z.zone_id}"></div>
                </div>
                <span class="zone-bar-value" id="bar-val-${z.zone_id}">0</span>
            `;
            barsContainer.appendChild(row);
        });
    }

    zones.forEach(z => {
        // Update SVG map
        const zoneGroup = document.getElementById(`zone-${z.zone_id}`);
        if (zoneGroup) {
            // Risk class
            zoneGroup.className.baseVal = `zone risk-${z.risk_level}`;
            if (z.alert_active) zoneGroup.classList.add('alert-active');

            // Text
            const popEl = document.getElementById(`pop-${z.zone_id}`);
            const densEl = document.getElementById(`density-${z.zone_id}`);
            if (popEl) popEl.textContent = `${z.current_population} / ${z.capacity}`;
            if (densEl) densEl.textContent = `${z.density.toFixed(2)} ppm²`;

            // Risk indicator
            const riskEl = document.getElementById(`risk-${z.zone_id}`);
            if (riskEl) {
                riskEl.setAttribute('fill', RISK_COLORS[z.risk_level] || RISK_COLORS.safe);
                riskEl.className.baseVal = `risk-indicator ${z.risk_level !== 'safe' ? z.risk_level : ''}`;
            }
        }

        // Update bar
        const barFill = document.getElementById(`bar-fill-${z.zone_id}`);
        const barVal = document.getElementById(`bar-val-${z.zone_id}`);
        if (barFill) {
            const pct = Math.min((z.current_population / z.capacity) * 100, 100);
            barFill.style.width = `${pct}%`;
            barFill.className = `zone-bar-fill ${z.risk_level !== 'safe' ? z.risk_level : ''}`;
        }
        if (barVal) {
            barVal.textContent = `${z.current_population}`;
        }
    });

    // Animate connection lines for flow
    updateFlowLines(zones);
}

function updateFlowLines(zones) {
    const connLines = document.querySelectorAll('.conn-line');
    connLines.forEach(line => {
        const from = line.dataset.from;
        const to = line.dataset.to;
        const fromZone = zones.find(z => z.zone_id === from);
        const toZone = zones.find(z => z.zone_id === to);

        if (fromZone && toZone) {
            const flowMagnitude = Math.abs(fromZone.outflow_rate) + Math.abs(toZone.inflow_rate);
            if (flowMagnitude > 5) {
                line.classList.add('active-flow');
                line.style.strokeWidth = Math.min(2 + flowMagnitude / 20, 5) + 'px';
            } else {
                line.classList.remove('active-flow');
                line.style.strokeWidth = '2px';
            }
        }
    });
}

function updateGrade() {
    const g = state.gradeResult;
    if (!g) return;

    document.getElementById('grade-letter').textContent = g.letter_grade || '—';
    document.getElementById('grade-score').textContent = `${(g.score || 0).toFixed(3)} / 1.000`;

    const comps = g.components || {};
    setCompBar('safety', comps.safety_score || 0);
    setCompBar('efficiency', comps.efficiency_score || 0);
    setCompBar('survival', comps.survival_score || 0);
    setCompBar('proactivity', comps.proactivity_score || 0);
}

function setCompBar(name, value) {
    const fill = document.getElementById(`comp-${name}`);
    const val = document.getElementById(`comp-${name}-val`);
    const pct = Math.min(value * 100, 100);
    if (fill) fill.style.width = `${pct}%`;
    if (val) val.textContent = value.toFixed(2);
}

// ─── Timeline Chart ──────────────────────────────────────────────────────────

function clearTimeline() {
    const canvas = document.getElementById('timeline-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawTimeline() {
    const canvas = document.getElementById('timeline-canvas');
    
    // Sync inner buffer with visual dimensions to prevent stretching/blurry lines
    if (canvas.clientWidth > 0 && canvas.width !== canvas.clientWidth) {
        canvas.width = canvas.clientWidth;
    }
    
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = 'rgba(255, 255, 255, 1)';
    ctx.fillRect(0, 0, W, H);

    // Threshold lines
    const maxDensity = 7.5;
    const thresholds = [
        { val: 2.0, color: 'rgba(16, 185, 129, 0.4)', textCol: '#059669', label: 'Safe' },
        { val: 3.5, color: 'rgba(245, 158, 11, 0.4)', textCol: '#d97706', label: 'Elevated' },
        { val: 5.0, color: 'rgba(239, 68, 68, 0.4)', textCol: '#b91c1c', label: 'Critical' },
    ];

    thresholds.forEach(t => {
        const y = H - (t.val / maxDensity) * H;
        ctx.strokeStyle = t.color;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = t.textCol;
        ctx.font = '600 9px Inter';
        ctx.fillText(t.label, 4, y - 3);
    });

    // Zone lines
    const zoneColors = {
        A: '#00ccb1', B: '#6366f1', C: '#ec4899',
        D: '#3b82f6', E: '#f59e0b', F: '#f97316',
    };

    ZONE_IDS.forEach(zid => {
        const data = state.timelineData[zid];
        if (!data || data.length < 2) return;

        ctx.strokeStyle = zoneColors[zid] || '#fff';
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.8;
        ctx.beginPath();

        const step = W / Math.max(state.maxTimeline - 1, 1);
        const offset = Math.max(0, state.maxTimeline - data.length);

        data.forEach((d, i) => {
            const x = (offset + i) * step;
            const y = H - (Math.min(d, maxDensity) / maxDensity) * H;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
    });

    // Legend
    let lx = W - 160;
    
    // Background behind legend to stop lines from overlapping the text
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.fillRect(lx - 5, 2, 160, 36);
    
    ctx.font = '9px Inter';
    ZONE_IDS.forEach((zid, i) => {
        const x = lx + (i % 3) * 55;
        const y = 14 + Math.floor(i / 3) * 14;
        ctx.fillStyle = zoneColors[zid];
        ctx.fillRect(x, y - 6, 8, 8);
        ctx.fillStyle = '#475569';
        ctx.font = '700 10px Inter';
        ctx.fillText(zid, x + 12, y + 1);
    });
}

// ─── Event Log ───────────────────────────────────────────────────────────────

function addEvent(text, cls = 'default') {
    const log = document.getElementById('event-log');
    const entry = document.createElement('div');
    entry.className = `event-entry ${cls}`;

    const step = state.observation ? state.observation.time_step : 0;
    entry.textContent = `[${step}] ${text}`;

    log.insertBefore(entry, log.firstChild);

    // Keep max 50 entries
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

// ─── Status ──────────────────────────────────────────────────────────────────

function setStatus(text, running = false, danger = false) {
    const pill = document.getElementById('status-pill');
    const dot = pill.querySelector('.status-dot');
    const label = pill.querySelector('.status-text');

    label.textContent = text;
    state.running = running;

    if (danger) {
        dot.style.background = '#ff0040';
        dot.style.boxShadow = '0 0 8px #ff0040';
    } else if (running) {
        dot.style.background = '#00f5d4';
        dot.style.boxShadow = '0 0 8px #00f5d4';
    } else {
        dot.style.background = '#7209b7';
        dot.style.boxShadow = '0 0 8px #7209b7';
    }
}

// ─── Auto-play ───────────────────────────────────────────────────────────────

function startAutoPlay() {
    if (state.autoPlaying) return;
    state.autoPlaying = true;

    // Update UI toggle switch
    document.getElementById('mode-toggle-input').checked = true;
    document.querySelector('.mode-toggle').classList.add('is-auto');
    document.getElementById('autoplay-overlay').style.display = 'flex';

    addEvent('▶ Auto-play started (RL model)', 'success');

    // Use async recursive loop instead of setInterval
    // This prevents network congestion on the single-threaded Python server
    async function autoPlayLoop() {
        if (!state.autoPlaying) return;
        
        if (state.episodeDone) {
            stopAutoPlay();
            return;
        }
        
        await stepEnv({ action_type: 'auto' }); // Server will use RL model
        
        if (state.autoPlaying && !state.episodeDone) {
            state.autoInterval = setTimeout(autoPlayLoop, state.speed);
        }
    }
    
    autoPlayLoop();
}

function stopAutoPlay() {
    state.autoPlaying = false;
    if (state.autoInterval) {
        clearTimeout(state.autoInterval);
        state.autoInterval = null;
    }
    
    // Update UI toggle switch
    document.getElementById('mode-toggle-input').checked = false;
    document.querySelector('.mode-toggle').classList.remove('is-auto');
    document.getElementById('autoplay-overlay').style.display = 'none';
}

// ─── Task Info ───────────────────────────────────────────────────────────────

const TASK_INFO = {
    easy: {
        name: '🟢 Matchday Warm-Up (Easy)',
        desc: 'Steady low-density crowd flow (100 steps). No surge events. Full exit capacity. Perfect for learning the basics.',
    },
    medium: {
        name: '🟡 Derby Day Rush (Medium)',
        desc: 'Multi-gate arrivals (200 steps) with 2 halftime surges. Higher volume requires proactive gate management. Exits at 75% capacity.',
    },
    hard: {
        name: '🔴 Championship Final (Hard)',
        desc: 'Massive crowds from 3 entry points (300 steps). 5 overlapping surges with high panic. Exits severely constrained (50%). Expert management only.',
    },
};

function updateTaskInfo(taskId) {
    const info = TASK_INFO[taskId] || TASK_INFO.easy;
    document.getElementById('task-name').textContent = info.name;
    document.getElementById('task-desc').textContent = info.desc;
}

// ─── Dynamic Dropdowns ───────────────────────────────────────────────────────

function updateTargetDropdown() {
    const sourceZone = document.getElementById('action-source').value;
    const targetSelect = document.getElementById('action-target');
    const neighbors = ZONE_NEIGHBORS[sourceZone] || [];
    
    targetSelect.innerHTML = '';
    neighbors.forEach(nid => {
        const opt = document.createElement('option');
        opt.value = nid;
        opt.textContent = `${nid} — ${ZONE_NAMES[nid]}`;
        targetSelect.appendChild(opt);
    });
    
    if (neighbors.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '(no neighbors)';
        targetSelect.appendChild(opt);
    }
}

function updateGateDropdown() {
    const sourceZone = document.getElementById('action-source').value;
    const gateSelect = document.getElementById('action-gate');
    const numGates = ZONE_GATE_COUNT[sourceZone] || 2;
    
    gateSelect.innerHTML = '';
    for (let i = 0; i < numGates; i++) {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `Gate ${i + 1} of ${numGates}`;
        gateSelect.appendChild(opt);
    }
}

// ─── Action Feedback Toast ───────────────────────────────────────────────────

let toastTimeout = null;
function showActionToast(message, type = 'default') {
    const toast = document.getElementById('action-toast');
    toast.textContent = message;
    toast.className = `action-toast ${type}`;
    toast.style.display = 'block';
    
    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.style.display = 'none';
    }, 4000);
}

// ─── Event Listeners ─────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Task selector
    document.querySelectorAll('.task-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentTask = btn.dataset.task;
            updateTaskInfo(state.currentTask);
            
            // Auto-reset when switching tasks for better UX
            stopAutoPlay();
            resetEnv();
        });
    });

    // Source zone change → update target and gate dropdowns
    document.getElementById('action-source').addEventListener('change', () => {
        updateTargetDropdown();
        updateGateDropdown();
    });

    // Initialize dropdowns
    updateTargetDropdown();
    updateGateDropdown();

    // Speed slider
    const speedSlider = document.getElementById('speed-slider');
    speedSlider.addEventListener('input', (e) => {
        state.speed = parseInt(e.target.value);
        document.getElementById('speed-label').textContent = `${state.speed}ms`;
        // Restart auto-play if running
        if (state.autoPlaying) {
            stopAutoPlay();
            startAutoPlay();
        }
    });

    // Reset
    document.getElementById('btn-reset').addEventListener('click', async () => {
        stopAutoPlay();
        await resetEnv();
    });

    // Mode Toggle Switch
    document.getElementById('mode-toggle-input').addEventListener('change', async (e) => {
        if (e.target.checked) {
            if (!state.running) await resetEnv();
            startAutoPlay();
        } else {
            stopAutoPlay();
            addEvent('⏸ Auto-play paused — manual control enabled', 'action');
        }
    });

    // Action buttons with feedback
    document.getElementById('btn-redirect').addEventListener('click', async () => {
        const src = document.getElementById('action-source').value;
        const tgt = document.getElementById('action-target').value;
        if (!tgt) {
            showActionToast(`❌ No valid target — ${ZONE_NAMES[src]} has no neighbors to redirect to`, 'error');
            return;
        }
        const data = await stepEnv({ action_type: 'redirect', source_zone: src, target_zone: tgt });
        showActionToast(`🔀 Redirecting crowd: ${ZONE_NAMES[src]} → ${ZONE_NAMES[tgt]} (lasts 10 steps)`, 'success');
    });

    document.getElementById('btn-close-gate').addEventListener('click', async () => {
        const src = document.getElementById('action-source').value;
        const gate = parseInt(document.getElementById('action-gate').value);
        await stepEnv({ action_type: 'gate_control', source_zone: src, gate_index: gate, gate_open: false });
        showActionToast(`🔴 Closed gate ${gate + 1} at ${ZONE_NAMES[src]} — throughput reduced`, 'default');
    });

    document.getElementById('btn-open-gate').addEventListener('click', async () => {
        const src = document.getElementById('action-source').value;
        const gate = parseInt(document.getElementById('action-gate').value);
        await stepEnv({ action_type: 'gate_control', source_zone: src, gate_index: gate, gate_open: true });
        showActionToast(`🟢 Opened gate ${gate + 1} at ${ZONE_NAMES[src]} — throughput restored`, 'success');
    });

    document.getElementById('btn-alert').addEventListener('click', async () => {
        const src = document.getElementById('action-source').value;
        await stepEnv({ action_type: 'alert', source_zone: src });
        showActionToast(`🚨 Alert toggled at ${ZONE_NAMES[src]} — inflow reduced by 40%`, 'default');
    });

    document.getElementById('btn-noop').addEventListener('click', async () => {
        await stepEnv({ action_type: 'no_op' });
        showActionToast('⏭ No action taken — simulation advanced 1 step', 'default');
    });

    // Zone click → select as source zone + update dropdowns
    document.querySelectorAll('.zone').forEach(zoneEl => {
        zoneEl.addEventListener('click', () => {
            const zid = zoneEl.dataset.zone;
            if (zid) {
                document.getElementById('action-source').value = zid;
                updateTargetDropdown();
                updateGateDropdown();
            }
        });
    });

    // Initialize
    updateTaskInfo('easy');
    setStatus('Ready');

    // Auto-reset on load
    resetEnv();
});
