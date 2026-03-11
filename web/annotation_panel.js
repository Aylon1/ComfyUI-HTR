/**
 * Sütterlin Word Annotation Panel + Transcription Review Panel
 * ComfyUI frontend extension for handwriting training data collection.
 *
 * Registers two floating panels accessible from the ComfyUI menu:
 *   ✏️ Annotate       — draw bounding boxes on line images
 *   📝 Review         — review/approve TrOCR transcriptions
 *
 * Uses the REST API at /tjk/annotation/* to load sessions and save annotations.
 */

import { app } from "../../scripts/app.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const API_BASE = "/tjk/annotation";
const CANVAS_MAX_WIDTH = 900;
const BOX_COLOR_FILL = "rgba(255, 100, 0, 0.25)";
const BOX_COLOR_STROKE = "rgba(255, 100, 0, 0.9)";
const BOX_ACTIVE_FILL = "rgba(255, 200, 0, 0.3)";
const BOX_ACTIVE_STROKE = "rgba(255, 220, 0, 1.0)";
const LABEL_BG = "rgba(255, 100, 0, 0.85)";

// ---------------------------------------------------------------------------
// AnnotationCanvas — handles all canvas drawing and mouse interaction
// ---------------------------------------------------------------------------
class AnnotationCanvas {
    constructor(canvasEl) {
        this.canvas = canvasEl;
        this.ctx = canvasEl.getContext("2d");
        this.boxes = [];          // [{x, y, w, h}, ...] in image-space coords
        this.drawing = false;
        this.startX = 0;
        this.startY = 0;
        this.currentX = 0;
        this.currentY = 0;
        this.imageScale = 1.0;   // canvas pixels per image pixel
        this.image = null;        // HTMLImageElement
        this.dpr = window.devicePixelRatio || 1;

        canvasEl.addEventListener("mousedown", this._onMouseDown.bind(this));
        canvasEl.addEventListener("mousemove", this._onMouseMove.bind(this));
        canvasEl.addEventListener("mouseup",   this._onMouseUp.bind(this));
        canvasEl.addEventListener("mouseleave", this._onMouseLeave.bind(this));
        // Prevent context menu on right-click (used for cancel)
        canvasEl.addEventListener("contextmenu", e => e.preventDefault());
    }

    /**
     * Load a line image from base64 data and set up canvas dimensions.
     * @param {string} imageB64 - base64-encoded PNG
     * @param {number} imgW - original image width in pixels
     * @param {number} imgH - original image height in pixels
     */
    loadImageFromB64(imageB64, imgW, imgH) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                this.image = img;
                // Scale to fit CANVAS_MAX_WIDTH, preserve aspect ratio
                const scale = Math.min(CANVAS_MAX_WIDTH / imgW, 1.0);
                this.imageScale = scale;

                const displayW = Math.round(imgW * scale);
                const displayH = Math.round(imgH * scale);

                // Set CSS size
                this.canvas.style.width = displayW + "px";
                this.canvas.style.height = displayH + "px";

                // Set backing store size (HiDPI)
                this.canvas.width = displayW * this.dpr;
                this.canvas.height = displayH * this.dpr;
                this.ctx.scale(this.dpr, this.dpr);

                this._redraw();
                resolve();
            };
            img.onerror = () => resolve(); // fail silently
            img.src = "data:image/png;base64," + imageB64;
        });
    }

    _getCanvasPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left),
            y: (e.clientY - rect.top),
        };
    }

    _canvasToImage(cx, cy) {
        return {
            x: Math.round(cx / this.imageScale),
            y: Math.round(cy / this.imageScale),
        };
    }

    _onMouseDown(e) {
        if (e.button === 2) {
            // Right-click: cancel current draw
            this.drawing = false;
            this._redraw();
            return;
        }
        const pos = this._getCanvasPos(e);
        const imgPos = this._canvasToImage(pos.x, pos.y);
        this.drawing = true;
        this.startX = imgPos.x;
        this.startY = imgPos.y;
        this.currentX = imgPos.x;
        this.currentY = imgPos.y;
    }

    _onMouseMove(e) {
        if (!this.drawing) return;
        const pos = this._getCanvasPos(e);
        const imgPos = this._canvasToImage(pos.x, pos.y);
        this.currentX = imgPos.x;
        this.currentY = imgPos.y;
        this._redraw();
    }

    _onMouseUp(e) {
        if (!this.drawing) return;
        this.drawing = false;

        const pos = this._getCanvasPos(e);
        const imgPos = this._canvasToImage(pos.x, pos.y);

        const x = Math.min(this.startX, imgPos.x);
        const y = Math.min(this.startY, imgPos.y);
        const w = Math.abs(imgPos.x - this.startX);
        const h = Math.abs(imgPos.y - this.startY);

        // Discard tiny boxes (< 10px in either dimension)
        if (w >= 10 && h >= 10) {
            this.boxes.push({ x, y, w, h });
        }
        this._redraw();
        // Notify panel that boxes changed
        if (this.onBoxesChanged) this.onBoxesChanged(this.boxes);
    }

    _onMouseLeave(e) {
        if (this.drawing) {
            this.drawing = false;
            this._redraw();
        }
    }

    _redraw() {
        const ctx = this.ctx;
        const scale = this.imageScale;
        const cssW = this.canvas.width / this.dpr;
        const cssH = this.canvas.height / this.dpr;

        ctx.clearRect(0, 0, cssW, cssH);

        // Draw image
        if (this.image) {
            ctx.drawImage(this.image, 0, 0, cssW, cssH);
        } else {
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(0, 0, cssW, cssH);
            ctx.fillStyle = "#888";
            ctx.font = "14px sans-serif";
            ctx.fillText("No image loaded", 20, 30);
        }

        // Check if this is a full_line annotation
        const isFullLine = this.boxes.length === 1 && this.boxes[0].full_line === true;

        if (isFullLine) {
            // Draw full-line badge overlay instead of individual boxes
            ctx.fillStyle = "rgba(0, 180, 200, 0.15)";
            ctx.fillRect(0, 0, cssW, cssH);
            ctx.strokeStyle = "rgba(0, 200, 220, 0.8)";
            ctx.lineWidth = 3;
            ctx.setLineDash([8, 4]);
            ctx.strokeRect(2, 2, cssW - 4, cssH - 4);
            ctx.setLineDash([]);

            // Badge text
            const badgeText = "📄 FULL LINE";
            ctx.font = "bold 16px monospace";
            const tw = ctx.measureText(badgeText).width + 16;
            const th = 28;
            const bx = (cssW - tw) / 2;
            const by = (cssH - th) / 2;
            ctx.fillStyle = "rgba(0, 160, 180, 0.9)";
            ctx.beginPath();
            ctx.roundRect(bx, by, tw, th, 6);
            ctx.fill();
            ctx.fillStyle = "#fff";
            ctx.fillText(badgeText, bx + 8, by + 19);
        } else {
            // Draw saved boxes normally
            this.boxes.forEach((box, idx) => {
                const cx = box.x * scale;
                const cy = box.y * scale;
                const cw = box.w * scale;
                const ch = box.h * scale;

                ctx.fillStyle = BOX_COLOR_FILL;
                ctx.fillRect(cx, cy, cw, ch);
                ctx.strokeStyle = BOX_COLOR_STROKE;
                ctx.lineWidth = 1.5;
                ctx.strokeRect(cx, cy, cw, ch);

                // Index label
                ctx.fillStyle = LABEL_BG;
                const label = String(idx + 1);
                ctx.font = "bold 11px monospace";
                const tw = ctx.measureText(label).width + 6;
                ctx.fillRect(cx, cy, tw, 16);
                ctx.fillStyle = "#fff";
                ctx.fillText(label, cx + 3, cy + 12);
            });
        }

        // Draw current rubber-band box
        if (this.drawing) {
            const x = Math.min(this.startX, this.currentX) * scale;
            const y = Math.min(this.startY, this.currentY) * scale;
            const w = Math.abs(this.currentX - this.startX) * scale;
            const h = Math.abs(this.currentY - this.startY) * scale;

            ctx.fillStyle = BOX_ACTIVE_FILL;
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = BOX_ACTIVE_STROKE;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
        }
    }

    undo() {
        if (this.boxes.length > 0) {
            this.boxes.pop();
            this._redraw();
            if (this.onBoxesChanged) this.onBoxesChanged(this.boxes);
        }
    }

    clear() {
        this.boxes = [];
        this._redraw();
        if (this.onBoxesChanged) this.onBoxesChanged(this.boxes);
    }

    getBoxes() { return [...this.boxes]; }

    setBoxes(boxes) {
        this.boxes = boxes ? [...boxes] : [];
        this._redraw();
    }
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function apiGet(path) {
    const resp = await fetch(API_BASE + path);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: resp.statusText }));
        throw new Error(err.error || resp.statusText);
    }
    return resp.json();
}

async function apiPost(path, body) {
    const resp = await fetch(API_BASE + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: resp.statusText }));
        throw new Error(err.error || resp.statusText);
    }
    return resp.json();
}

async function apiDelete(path) {
    const resp = await fetch(API_BASE + path, { method: "DELETE" });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: resp.statusText }));
        throw new Error(err.error || resp.statusText);
    }
    return resp.json();
}

// ---------------------------------------------------------------------------
// AnnotationPanel — the full floating annotation UI
// ---------------------------------------------------------------------------
class AnnotationPanel {
    constructor() {
        this.sessionId = null;
        this.sessionData = null;
        this.currentLineIndex = 0;
        this.totalLines = 0;
        this.annotationCanvas = null;
        this.panelEl = null;
        this.visible = false;
        this._keyHandler = this._onKeyDown.bind(this);
    }

    // -----------------------------------------------------------------------
    // Build DOM
    // -----------------------------------------------------------------------
    _buildPanel() {
        const panel = document.createElement("div");
        panel.id = "tjk-annotation-panel";
        panel.style.cssText = `
            position: fixed;
            top: 60px;
            left: 50%;
            transform: translateX(-50%);
            width: min(960px, 96vw);
            max-height: 90vh;
            overflow-y: auto;
            background: #1e1e2e;
            border: 1px solid #444;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.7);
            z-index: 9999;
            font-family: 'Segoe UI', system-ui, sans-serif;
            font-size: 13px;
            color: #cdd6f4;
            display: none;
        `;

        panel.innerHTML = `
            <div id="tjk-ann-header" style="
                display:flex; align-items:center; justify-content:space-between;
                padding:10px 16px; background:#181825; border-radius:8px 8px 0 0;
                border-bottom:1px solid #333; cursor:move; user-select:none;">
                <span style="font-weight:700; font-size:15px;">✏️ Sütterlin Word Annotator</span>
                <div style="display:flex; gap:8px; align-items:center;">
                    <select id="tjk-session-select" style="
                        background:#313244; color:#cdd6f4; border:1px solid #555;
                        border-radius:4px; padding:3px 8px; font-size:12px;">
                        <option value="">— select session —</option>
                    </select>
                    <button id="tjk-refresh-sessions" title="Refresh sessions" style="
                        background:#313244; color:#cdd6f4; border:1px solid #555;
                        border-radius:4px; padding:3px 8px; cursor:pointer;">⟳</button>
                    <button id="tjk-ann-close" style="
                        background:#f38ba8; color:#1e1e2e; border:none;
                        border-radius:4px; padding:3px 10px; cursor:pointer; font-weight:700;">✕</button>
                </div>
            </div>

            <div style="padding:10px 16px 4px;">
                <!-- Progress bar -->
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                    <div style="flex:1; background:#313244; border-radius:4px; height:8px; overflow:hidden;">
                        <div id="tjk-progress-bar" style="height:100%; background:#a6e3a1; width:0%; transition:width 0.3s;"></div>
                    </div>
                    <span id="tjk-progress-text" style="font-size:12px; color:#a6adc8; white-space:nowrap;">0 / 0</span>
                </div>

                <!-- Navigation row -->
                <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:8px;">
                    <button id="tjk-btn-prev" style="
                        background:#313244; color:#cdd6f4; border:1px solid #555;
                        border-radius:4px; padding:4px 12px; cursor:pointer;">◀ Prev</button>
                    <span id="tjk-line-label" style="font-size:13px; color:#cdd6f4; min-width:100px; text-align:center;">
                        Line — / —
                    </span>
                    <button id="tjk-btn-next" style="
                        background:#313244; color:#cdd6f4; border:1px solid #555;
                        border-radius:4px; padding:4px 12px; cursor:pointer;">Next ▶</button>
                    <button id="tjk-btn-skip" style="
                        background:#45475a; color:#cdd6f4; border:1px solid #555;
                        border-radius:4px; padding:4px 12px; cursor:pointer;">Skip (S)</button>
                    <button id="tjk-btn-clear" style="
                        background:#45475a; color:#f38ba8; border:1px solid #555;
                        border-radius:4px; padding:4px 12px; cursor:pointer;">Clear (C)</button>
                    <button id="tjk-btn-full-line" style="
                        background:#0097a7; color:#fff; border:none;
                        border-radius:4px; padding:4px 14px; cursor:pointer; font-weight:700;">
                        📄 Full Line (F)
                    </button>
                    <button id="tjk-btn-save-next" style="
                        background:#a6e3a1; color:#1e1e2e; border:none;
                        border-radius:4px; padding:4px 14px; cursor:pointer; font-weight:700;">
                        Save &amp; Next (Space)
                    </button>
                </div>

                <!-- Error/status message -->
                <div id="tjk-status-msg" style="
                    display:none; padding:6px 10px; border-radius:4px;
                    background:#f38ba822; color:#f38ba8; font-size:12px; margin-bottom:6px;">
                </div>
            </div>

            <!-- Canvas area -->
            <div style="padding:0 16px 8px; overflow-x:auto;">
                <canvas id="tjk-ann-canvas" style="
                    display:block; cursor:crosshair;
                    border:1px solid #444; border-radius:4px;
                    background:#0d0d1a;
                    max-width:100%;"></canvas>
            </div>

            <!-- Box list -->
            <div style="padding:4px 16px 12px;">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:12px; color:#a6adc8;">
                        Boxes drawn: <strong id="tjk-box-count">0</strong>
                    </span>
                    <div style="display:flex; gap:6px;">
                        <button id="tjk-btn-undo" style="
                            background:#313244; color:#cdd6f4; border:1px solid #555;
                            border-radius:4px; padding:3px 10px; cursor:pointer; font-size:12px;">
                            Undo (U)
                        </button>
                    </div>
                </div>
                <div id="tjk-box-list" style="
                    display:flex; flex-wrap:wrap; gap:4px; min-height:24px;"></div>
                <div style="margin-top:8px; font-size:11px; color:#585b70;">
                    Keyboard: <kbd>Space</kbd>=Save&amp;Next &nbsp;
                    <kbd>F</kbd>=Full Line &nbsp;
                    <kbd>S</kbd>=Skip &nbsp;
                    <kbd>U</kbd>=Undo &nbsp;
                    <kbd>C</kbd>=Clear &nbsp;
                    <kbd>←</kbd><kbd>→</kbd>=Navigate &nbsp;
                    <kbd>Esc</kbd>=Close
                </div>
            </div>
        `;

        return panel;
    }

    // -----------------------------------------------------------------------
    // Show / hide
    // -----------------------------------------------------------------------
    show() {
        if (!this.panelEl) {
            this.panelEl = this._buildPanel();
            document.body.appendChild(this.panelEl);
            this._bindEvents();
        }
        this.panelEl.style.display = "block";
        this.visible = true;
        document.addEventListener("keydown", this._keyHandler);
        this._loadSessions();
    }

    hide() {
        if (this.panelEl) this.panelEl.style.display = "none";
        this.visible = false;
        document.removeEventListener("keydown", this._keyHandler);
    }

    toggle() {
        if (this.visible) this.hide(); else this.show();
    }

    // -----------------------------------------------------------------------
    // Event binding
    // -----------------------------------------------------------------------
    _bindEvents() {
        const $ = id => this.panelEl.querySelector("#" + id);

        $("tjk-ann-close").addEventListener("click", () => this.hide());
        $("tjk-refresh-sessions").addEventListener("click", () => this._loadSessions());
        $("tjk-session-select").addEventListener("change", e => {
            if (e.target.value) this._loadSession(e.target.value);
        });

        $("tjk-btn-prev").addEventListener("click", () => this._navigate(-1));
        $("tjk-btn-next").addEventListener("click", () => this._navigate(+1));
        $("tjk-btn-skip").addEventListener("click", () => this._skipLine());
        $("tjk-btn-clear").addEventListener("click", () => {
            if (this.annotationCanvas) this.annotationCanvas.clear();
        });
        $("tjk-btn-full-line").addEventListener("click", () => this._saveFullLine());
        $("tjk-btn-save-next").addEventListener("click", () => this._saveAndNext());
        $("tjk-btn-undo").addEventListener("click", () => {
            if (this.annotationCanvas) this.annotationCanvas.undo();
        });

        // Set up canvas
        const canvasEl = $("tjk-ann-canvas");
        this.annotationCanvas = new AnnotationCanvas(canvasEl);
        this.annotationCanvas.onBoxesChanged = (boxes) => this._updateBoxList(boxes);

        // Make panel draggable by header
        this._makeDraggable($("tjk-ann-header"), this.panelEl);
    }

    _makeDraggable(handle, panel) {
        let dragging = false, ox = 0, oy = 0;
        handle.addEventListener("mousedown", e => {
            if (e.target.tagName === "BUTTON" || e.target.tagName === "SELECT") return;
            dragging = true;
            const rect = panel.getBoundingClientRect();
            ox = e.clientX - rect.left;
            oy = e.clientY - rect.top;
            e.preventDefault();
        });
        document.addEventListener("mousemove", e => {
            if (!dragging) return;
            panel.style.left = (e.clientX - ox) + "px";
            panel.style.top = (e.clientY - oy) + "px";
            panel.style.transform = "none";
        });
        document.addEventListener("mouseup", () => { dragging = false; });
    }

    // -----------------------------------------------------------------------
    // Session loading
    // -----------------------------------------------------------------------
    async _loadSessions() {
        try {
            const sessions = await apiGet("/sessions");
            const select = this.panelEl.querySelector("#tjk-session-select");
            const current = select.value;
            select.innerHTML = '<option value="">— select session —</option>';
            sessions.forEach(s => {
                const opt = document.createElement("option");
                opt.value = s.session_id;
                const pct = s.total_lines > 0
                    ? Math.round((s.annotated_count || 0) / s.total_lines * 100)
                    : 0;
                opt.textContent = `${s.session_id} (${s.annotated_count || 0}/${s.total_lines} — ${pct}%)`;
                if (s.session_id === current) opt.selected = true;
                select.appendChild(opt);
            });
            // Auto-select if only one session
            if (sessions.length === 1 && !this.sessionId) {
                select.value = sessions[0].session_id;
                await this._loadSession(sessions[0].session_id);
            }
        } catch (err) {
            this._showError("Could not load sessions: " + err.message);
        }
    }

    async _loadSession(sessionId) {
        this._showError(null);
        try {
            const data = await apiGet(`/session/${sessionId}`);
            this.sessionId = sessionId;
            this.sessionData = data;
            this.totalLines = data.total_lines || 0;

            // Find first pending line
            const annotations = data.annotations || {};
            let startLine = data.current_line_index || 0;
            // Try to find first unannotated line
            for (let i = 0; i < this.totalLines; i++) {
                const ann = annotations[String(i)];
                if (!ann || ann.status === "pending") {
                    startLine = i;
                    break;
                }
            }

            await this._loadLine(startLine);
        } catch (err) {
            this._showError("Could not load session: " + err.message);
        }
    }

    async _loadLine(lineIndex) {
        if (!this.sessionId) return;
        if (lineIndex < 0) lineIndex = 0;
        if (lineIndex >= this.totalLines) lineIndex = this.totalLines - 1;

        this.currentLineIndex = lineIndex;
        this._updateNavUI();

        try {
            const data = await apiGet(`/session/${this.sessionId}/line/${lineIndex}`);

            // Load image into canvas
            if (this.annotationCanvas && data.image_b64) {
                await this.annotationCanvas.loadImageFromB64(
                    data.image_b64,
                    data.width,
                    data.height
                );
            }

            // Restore existing boxes
            const boxes = (data.annotation && data.annotation.boxes) ? data.annotation.boxes : [];
            if (this.annotationCanvas) {
                this.annotationCanvas.setBoxes(boxes);
            }
            this._updateBoxList(boxes);
            this._updateProgress();
        } catch (err) {
            this._showError("Could not load line: " + err.message);
        }
    }

    // -----------------------------------------------------------------------
    // Actions
    // -----------------------------------------------------------------------
    async _saveAndNext() {
        if (!this.sessionId) return;
        const boxes = this.annotationCanvas ? this.annotationCanvas.getBoxes() : [];

        try {
            await apiPost(`/session/${this.sessionId}/annotate`, {
                line_index: this.currentLineIndex,
                boxes: boxes,
            });
            await apiPost(`/session/${this.sessionId}/advance`, {});
            this._updateProgress();

            // Move to next pending line
            const nextLine = await this._findNextPending(this.currentLineIndex + 1);
            if (nextLine !== null) {
                await this._loadLine(nextLine);
            } else {
                this._showComplete();
            }
        } catch (err) {
            this._showError("Save failed: " + err.message);
        }
    }

    async _saveFullLine() {
        if (!this.sessionId) return;

        // Disable button briefly to prevent double-click
        const btn = this.panelEl && this.panelEl.querySelector("#tjk-btn-full-line");
        if (btn) { btn.disabled = true; btn.textContent = "📄 Saving…"; }

        try {
            await apiPost(`/session/${this.sessionId}/annotate`, {
                line_index: this.currentLineIndex,
                boxes: [{ x: 0, y: 0, w: -1, h: -1, full_line: true }],
            });
            await apiPost(`/session/${this.sessionId}/advance`, {});
            this._updateProgress();

            // Show brief success feedback, then advance
            this._showStatus("✅ Full line saved!", "success");
            await new Promise(r => setTimeout(r, 600));
            this._showStatus(null);

            // Move to next pending line
            const nextLine = await this._findNextPending(this.currentLineIndex + 1);
            if (nextLine !== null) {
                await this._loadLine(nextLine);
            } else {
                this._showComplete();
            }
        } catch (err) {
            this._showError("Save Full Line failed: " + err.message);
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = "📄 Full Line (F)"; }
        }
    }

    async _skipLine() {
        if (!this.sessionId) return;
        try {
            await apiPost(`/session/${this.sessionId}/line/${this.currentLineIndex}/skip`, {});
            const nextLine = await this._findNextPending(this.currentLineIndex + 1);
            if (nextLine !== null) {
                await this._loadLine(nextLine);
            } else {
                this._showComplete();
            }
        } catch (err) {
            this._showError("Skip failed: " + err.message);
        }
    }

    async _navigate(delta) {
        const newIndex = this.currentLineIndex + delta;
        if (newIndex >= 0 && newIndex < this.totalLines) {
            await this._loadLine(newIndex);
        }
    }

    async _findNextPending(fromIndex) {
        try {
            const data = await apiGet(`/session/${this.sessionId}`);
            const annotations = data.annotations || {};
            for (let i = fromIndex; i < this.totalLines; i++) {
                const ann = annotations[String(i)];
                if (!ann || ann.status === "pending") return i;
            }
            return null;
        } catch {
            return null;
        }
    }

    async _deleteBox(lineIndex, boxIndex) {
        if (!this.sessionId) return;
        try {
            await apiDelete(
                `/session/${this.sessionId}/line/${lineIndex}/box/${boxIndex}`
            );
            // Reload current line to sync
            await this._loadLine(this.currentLineIndex);
        } catch (err) {
            this._showError("Delete failed: " + err.message);
        }
    }

    // -----------------------------------------------------------------------
    // UI updates
    // -----------------------------------------------------------------------
    _updateNavUI() {
        const label = this.panelEl.querySelector("#tjk-line-label");
        if (label) {
            label.textContent = `Line ${this.currentLineIndex + 1} of ${this.totalLines}`;
        }
    }

    async _updateProgress() {
        if (!this.sessionId) return;
        try {
            const data = await apiGet(`/session/${this.sessionId}`);
            const annotations = data.annotations || {};
            const annotated = Object.values(annotations).filter(
                a => a.status === "annotated"
            ).length;
            const skipped = Object.values(annotations).filter(
                a => a.status === "skipped"
            ).length;
            const total = data.total_lines || 0;
            const done = annotated + skipped;
            const pct = total > 0 ? (done / total * 100).toFixed(0) : 0;

            const bar = this.panelEl.querySelector("#tjk-progress-bar");
            const text = this.panelEl.querySelector("#tjk-progress-text");
            if (bar) bar.style.width = pct + "%";
            if (text) text.textContent = `${annotated} annotated, ${skipped} skipped / ${total} total`;
        } catch { /* ignore */ }
    }

    _updateBoxList(boxes) {
        const countEl = this.panelEl.querySelector("#tjk-box-count");
        const listEl = this.panelEl.querySelector("#tjk-box-list");
        if (countEl) countEl.textContent = boxes.length;
        if (!listEl) return;

        listEl.innerHTML = "";
        boxes.forEach((box, idx) => {
            const chip = document.createElement("div");
            chip.style.cssText = `
                display:inline-flex; align-items:center; gap:4px;
                background:#313244; border:1px solid #555; border-radius:4px;
                padding:2px 6px; font-size:11px; color:#cdd6f4;
            `;
            chip.innerHTML = `
                <span style="color:#fab387;">#${idx + 1}</span>
                <span>${box.x},${box.y} ${box.w}×${box.h}</span>
                <button data-idx="${idx}" style="
                    background:none; border:none; color:#f38ba8;
                    cursor:pointer; padding:0 2px; font-size:12px; line-height:1;">❌</button>
            `;
            chip.querySelector("button").addEventListener("click", () => {
                if (this.annotationCanvas) {
                    this.annotationCanvas.boxes.splice(idx, 1);
                    this.annotationCanvas._redraw();
                    this._updateBoxList(this.annotationCanvas.getBoxes());
                }
            });
            listEl.appendChild(chip);
        });
    }

    _showError(msg) {
        const el = this.panelEl && this.panelEl.querySelector("#tjk-status-msg");
        if (!el) return;
        if (msg) {
            el.textContent = "⚠ " + msg;
            el.style.background = "#f38ba822";
            el.style.color = "#f38ba8";
            el.style.display = "block";
        } else {
            el.style.display = "none";
        }
    }

    _showStatus(msg, type) {
        const el = this.panelEl && this.panelEl.querySelector("#tjk-status-msg");
        if (!el) return;
        if (msg) {
            el.textContent = msg;
            el.style.background = type === "success" ? "#a6e3a122" : "#89b4fa22";
            el.style.color     = type === "success" ? "#a6e3a1"   : "#89b4fa";
            el.style.display = "block";
        } else {
            el.style.display = "none";
        }
    }

    _showComplete() {
        this._showError(null);
        const label = this.panelEl.querySelector("#tjk-line-label");
        if (label) label.textContent = "✅ All lines annotated!";
        const bar = this.panelEl.querySelector("#tjk-progress-bar");
        if (bar) { bar.style.width = "100%"; bar.style.background = "#a6e3a1"; }
    }

    // -----------------------------------------------------------------------
    // Keyboard shortcuts
    // -----------------------------------------------------------------------
    _onKeyDown(e) {
        // Don't intercept when typing in an input/select
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" ||
            e.target.tagName === "SELECT") return;

        switch (e.key) {
            case " ":
            case "Enter":
                e.preventDefault();
                this._saveAndNext();
                break;
            case "f":
            case "F":
                e.preventDefault();
                this._saveFullLine();
                break;
            case "s":
            case "S":
                e.preventDefault();
                this._skipLine();
                break;
            case "u":
            case "U":
                e.preventDefault();
                if (this.annotationCanvas) this.annotationCanvas.undo();
                break;
            case "c":
            case "C":
                e.preventDefault();
                if (this.annotationCanvas) this.annotationCanvas.clear();
                break;
            case "ArrowLeft":
                e.preventDefault();
                this._navigate(-1);
                break;
            case "ArrowRight":
                e.preventDefault();
                this._navigate(+1);
                break;
            case "Escape":
                this.hide();
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// ReviewPanel — Transcription Review UI
// ---------------------------------------------------------------------------
class ReviewPanel {
    constructor() {
        this.label = "handwritten";
        this.outputDir = "annotation_output";
        this.crops = [];
        this.currentIndex = 0;
        this.stats = { total: 0, pending: 0, approved: 0, rejected: 0, skipped: 0 };
        this.panelEl = null;
        this.visible = false;
        this._keyHandler = this._onKeyDown.bind(this);
    }

    // -----------------------------------------------------------------------
    // Build DOM
    // -----------------------------------------------------------------------
    _buildPanel() {
        const panel = document.createElement("div");
        panel.id = "tjk-review-panel";
        panel.style.cssText = `
            position: fixed;
            top: 60px;
            left: 50%;
            transform: translateX(-50%);
            width: min(700px, 96vw);
            max-height: 92vh;
            overflow-y: auto;
            background: #1e1e2e;
            border: 1px solid #444;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.7);
            z-index: 9999;
            font-family: 'Segoe UI', system-ui, sans-serif;
            font-size: 13px;
            color: #cdd6f4;
            display: none;
        `;

        panel.innerHTML = `
            <!-- Header -->
            <div id="tjk-rev-header" style="
                display:flex; align-items:center; justify-content:space-between;
                padding:10px 16px; background:#181825; border-radius:8px 8px 0 0;
                border-bottom:1px solid #333; cursor:move; user-select:none;">
                <span style="font-weight:700; font-size:15px;">📝 Transcription Review</span>
                <div style="display:flex; gap:8px; align-items:center;">
                    <select id="tjk-rev-label-select" style="
                        background:#313244; color:#cdd6f4; border:1px solid #555;
                        border-radius:4px; padding:3px 8px; font-size:12px;">
                        <option value="handwritten">handwritten</option>
                        <option value="printed">printed</option>
                    </select>
                    <input id="tjk-rev-output-dir" type="text" value="annotation_output"
                        placeholder="output_dir" style="
                        background:#313244; color:#cdd6f4; border:1px solid #555;
                        border-radius:4px; padding:3px 8px; font-size:12px; width:140px;" />
                    <button id="tjk-rev-load" style="
                        background:#89b4fa; color:#1e1e2e; border:none;
                        border-radius:4px; padding:3px 10px; cursor:pointer; font-weight:700;">Load</button>
                    <button id="tjk-rev-close" style="
                        background:#f38ba8; color:#1e1e2e; border:none;
                        border-radius:4px; padding:3px 10px; cursor:pointer; font-weight:700;">✕</button>
                </div>
            </div>

            <div style="padding:12px 16px 4px;">
                <!-- Progress bar -->
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                    <div style="flex:1; background:#313244; border-radius:4px; height:8px; overflow:hidden;">
                        <div id="tjk-rev-progress-bar" style="height:100%; background:#a6e3a1; width:0%; transition:width 0.3s;"></div>
                    </div>
                    <span id="tjk-rev-progress-text" style="font-size:12px; color:#a6adc8; white-space:nowrap;">0 / 0</span>
                </div>
                <!-- Stats line -->
                <div id="tjk-rev-stats" style="font-size:11px; color:#585b70; margin-bottom:10px;">
                    — pending | — approved | — rejected | — skipped
                </div>

                <!-- Navigation -->
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                    <button id="tjk-rev-btn-prev" style="
                        background:#313244; color:#cdd6f4; border:1px solid #555;
                        border-radius:4px; padding:4px 12px; cursor:pointer;">◀</button>
                    <span id="tjk-rev-crop-label" style="
                        font-size:12px; color:#a6adc8; min-width:120px; text-align:center;">
                        — / —
                    </span>
                    <button id="tjk-rev-btn-next" style="
                        background:#313244; color:#cdd6f4; border:1px solid #555;
                        border-radius:4px; padding:4px 12px; cursor:pointer;">▶</button>
                    <span id="tjk-rev-filename" style="font-size:11px; color:#585b70; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"></span>
                </div>

                <!-- Crop image display -->
                <div style="text-align:center; margin-bottom:12px; min-height:80px;
                    background:#0d0d1a; border:1px solid #333; border-radius:6px; padding:8px;">
                    <img id="tjk-rev-crop-img" src="" alt="crop"
                        style="max-width:600px; max-height:300px; object-fit:contain;
                        display:none; border-radius:4px;" />
                    <div id="tjk-rev-no-image" style="color:#585b70; padding:20px; font-size:12px;">
                        No crop loaded
                    </div>
                </div>

                <!-- Status badge -->
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    <span style="font-size:12px; color:#a6adc8;">Status:</span>
                    <span id="tjk-rev-status-badge" style="
                        font-size:12px; font-weight:700; padding:2px 8px;
                        border-radius:4px; background:#313244; color:#cdd6f4;">pending</span>
                </div>

                <!-- Transcription text input -->
                <div style="margin-bottom:10px;">
                    <label style="font-size:12px; color:#a6adc8; display:block; margin-bottom:4px;">
                        Transcription (Tab to focus, Enter to accept):
                    </label>
                    <input id="tjk-rev-text-input" type="text" style="
                        width:100%; box-sizing:border-box;
                        background:#313244; color:#cdd6f4;
                        border:1px solid #555; border-radius:6px;
                        padding:8px 12px; font-size:18px;
                        font-family: 'Segoe UI', system-ui, sans-serif;
                        outline:none;" placeholder="Transcription…" />
                </div>

                <!-- Action buttons -->
                <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">
                    <button id="tjk-rev-btn-accept" style="
                        flex:1; min-width:100px;
                        background:#a6e3a1; color:#1e1e2e; border:none;
                        border-radius:6px; padding:8px 16px; cursor:pointer;
                        font-weight:700; font-size:13px;">
                        ✅ Accept (Enter)
                    </button>
                    <button id="tjk-rev-btn-edit" style="
                        flex:1; min-width:100px;
                        background:#89b4fa; color:#1e1e2e; border:none;
                        border-radius:6px; padding:8px 16px; cursor:pointer;
                        font-weight:700; font-size:13px;">
                        ✏️ Edit &amp; Accept (Tab)
                    </button>
                    <button id="tjk-rev-btn-reject" style="
                        flex:1; min-width:100px;
                        background:#f38ba8; color:#1e1e2e; border:none;
                        border-radius:6px; padding:8px 16px; cursor:pointer;
                        font-weight:700; font-size:13px;">
                        ❌ Reject (R)
                    </button>
                    <button id="tjk-rev-btn-skip" style="
                        flex:1; min-width:100px;
                        background:#45475a; color:#cdd6f4; border:1px solid #555;
                        border-radius:6px; padding:8px 16px; cursor:pointer;
                        font-weight:700; font-size:13px;">
                        ⏭️ Skip (S)
                    </button>
                </div>

                <!-- Error/status message -->
                <div id="tjk-rev-msg" style="
                    display:none; padding:6px 10px; border-radius:4px;
                    background:#f38ba822; color:#f38ba8; font-size:12px; margin-bottom:6px;">
                </div>

                <!-- Keyboard hint -->
                <div style="font-size:11px; color:#585b70; margin-bottom:8px;">
                    Keyboard: <kbd>Enter</kbd>=Accept &nbsp;
                    <kbd>Tab</kbd>=Edit &nbsp;
                    <kbd>R</kbd>=Reject &nbsp;
                    <kbd>S</kbd>=Skip &nbsp;
                    <kbd>←</kbd><kbd>→</kbd>=Navigate &nbsp;
                    <kbd>Esc</kbd>=Close
                </div>
            </div>
        `;

        return panel;
    }

    // -----------------------------------------------------------------------
    // Show / hide
    // -----------------------------------------------------------------------
    show() {
        if (!this.panelEl) {
            this.panelEl = this._buildPanel();
            document.body.appendChild(this.panelEl);
            this._bindEvents();
        }
        this.panelEl.style.display = "block";
        this.visible = true;
        document.addEventListener("keydown", this._keyHandler);
    }

    hide() {
        if (this.panelEl) this.panelEl.style.display = "none";
        this.visible = false;
        document.removeEventListener("keydown", this._keyHandler);
    }

    toggle() {
        if (this.visible) this.hide(); else this.show();
    }

    // -----------------------------------------------------------------------
    // Event binding
    // -----------------------------------------------------------------------
    _bindEvents() {
        const $ = id => this.panelEl.querySelector("#" + id);

        $("tjk-rev-close").addEventListener("click", () => this.hide());
        $("tjk-rev-load").addEventListener("click", () => this._loadReview());

        $("tjk-rev-btn-prev").addEventListener("click", () => this._navigateTo(this.currentIndex - 1));
        $("tjk-rev-btn-next").addEventListener("click", () => this._navigateTo(this.currentIndex + 1));

        $("tjk-rev-btn-accept").addEventListener("click", () => this._accept());
        $("tjk-rev-btn-edit").addEventListener("click", () => {
            $("tjk-rev-text-input").focus();
            $("tjk-rev-text-input").select();
        });
        $("tjk-rev-btn-reject").addEventListener("click", () => this._reject());
        $("tjk-rev-btn-skip").addEventListener("click", () => this._skip());

        // Accept on Enter inside text input
        $("tjk-rev-text-input").addEventListener("keydown", e => {
            if (e.key === "Enter") {
                e.preventDefault();
                e.stopPropagation();
                this._accept();
            }
        });

        // Make panel draggable by header
        this._makeDraggable($("tjk-rev-header"), this.panelEl);
    }

    _makeDraggable(handle, panel) {
        let dragging = false, ox = 0, oy = 0;
        handle.addEventListener("mousedown", e => {
            if (e.target.tagName === "BUTTON" || e.target.tagName === "SELECT" ||
                e.target.tagName === "INPUT") return;
            dragging = true;
            const rect = panel.getBoundingClientRect();
            ox = e.clientX - rect.left;
            oy = e.clientY - rect.top;
            e.preventDefault();
        });
        document.addEventListener("mousemove", e => {
            if (!dragging) return;
            panel.style.left = (e.clientX - ox) + "px";
            panel.style.top = (e.clientY - oy) + "px";
            panel.style.transform = "none";
        });
        document.addEventListener("mouseup", () => { dragging = false; });
    }

    // -----------------------------------------------------------------------
    // Data loading
    // -----------------------------------------------------------------------
    async _loadReview() {
        const labelSel = this.panelEl.querySelector("#tjk-rev-label-select");
        const outputDirInput = this.panelEl.querySelector("#tjk-rev-output-dir");
        this.label = labelSel ? labelSel.value : "handwritten";
        this.outputDir = outputDirInput ? outputDirInput.value.trim() : "annotation_output";

        this._showMsg(null);
        try {
            const data = await apiGet(`/review/${this.label}?output_dir=${encodeURIComponent(this.outputDir)}`);
            this.crops = data.crops || [];
            this._showMsg(null);

            if (this.crops.length === 0) {
                this._showMsg("No crops found. Run TranscriptionReviewNode first.", "info");
                this._updateProgressUI();
                return;
            }

            // Find first pending crop
            const firstPending = this.crops.findIndex(c => c.status === "pending");
            const startIdx = firstPending >= 0 ? firstPending : 0;
            await this._loadCrop(startIdx);
            await this._refreshStats();
        } catch (err) {
            this._showMsg("Could not load review session: " + err.message);
        }
    }

    async _loadCrop(index) {
        if (index < 0) index = 0;
        if (index >= this.crops.length) index = this.crops.length - 1;
        if (this.crops.length === 0) return;

        this.currentIndex = index;
        this._showMsg(null);

        try {
            const data = await apiGet(
                `/review/${this.label}/crop/${index}?output_dir=${encodeURIComponent(this.outputDir)}`
            );

            // Update image
            const img = this.panelEl.querySelector("#tjk-rev-crop-img");
            const noImg = this.panelEl.querySelector("#tjk-rev-no-image");
            if (data.image_b64) {
                img.src = "data:image/png;base64," + data.image_b64;
                img.style.display = "block";
                noImg.style.display = "none";
            } else {
                img.style.display = "none";
                noImg.style.display = "block";
            }

            // Update text input
            const textInput = this.panelEl.querySelector("#tjk-rev-text-input");
            if (textInput) {
                const displayText = data.approved_text !== null && data.approved_text !== undefined
                    ? data.approved_text
                    : (data.predicted_text || "");
                textInput.value = displayText;
            }

            // Update filename
            const fnEl = this.panelEl.querySelector("#tjk-rev-filename");
            if (fnEl) fnEl.textContent = data.filename || "";

            // Update status badge
            this._updateStatusBadge(data.status || "pending");

            // Update nav label
            const navLabel = this.panelEl.querySelector("#tjk-rev-crop-label");
            if (navLabel) navLabel.textContent = `${index + 1} / ${this.crops.length}`;

            // Sync local crops array
            if (this.crops[index]) {
                this.crops[index].status = data.status;
                this.crops[index].approved_text = data.approved_text;
                this.crops[index].predicted_text = data.predicted_text;
            }

        } catch (err) {
            this._showMsg("Could not load crop: " + err.message);
        }
    }

    async _refreshStats() {
        try {
            const stats = await apiGet(
                `/review/${this.label}/stats?output_dir=${encodeURIComponent(this.outputDir)}`
            );
            this.stats = stats;
            this._updateProgressUI();
        } catch { /* ignore */ }
    }

    // -----------------------------------------------------------------------
    // Actions
    // -----------------------------------------------------------------------
    async _accept() {
        if (this.crops.length === 0) return;
        const textInput = this.panelEl.querySelector("#tjk-rev-text-input");
        const text = textInput ? textInput.value : "";

        try {
            await apiPost(`/review/${this.label}/approve?output_dir=${encodeURIComponent(this.outputDir)}`, {
                index: this.currentIndex,
                text: text,
            });
            if (this.crops[this.currentIndex]) {
                this.crops[this.currentIndex].status = "approved";
                this.crops[this.currentIndex].approved_text = text;
            }
            this._updateStatusBadge("approved");
            await this._refreshStats();
            // Advance to next pending
            const next = this._findNextPending(this.currentIndex + 1);
            if (next !== null) {
                await this._loadCrop(next);
            } else {
                this._showMsg("✅ All crops reviewed!", "success");
            }
        } catch (err) {
            this._showMsg("Accept failed: " + err.message);
        }
    }

    async _reject() {
        if (this.crops.length === 0) return;
        try {
            await apiPost(`/review/${this.label}/reject?output_dir=${encodeURIComponent(this.outputDir)}`, {
                index: this.currentIndex,
            });
            if (this.crops[this.currentIndex]) {
                this.crops[this.currentIndex].status = "rejected";
            }
            this._updateStatusBadge("rejected");
            await this._refreshStats();
            const next = this._findNextPending(this.currentIndex + 1);
            if (next !== null) {
                await this._loadCrop(next);
            } else {
                this._showMsg("All crops reviewed!", "success");
            }
        } catch (err) {
            this._showMsg("Reject failed: " + err.message);
        }
    }

    async _skip() {
        if (this.crops.length === 0) return;
        try {
            await apiPost(`/review/${this.label}/skip?output_dir=${encodeURIComponent(this.outputDir)}`, {
                index: this.currentIndex,
            });
            if (this.crops[this.currentIndex]) {
                this.crops[this.currentIndex].status = "skipped";
            }
            this._updateStatusBadge("skipped");
            await this._refreshStats();
            const next = this._findNextPending(this.currentIndex + 1);
            if (next !== null) {
                await this._loadCrop(next);
            } else {
                this._showMsg("All crops reviewed!", "success");
            }
        } catch (err) {
            this._showMsg("Skip failed: " + err.message);
        }
    }

    async _navigateTo(index) {
        if (this.crops.length === 0) return;
        if (index < 0) index = 0;
        if (index >= this.crops.length) index = this.crops.length - 1;
        await this._loadCrop(index);
    }

    _findNextPending(fromIndex) {
        for (let i = fromIndex; i < this.crops.length; i++) {
            if (this.crops[i].status === "pending") return i;
        }
        return null;
    }

    // -----------------------------------------------------------------------
    // UI updates
    // -----------------------------------------------------------------------
    _updateProgressUI() {
        const total = this.stats.total || this.crops.length;
        const approved = this.stats.approved || 0;
        const pending = this.stats.pending || 0;
        const rejected = this.stats.rejected || 0;
        const skipped = this.stats.skipped || 0;
        const pct = total > 0 ? ((approved + rejected + skipped) / total * 100).toFixed(0) : 0;

        const bar = this.panelEl.querySelector("#tjk-rev-progress-bar");
        const text = this.panelEl.querySelector("#tjk-rev-progress-text");
        const statsEl = this.panelEl.querySelector("#tjk-rev-stats");

        if (bar) bar.style.width = pct + "%";
        if (text) text.textContent = `${approved} / ${total} approved`;
        if (statsEl) {
            statsEl.textContent =
                `${pending} pending | ${approved} approved | ${rejected} rejected | ${skipped} skipped`;
        }
    }

    _updateStatusBadge(status) {
        const badge = this.panelEl.querySelector("#tjk-rev-status-badge");
        if (!badge) return;
        const colors = {
            pending:  { bg: "#313244", color: "#cdd6f4" },
            approved: { bg: "#a6e3a122", color: "#a6e3a1" },
            rejected: { bg: "#f38ba822", color: "#f38ba8" },
            skipped:  { bg: "#45475a",  color: "#a6adc8" },
        };
        const c = colors[status] || colors.pending;
        badge.style.background = c.bg;
        badge.style.color = c.color;
        badge.textContent = status;
    }

    _showMsg(msg, type) {
        const el = this.panelEl && this.panelEl.querySelector("#tjk-rev-msg");
        if (!el) return;
        if (msg) {
            el.textContent = (type === "success" ? "✅ " : type === "info" ? "ℹ️ " : "⚠ ") + msg;
            el.style.background = type === "success" ? "#a6e3a122"
                                 : type === "info"    ? "#89b4fa22"
                                 : "#f38ba822";
            el.style.color = type === "success" ? "#a6e3a1"
                            : type === "info"    ? "#89b4fa"
                            : "#f38ba8";
            el.style.display = "block";
        } else {
            el.style.display = "none";
        }
    }

    // -----------------------------------------------------------------------
    // Keyboard shortcuts
    // -----------------------------------------------------------------------
    _onKeyDown(e) {
        // Don't intercept when typing in the text input (except Enter/Tab which are handled there)
        const tag = e.target.tagName;
        const isTextInput = (tag === "INPUT" || tag === "TEXTAREA") &&
                            e.target.id !== "tjk-rev-label-select" &&
                            e.target.id !== "tjk-rev-output-dir";

        if (isTextInput && e.key !== "Escape") return;

        switch (e.key) {
            case "Enter":
                if (!isTextInput) {
                    e.preventDefault();
                    this._accept();
                }
                break;
            case "Tab":
                e.preventDefault();
                {
                    const inp = this.panelEl.querySelector("#tjk-rev-text-input");
                    if (inp) { inp.focus(); inp.select(); }
                }
                break;
            case "r":
            case "R":
                if (!isTextInput) {
                    e.preventDefault();
                    this._reject();
                }
                break;
            case "s":
            case "S":
                if (!isTextInput) {
                    e.preventDefault();
                    this._skip();
                }
                break;
            case "ArrowLeft":
                if (!isTextInput) {
                    e.preventDefault();
                    this._navigateTo(this.currentIndex - 1);
                }
                break;
            case "ArrowRight":
                if (!isTextInput) {
                    e.preventDefault();
                    this._navigateTo(this.currentIndex + 1);
                }
                break;
            case "Escape":
                this.hide();
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// ComfyUI Extension Registration
// ---------------------------------------------------------------------------
let _annotationPanel = null;
let _reviewPanel = null;

function getAnnotationPanel() {
    if (!_annotationPanel) _annotationPanel = new AnnotationPanel();
    return _annotationPanel;
}

function getReviewPanel() {
    if (!_reviewPanel) _reviewPanel = new ReviewPanel();
    return _reviewPanel;
}

app.registerExtension({
    name: "tjk_suetterlin.AnnotationPanel",

    async setup() {
        // Add menu buttons to ComfyUI's top menu bar
        try {
            const menuBar = document.querySelector(".comfyui-menu") ||
                             document.querySelector("#comfy-menu") ||
                             document.querySelector(".menu");
            if (menuBar) {
                const annotateBtn = _makeMenuButton("✏️ Annotate", () => getAnnotationPanel().toggle());
                const reviewBtn   = _makeMenuButton("📝 Review Transcriptions", () => getReviewPanel().toggle());
                menuBar.appendChild(annotateBtn);
                menuBar.appendChild(reviewBtn);
            } else {
                _addFloatingTriggers();
            }
        } catch (err) {
            console.warn("[tjk_suetterlin] Could not add menu buttons:", err);
            _addFloatingTriggers();
        }
    },
});

function _makeMenuButton(label, onClick) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.style.cssText = `
        background: #313244;
        color: #cdd6f4;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 4px 10px;
        cursor: pointer;
        font-size: 12px;
        margin: 0 4px;
    `;
    btn.addEventListener("click", onClick);
    return btn;
}

function _addFloatingTriggers() {
    // Annotation trigger
    const annotateBtn = document.createElement("button");
    annotateBtn.id = "tjk-annotation-trigger";
    annotateBtn.textContent = "✏️ Annotate";
    annotateBtn.title = "Open Sütterlin Word Annotation Panel";
    annotateBtn.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 9998;
        background: #313244;
        color: #cdd6f4;
        border: 1px solid #555;
        border-radius: 6px;
        padding: 6px 14px;
        cursor: pointer;
        font-size: 13px;
        font-family: 'Segoe UI', system-ui, sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    `;
    annotateBtn.addEventListener("click", () => getAnnotationPanel().toggle());
    document.body.appendChild(annotateBtn);

    // Review trigger
    const reviewBtn = document.createElement("button");
    reviewBtn.id = "tjk-review-trigger";
    reviewBtn.textContent = "📝 Review Transcriptions";
    reviewBtn.title = "Open Transcription Review Panel";
    reviewBtn.style.cssText = `
        position: fixed;
        top: 10px;
        right: 130px;
        z-index: 9998;
        background: #313244;
        color: #cdd6f4;
        border: 1px solid #555;
        border-radius: 6px;
        padding: 6px 14px;
        cursor: pointer;
        font-size: 13px;
        font-family: 'Segoe UI', system-ui, sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    `;
    reviewBtn.addEventListener("click", () => getReviewPanel().toggle());
    document.body.appendChild(reviewBtn);
}
