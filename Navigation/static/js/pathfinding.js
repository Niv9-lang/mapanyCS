/**
 * pathfinding.js
 * Grille ternaire : 0=libre · 1=obstacle · 2=inconnu/hors scan
 */

class PathfindingUI {
    constructor() {
        this.canvas = document.getElementById('grid-canvas');
        this.ctx    = this.canvas.getContext('2d');

        this.gridData   = null;
        this.floorInfo  = null;
        this._yRange    = 1.0;

        this.pointA = null;
        this.pointB = null;
        this.mode   = 'none';
        this.path   = null;

        this._offscreen = null;

        this.vp = { zoom:1, panX:0, panY:0,
                    drag:{ active:false, sx:0, sy:0, px:0, py:0 } };

        this._resizeCanvas();
        this._bindSliders();
        this._bindButtons();
        this._bindCanvas();
        window.addEventListener('resize', () => { this._resizeCanvas(); this.draw(); });
        window.addEventListener('keydown', e => {
            if (e.key.toLowerCase() === 'r') { this._resetVP(); this.draw(); }
        });

        this._loadFloorInfo();
    }

    // ─────────────────────────────────────────────
    //  Infos sol
    // ─────────────────────────────────────────────

    async _loadFloorInfo() {
        try {
            const r = await fetch('/api/floor-info');
            this.floorInfo = await r.json();
            this._yRange   = this.floorInfo.y_range || 1.0;

            const fi = this.floorInfo;
            document.getElementById('floor-detected').innerHTML =
                `Sol Y : [<b>${fi.floor_y_min.toFixed(3)}</b>, <b>${fi.floor_y_max.toFixed(3)}</b>]<br>`
                + `PLY brut : Y∈[${fi.y_min.toFixed(3)}, ${fi.y_max.toFixed(3)}]`;

            // Pré-remplir les champs manuels avec la valeur détectée
            document.getElementById('inp-fy-min').value = fi.floor_y_min.toFixed(3);
            document.getElementById('inp-fy-max').value = fi.floor_y_max.toFixed(3);

            // Avertissement si le sol détecté semble bizarre
            if (fi.ceil_y_center !== null) {
                document.getElementById('floor-warn').textContent =
                    `Plafond détecté à Y=${fi.ceil_y_center.toFixed(3)}`;
            }

            this._updateSliderLabels();
            this._setStatus('idle', 'Sol détecté. Ajuster si incorrect, puis générer la carte.');
        } catch (e) {
            document.getElementById('floor-detected').textContent = 'Erreur chargement';
        }
    }

    // ─────────────────────────────────────────────
    //  Génération de la grille
    // ─────────────────────────────────────────────

    async _generateGrid() {
        const p  = this._getParams();
        const qs = new URLSearchParams(p).toString();

        this._setStatus('loading', '<span class="spinner"></span>Calcul de la grille…');
        document.getElementById('btn-gen').disabled = true;

        this.pointA = null; this.pointB = null;
        this.path   = null; this.mode   = 'A';
        this._updatePtLabels(); this._resetResults();

        try {
            const resp = await fetch('/api/occupancy-grid?' + qs);
            this.gridData = await resp.json();

            if (this.gridData.error) {
                this._setStatus('err', '✗ ' + this.gridData.error);
                return;
            }

            document.getElementById('wait-overlay').classList.add('hidden');
            this._prerenderGrid();
            this._resetVP();
            this._clearStale();
            this.draw();

            const pct = (100 * this.gridData.obstacle_cells / this.gridData.total_cells).toFixed(1);
            document.getElementById('res-pct').textContent  = pct + '%';
            document.getElementById('res-free').textContent = this.gridData.free_cells.toLocaleString();

            this._setBadge('A');
            document.getElementById('btn-path').disabled = true;

            const dir = this.gridData.obstacles_above ? 'au-dessus' : 'en-dessous';
            this._setStatus('ok',
                `✓ Carte générée — sol Y[${this.gridData.floor_y_min.toFixed(3)}, `
                + `${this.gridData.floor_y_max.toFixed(3)}] — obstacles ${dir} — `
                + `${pct}% obstacles — ${this.gridData.free_cells.toLocaleString()} libres`);

        } catch (e) {
            this._setStatus('err', '✗ ' + e.message);
        } finally {
            document.getElementById('btn-gen').disabled = false;
        }
    }

    // ─────────────────────────────────────────────
    //  Calcul de trajectoire
    // ─────────────────────────────────────────────

    async _computePath() {
        if (!this.pointA || !this.pointB) return;
        const qs = new URLSearchParams({
            ...this._getParams(),
            ax: this.pointA.x, az: this.pointA.z,
            bx: this.pointB.x, bz: this.pointB.z,
        }).toString();

        this._setStatus('loading', '<span class="spinner"></span>Calcul A*…');
        document.getElementById('btn-path').disabled = true;

        try {
            const resp   = await fetch('/api/pathfind?' + qs);
            const result = await resp.json();

            if (result.error) {
                this._setStatus('err', '✗ ' + result.error);
                this.path = null;
                // Même en cas d'échec, mettre à jour la grille si le serveur en retourne une
                // (paramètres changés depuis la dernière génération)
                if (result.grid_data) {
                    this.gridData = result.grid_data;
                    this._prerenderGrid();
                    this._clearStale();
                }
            } else {
                this.path = result.path;
                if (result.grid_data) {
                    this.gridData = result.grid_data;
                    this._prerenderGrid();
                    this._clearStale();
                }
                const scale   = parseFloat(document.getElementById('sl-scale').value) || 100;
                const distCm  = result.length * scale;
                const distStr = distCm >= 100
                    ? `${(distCm / 100).toFixed(2)} m  (${Math.round(distCm)} cm)`
                    : `${Math.round(distCm)} cm`;
                document.getElementById('res-len').textContent  = Math.round(distCm);
                document.getElementById('res-unit').textContent = 'cm';
                document.getElementById('res-wp').textContent   = result.path_smooth_count;
                this._setStatus('ok',
                    `✓ Chemin trouvé — ${distStr} — `
                    + `${result.path_smooth_count} waypoints — A* : ${result.iterations} itérations`);
            }
        } catch (e) {
            this._setStatus('err', '✗ ' + e.message);
            this.path = null;
        } finally {
            document.getElementById('btn-path').disabled = false;
            this.draw();
        }
    }

    // ─────────────────────────────────────────────
    //  Pré-rendu de la grille
    // ─────────────────────────────────────────────

    _prerenderGrid() {
        const gs   = this.gridData.grid_size;
        const grid = this.gridData.grid;

        this._offscreen        = document.createElement('canvas');
        this._offscreen.width  = gs;
        this._offscreen.height = gs;
        const gctx = this._offscreen.getContext('2d');
        const img  = gctx.createImageData(gs, gs);
        const d    = img.data;

        for (let row = 0; row < gs; row++) {
            for (let col = 0; col < gs; col++) {
                const idx = (row * gs + col) * 4;
                const val = grid[row][col];
                if (val === 2) {
                    // Inconnu / hors scan → noir quasi pur
                    d[idx]=7; d[idx+1]=10; d[idx+2]=16; d[idx+3]=255;
                } else if (val === 0) {
                    // Sol libre → bleu très foncé
                    d[idx]=10; d[idx+1]=26; d[idx+2]=48; d[idx+3]=255;
                } else {
                    // Obstacle + marge robot → orange-rouge bien visible
                    d[idx]=160; d[idx+1]=45; d[idx+2]=20; d[idx+3]=255;
                }
            }
        }
        gctx.putImageData(img, 0, 0);
    }

    // ─────────────────────────────────────────────
    //  Rendu
    // ─────────────────────────────────────────────

    draw() {
        const W = this.canvas.width, H = this.canvas.height;
        const ctx = this.ctx;
        ctx.fillStyle = '#07090f'; ctx.fillRect(0,0,W,H);
        if (!this._offscreen) return;

        ctx.save();
        ctx.translate(W/2+this.vp.panX, H/2+this.vp.panY);
        ctx.scale(this.vp.zoom, this.vp.zoom);
        const size = Math.min(W,H)*0.92, half = size/2;

        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(this._offscreen, -half, -half, size, size);

        if (this.path && this.path.length >= 2) {
            const pts = this.path.map(p => this._w2c(p[0],p[1],half,size));
            ctx.strokeStyle='rgba(255,193,7,0.2)'; ctx.lineWidth=9/this.vp.zoom;
            ctx.lineJoin='round'; ctx.lineCap='round';
            ctx.beginPath(); pts.forEach((p,i)=>i===0?ctx.moveTo(p[0],p[1]):ctx.lineTo(p[0],p[1])); ctx.stroke();
            ctx.strokeStyle='#ffc107'; ctx.lineWidth=2.5/this.vp.zoom;
            ctx.beginPath(); pts.forEach((p,i)=>i===0?ctx.moveTo(p[0],p[1]):ctx.lineTo(p[0],p[1])); ctx.stroke();
            this._drawArrows(ctx, pts);
        }

        if (this.pointA) { const [px,py]=this._w2c(this.pointA.x,this.pointA.z,half,size); this._drawMarker(ctx,px,py,'#a3e635','A'); }
        if (this.pointB) { const [px,py]=this._w2c(this.pointB.x,this.pointB.z,half,size); this._drawMarker(ctx,px,py,'#ff3d6e','B'); }

        ctx.restore();
    }

    _drawMarker(ctx,px,py,color,label) {
        const r=7/this.vp.zoom;
        ctx.fillStyle=color+'33'; ctx.beginPath(); ctx.arc(px,py,r*2.2,0,Math.PI*2); ctx.fill();
        ctx.fillStyle=color; ctx.strokeStyle='#fff'; ctx.lineWidth=1.5/this.vp.zoom;
        ctx.beginPath(); ctx.arc(px,py,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
        ctx.fillStyle='#07090f'; ctx.font=`bold ${Math.max(8,8/this.vp.zoom)}px 'Space Mono',monospace`;
        ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(label,px,py);
    }

    _drawArrows(ctx,pts) {
        if (pts.length<2) return;
        const step=Math.max(1,Math.floor(pts.length/5));
        ctx.fillStyle='#ffc107';
        for (let i=step;i<pts.length-1;i+=step) {
            const dx=pts[i][0]-pts[i-1][0],dy=pts[i][1]-pts[i-1][1];
            const angle=Math.atan2(dy,dx),sz=5/this.vp.zoom;
            ctx.save(); ctx.translate(pts[i][0],pts[i][1]); ctx.rotate(angle);
            ctx.beginPath(); ctx.moveTo(sz,0); ctx.lineTo(-sz,-sz*.6); ctx.lineTo(-sz,sz*.6);
            ctx.closePath(); ctx.fill(); ctx.restore();
        }
    }

    // ─────────────────────────────────────────────
    //  Conversions
    // ─────────────────────────────────────────────

    _w2c(wx,wz,half,size) {
        const b=this.gridData.bounds;
        const sx=(wx-b.xmin)/(b.xmax-b.xmin||1);
        // L'image est rendue avec row=0 en haut (z=zmin) et row=gs-1 en bas (z=zmax).
        // On ne doit PAS inverser l'axe Z ici pour être cohérent avec le rendu.
        const sz=(wz-b.zmin)/(b.zmax-b.zmin||1);
        return [-half+sx*size,-half+sz*size];
    }

    _c2w(cx,cy) {
        if (!this.gridData) return null;
        const W=this.canvas.width,H=this.canvas.height;
        const size=Math.min(W,H)*0.92,half=size/2;
        const lx=(cx-W/2-this.vp.panX)/this.vp.zoom;
        const ly=(cy-H/2-this.vp.panY)/this.vp.zoom;
        const b=this.gridData.bounds;
        return {
            x: b.xmin+((lx+half)/size)*(b.xmax-b.xmin),
            z: b.zmin+((ly+half)/size)*(b.zmax-b.zmin),
        };
    }

    // ─────────────────────────────────────────────
    //  Événements canvas
    // ─────────────────────────────────────────────

    _bindCanvas() {
        const c=this.canvas;
        c.addEventListener('click', (e) => {
            if (this.mode==='none'||!this.gridData) return;
            const rect=c.getBoundingClientRect();
            const cx=(e.clientX-rect.left)*(c.width/rect.width);
            const cy=(e.clientY-rect.top)*(c.height/rect.height);
            const wp=this._c2w(cx,cy); if (!wp) return;
            if (this.mode==='A') { this.pointA=wp; this.mode='B'; }
            else if (this.mode==='B') { this.pointB=wp; this.mode='done'; }
            else { this.pointA=wp; this.pointB=null; this.path=null; this.mode='B'; this._resetResults(); }
            this._setBadge(this.mode); this._updatePtLabels();
            document.getElementById('btn-path').disabled=!(this.pointA&&this.pointB);
            this.draw();
        });
        c.addEventListener('mousedown',(e)=>{
            if (e.button===2||e.button===1) { e.preventDefault(); this.vp.drag={active:true,sx:e.clientX,sy:e.clientY,px:this.vp.panX,py:this.vp.panY}; }
        });
        c.addEventListener('contextmenu',e=>e.preventDefault());
        window.addEventListener('mousemove',(e)=>{
            if (this.vp.drag.active) { this.vp.panX=this.vp.drag.px+(e.clientX-this.vp.drag.sx); this.vp.panY=this.vp.drag.py+(e.clientY-this.vp.drag.sy); this.draw(); }
            if (this.gridData) {
                const rect=c.getBoundingClientRect();
                if (e.clientX>=rect.left&&e.clientX<=rect.right&&e.clientY>=rect.top&&e.clientY<=rect.bottom) {
                    const wp=this._c2w((e.clientX-rect.left)*(c.width/rect.width),(e.clientY-rect.top)*(c.height/rect.height));
                    if (wp) document.getElementById('cursor-coords').textContent=`X: ${wp.x.toFixed(3)}   Z: ${wp.z.toFixed(3)}`;
                }
            }
        });
        window.addEventListener('mouseup',()=>{ this.vp.drag.active=false; });
        c.addEventListener('wheel',(e)=>{ e.preventDefault(); this.vp.zoom=Math.max(0.3,Math.min(14,this.vp.zoom+(e.deltaY>0?-0.1:0.1))); this.draw(); },{passive:false});
    }

    // ─────────────────────────────────────────────
    //  Boutons et sliders
    // ─────────────────────────────────────────────

    _bindButtons() {
        document.getElementById('btn-gen').addEventListener('click',()=>this._generateGrid());
        document.getElementById('btn-path').addEventListener('click',()=>this._computePath());
        document.getElementById('btn-clear').addEventListener('click',()=>{
            this.pointA=null; this.pointB=null; this.path=null;
            this.mode=this.gridData?'A':'none';
            this._setBadge(this.mode); this._updatePtLabels(); this._resetResults();
            document.getElementById('btn-path').disabled=true;
            this._setStatus('idle','Points effacés.'); this.draw();
        });
        document.getElementById('btn-reset-floor').addEventListener('click',()=>{
            if (this.floorInfo) {
                document.getElementById('inp-fy-min').value=this.floorInfo.floor_y_min.toFixed(3);
                document.getElementById('inp-fy-max').value=this.floorInfo.floor_y_max.toFixed(3);
            }
        });
    }

    _bindSliders() {
        // Sliders qui affectent la grille → marquer comme périmée si la carte est déjà générée
        ['sl-gs','sl-minh','sl-maxh','sl-rad'].forEach(id=>
            document.getElementById(id).addEventListener('input',()=>{
                this._updateSliderLabels();
                if (this.gridData) this._markStale();
            }));
        ['inp-fy-min','inp-fy-max'].forEach(id=>
            document.getElementById(id).addEventListener('input',()=>{
                if (this.gridData) this._markStale();
            }));
        // Slider d'échelle : affichage seulement, pas besoin de regénérer
        document.getElementById('sl-scale').addEventListener('input',()=>this._updateSliderLabels());
    }

    _markStale() {
        document.getElementById('btn-gen').classList.add('stale');
        this._setStatus('idle','⚠ Paramètres modifiés — regénérer la carte avant de calculer.');
    }

    _clearStale() {
        document.getElementById('btn-gen').classList.remove('stale');
    }

    _updateSliderLabels() {
        const yr=this._yRange||1.0;
        document.getElementById('lbl-gs').textContent    = document.getElementById('sl-gs').value;
        document.getElementById('lbl-rad').textContent   = document.getElementById('sl-rad').value;
        document.getElementById('lbl-scale').textContent = document.getElementById('sl-scale').value;
        const minPct=parseInt(document.getElementById('sl-minh').value);
        const maxPct=parseInt(document.getElementById('sl-maxh').value);
        document.getElementById('lbl-minh').textContent=(yr*minPct/100).toFixed(3);
        document.getElementById('lbl-maxh').textContent=(yr*maxPct/100).toFixed(3);
    }

    _getParams() {
        const yr    =this._yRange||1.0;
        const minPct=parseInt(document.getElementById('sl-minh').value);
        const maxPct=parseInt(document.getElementById('sl-maxh').value);
        const fyMin =document.getElementById('inp-fy-min').value.trim();
        const fyMax =document.getElementById('inp-fy-max').value.trim();
        const p={
            grid_size:    parseInt(document.getElementById('sl-gs').value),
            min_h:        parseFloat((yr*minPct/100).toFixed(4)),
            max_h:        parseFloat((yr*maxPct/100).toFixed(4)),
            robot_radius: parseInt(document.getElementById('sl-rad').value),
        };
        if (fyMin!==''&&fyMax!=='') { p.floor_y_min=parseFloat(fyMin); p.floor_y_max=parseFloat(fyMax); }
        return p;
    }

    // ─────────────────────────────────────────────
    //  Helpers UI
    // ─────────────────────────────────────────────

    _setBadge(mode) {
        const el=document.getElementById('mode-badge'); el.className='';
        if (mode==='A')      { el.className='a';    el.textContent='● PLACER A'; }
        else if (mode==='B') { el.className='b';    el.textContent='● PLACER B'; }
        else if (mode==='done') { el.className='done'; el.textContent='POINTS PLACÉS'; }
    }

    _updatePtLabels() {
        const fmt=p=>p?`[${p.x.toFixed(3)}, ${p.z.toFixed(3)}]`:null;
        const elA=document.getElementById('pt-a'),elB=document.getElementById('pt-b');
        if (this.pointA){elA.textContent=fmt(this.pointA);elA.classList.remove('empty');}
        else            {elA.textContent='—';elA.classList.add('empty');}
        if (this.pointB){elB.textContent=fmt(this.pointB);elB.classList.remove('empty');}
        else            {elB.textContent='—';elB.classList.add('empty');}
    }

    _resetResults() {
        ['res-len','res-wp'].forEach(id=>document.getElementById(id).textContent='–');
        document.getElementById('res-unit').textContent='–';
    }

    _setStatus(type,html) {
        const el=document.getElementById('info-bar'); el.className=type; el.innerHTML=html;
    }

    _resizeCanvas() {
        const p=document.getElementById('canvas-panel');
        this.canvas.width=p.clientWidth; this.canvas.height=p.clientHeight;
    }

    _resetVP() { this.vp.zoom=1; this.vp.panX=0; this.vp.panY=0; }
}

document.addEventListener('DOMContentLoaded',()=>{ new PathfindingUI(); });
