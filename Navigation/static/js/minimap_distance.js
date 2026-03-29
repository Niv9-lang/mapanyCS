class MinimapDistance {
    constructor() {
        this.canvas = document.getElementById('minimap-distance');
        this.ctx = this.canvas.getContext('2d');

        this.sceneInfo = null;
        this.minimapData = null;

        this.view = 'xz'; // 'xz' | 'xy' | 'yz'
        this.zoom = 1.0;
        this.pan = { x: 0, y: 0 };
        this.drag = { active: false, startX: 0, startY: 0, startPanX: 0, startPanY: 0 };

        this.pointA = null; // { x,y,z }
        this.pointB = null;

        this.init();
    }

    async init() {
        try {
            const [sceneResp, minimapResp] = await Promise.all([
                fetch('/api/scene-info'),
                fetch('/api/minimap-data'),
            ]);
            this.sceneInfo = await sceneResp.json();
            this.minimapData = await minimapResp.json();

            this.bindUI();
            this.draw();
        } catch (e) {
            console.error('Erreur init minimap distance:', e);
        }
    }

    bindUI() {
        document.getElementById('view-xz').addEventListener('click', () => this.setView('xz'));
        document.getElementById('view-xy').addEventListener('click', () => this.setView('xy'));
        document.getElementById('view-yz').addEventListener('click', () => this.setView('yz'));
        document.getElementById('btn-clear').addEventListener('click', () => this.clearPoints());
        this.setView('xz');

        // Zoom molette
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            this.zoom = Math.max(0.5, Math.min(6, this.zoom + delta));
            this.draw();
        }, { passive: false });

        // Pan par glisser
        this.canvas.addEventListener('mousedown', (e) => {
            if (e.button === 1 || e.button === 2) {
                this.drag = {
                    active: true,
                    startX: e.clientX,
                    startY: e.clientY,
                    startPanX: this.pan.x,
                    startPanY: this.pan.y,
                };
            }
        });
        window.addEventListener('mousemove', (e) => {
            if (this.drag.active) {
                this.pan.x = this.drag.startPanX + (e.clientX - this.drag.startX);
                this.pan.y = this.drag.startPanY + (e.clientY - this.drag.startY);
                this.draw();
            }
        });
        window.addEventListener('mouseup', () => { this.drag.active = false; });

        // Sélection de points (clic gauche)
        this.canvas.addEventListener('click', (e) => {
            if (this.drag.active) return;
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.handleClick(x, y);
        });
    }

    setView(view) {
        this.view = view;
        document.getElementById('dist-view').textContent = view.toUpperCase();
        ['view-xz', 'view-xy', 'view-yz'].forEach((id, i) => {
            const v = ['xz', 'xy', 'yz'][i];
            document.getElementById(id).classList.toggle('active', v === view);
        });
        this.draw();
    }

    clearPoints() {
        this.pointA = null;
        this.pointB = null;
        document.getElementById('dist-a').textContent = '–';
        document.getElementById('dist-b').textContent = '–';
        document.getElementById('dist-2d').textContent = '–';
        document.getElementById('dist-3d').textContent = '–';
        this.draw();
    }

    getViewData() {
        const verts = this.minimapData[`vertices_${this.view}`];
        const bounds = this.minimapData[`bounds_${this.view}`];
        if (!verts || !bounds) return null;
        const min = bounds.min;
        const max = bounds.max;
        return { verts, min, max };
    }

    worldToScreen(wx, wy, min, max) {
        const { width: W, height: H } = this.canvas;
        const rangeX = max[0] - min[0] || 1;
        const rangeY = max[1] - min[1] || 1;
        const sx = (wx - min[0]) / rangeX;
        const sy = 1 - (wy - min[1]) / rangeY;
        const px = sx * W;
        const py = sy * H;
        const cx = W / 2;
        const cy = H / 2;
        const zx = cx + (px - cx) * this.zoom;
        const zy = cy + (py - cy) * this.zoom;
        return [zx + this.pan.x, zy + this.pan.y];
    }

    draw() {
        if (!this.minimapData) return;
        const data = this.getViewData();
        if (!data) return;

        const ctx = this.ctx;
        const { width: W, height: H } = this.canvas;
        ctx.clearRect(0, 0, W, H);

        // Fond
        ctx.fillStyle = '#050711';
        ctx.fillRect(0, 0, W, H);

        // Grille
        ctx.strokeStyle = '#181b2a';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 10; i++) {
            const t = i / 10;
            const x = t * W;
            const y = t * H;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, H);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
        }

        // Points de la salle
        ctx.fillStyle = '#3d7cff';
        const { verts, min, max } = data;
        for (let i = 0; i < verts.length; i++) {
            const [wx, wy] = verts[i];
            const [sx, sy] = this.worldToScreen(wx, wy, min, max);
            ctx.fillRect(sx, sy, 1.5, 1.5);
        }

        // Points A et B
        const drawMarker = (p, color) => {
            if (!p) return;
            const [wx, wy] = this.projectToView(p);
            const [sx, sy] = this.worldToScreen(wx, wy, min, max);
            ctx.fillStyle = color;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(sx, sy, 7, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        };
        drawMarker(this.pointA, '#00ff88');
        drawMarker(this.pointB, '#ff3366');

        // Segment AB
        if (this.pointA && this.pointB) {
            const [ax, ay] = this.worldToScreen(...this.projectToView(this.pointA), min, max);
            const [bx, by] = this.worldToScreen(...this.projectToView(this.pointB), min, max);
            ctx.strokeStyle = '#ffcc00';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    handleClick(sx, sy) {
        const data = this.getViewData();
        if (!data) return;
        const { verts, min, max } = data;

        // Trouver le vertex le plus proche du clic en coordonnées écran
        let bestIdx = -1;
        let bestDist2 = Infinity;
        for (let i = 0; i < verts.length; i++) {
            const [wx, wy] = verts[i];
            const [px, py] = this.worldToScreen(wx, wy, min, max);
            const dx = px - sx;
            const dy = py - sy;
            const d2 = dx * dx + dy * dy;
            if (d2 < bestDist2) {
                bestDist2 = d2;
                bestIdx = i;
            }
        }

        // Seuil : ignorer si clic trop loin
        if (bestIdx === -1 || bestDist2 > 20 * 20) return;

        const [wx, wy] = verts[bestIdx];
        const worldPoint = this.unprojectFromView(wx, wy);

        if (!this.pointA) {
            this.pointA = worldPoint;
            document.getElementById('dist-a').textContent = this.formatPoint(worldPoint);
        } else if (!this.pointB) {
            this.pointB = worldPoint;
            document.getElementById('dist-b').textContent = this.formatPoint(worldPoint);
            this.updateDistances();
        } else {
            // Remplacer A et effacer B
            this.pointA = worldPoint;
            this.pointB = null;
            document.getElementById('dist-a').textContent = this.formatPoint(worldPoint);
            document.getElementById('dist-b').textContent = '–';
            document.getElementById('dist-2d').textContent = '–';
            document.getElementById('dist-3d').textContent = '–';
        }

        this.draw();
    }

    projectToView(p) {
        if (this.view === 'xz') return [p.x, p.z];
        if (this.view === 'xy') return [p.x, p.y];
        return [p.y, p.z]; // yz
    }

    unprojectFromView(wx, wy) {
        // On reconstruit un point 3D cohérent avec la vue courante.
        // Les coordonnées manquantes sont prises depuis le centre global.
        const c = this.sceneInfo.center || [0, 0, 0];
        if (this.view === 'xz') {
            return new THREE.Vector3(wx, c[1], wy);
        }
        if (this.view === 'xy') {
            return new THREE.Vector3(wx, wy, c[2]);
        }
        return new THREE.Vector3(c[0], wx, wy); // yz
    }

    updateDistances() {
        if (!this.pointA || !this.pointB) return;

        const pa = this.pointA;
        const pb = this.pointB;

        // Distance 3D
        const d3 = pa.distanceTo(pb);

        // Distance projetée dans le plan courant
        let d2;
        if (this.view === 'xz') {
            d2 = Math.hypot(pb.x - pa.x, pb.z - pa.z);
        } else if (this.view === 'xy') {
            d2 = Math.hypot(pb.x - pa.x, pb.y - pa.y);
        } else {
            d2 = Math.hypot(pb.y - pa.y, pb.z - pa.z);
        }

        document.getElementById('dist-2d').textContent = `${d2.toFixed(3)} m`;
        document.getElementById('dist-3d').textContent = `${d3.toFixed(3)} m`;
    }

    formatPoint(p) {
        return `[${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}]`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new MinimapDistance();
});

