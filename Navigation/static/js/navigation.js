/**
 * Navigation 3D - Logique principale
 * Gère la navigation, l'affichage et l'interaction avec la scène 3D
 */

class SceneNavigator {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.model = null;

        
        this.sceneInfo = null;
        this.cameraConfig = null;
        this.minimapData = null;
        
        this.stats = {
            fps: 0,
            frameCount: 0,
            lastTime: Date.now(),
        };
        
        this.minimapVisible = true;
        this.lastZoom = 1;
        
        // Minimap type Google Maps : pan, zoom, vues
        this.minimapView = 'xz';  // 'xz' | 'xy' | 'yz'
        this.minimapPan = { x: 0, y: 0 };
        this.minimapZoom = 1;
        this.minimapDrag = { active: false, startX: 0, startY: 0, startPanX: 0, startPanY: 0 };
        
        this.init();
    }

    async init() {
        console.log("🚀 Initialisation de la navigation 3D...");
        
        try {
            // Charger les données de configuration
            await this.loadConfiguration();
            
            // Initialiser Three.js
            this.initThreeJS();
            
            // Charger le modèle GLB
            await this.loadModel();
            
            // Initialiser les contrôles
            this.initControls();
            
            // Initialiser l'interface
            this.initUI();
            
            // Démarrer la boucle de rendu
            this.animate();
            
            console.log("✓ Navigation prête!");
        } catch (error) {
            console.error("✗ Erreur lors de l'initialisation:", error);
            this.showError(error.message);
        }
    }

    async loadConfiguration() {
        console.log("📋 Chargement de la configuration...");
        
        try {
            const [sceneResp, cameraResp, minimapResp] = await Promise.all([
                fetch('/api/scene-info'),
                fetch('/api/camera-config'),
                fetch('/api/minimap-data'),
            ]);
            
            this.sceneInfo = await sceneResp.json();
            this.cameraConfig = await cameraResp.json();
            this.minimapData = await minimapResp.json();
            
            console.log("✓ Configuration chargée", {
                scene: this.sceneInfo,
                camera: this.cameraConfig,
            });
        } catch (error) {
            throw new Error(`Erreur chargement configuration: ${error.message}`);
        }
    }

    initThreeJS() {
        console.log("🎨 Initialisation Three.js...");
        
        // Scène
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        this.scene.fog = new THREE.Fog(0x1a1a1a, this.sceneInfo.radius * 100, this.sceneInfo.radius * 200);

        // Caméra
        const config = this.cameraConfig;
        this.camera = new THREE.PerspectiveCamera(
            config.fov,
            window.innerWidth / window.innerHeight,
            config.near,
            config.far
        );
        this.camera.position.fromArray(config.position);
        this.camera.lookAt(...config.target);

        // Renderer
        const container = document.getElementById('canvas-container');
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFShadowMap;
        this.renderer.outputEncoding = THREE.LinearEncoding;
        this.renderer.toneMapping = THREE.NoToneMapping;
        container.appendChild(this.renderer.domElement);

        // Lumières
        this.setupLighting();

        // Événements de redimensionnement
        window.addEventListener('resize', () => this.onWindowResize());
    }

    setupLighting() {
        console.log("💡 Configuration des lumières...");
        
        // HemisphereLight simule ciel/sol (comme une IBL simplifiée)
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.2);
        hemiLight.position.set(0, 20, 0);
        this.scene.add(hemiLight);

        // Lumière ambiante pour remplir les ombres
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        this.scene.add(ambientLight);

        // Lumière directionnelle principale (soleil)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
        directionalLight.position.set(5, 10, 7);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.1;
        directionalLight.shadow.camera.far = this.sceneInfo.radius * 100;
        
        const shadowSize = this.sceneInfo.radius * 2;
        directionalLight.shadow.camera.left = -shadowSize;
        directionalLight.shadow.camera.right = shadowSize;
        directionalLight.shadow.camera.top = shadowSize;
        directionalLight.shadow.camera.bottom = -shadowSize;
        
        this.scene.add(directionalLight);

        // Lumière de remplissage (face opposée)
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);

        // Lumière point pour relief
        const pointLight = new THREE.PointLight(0xffffff, 0.5);
        pointLight.position.set(0, 10, 0);
        this.scene.add(pointLight);
    }

    async loadModel() {
        const format = this.sceneInfo.model_format || 'glb';
        const url = this.sceneInfo.model_url || '/model';
        console.log(`📦 Chargement du modèle ${format.toUpperCase()}...`);
        
        const handleError = (err) => {
            const msg = err?.message || err?.target?.statusText || 
                (err?.target?.status ? `HTTP ${err.target.status}` : 'Erreur inconnue');
            throw new Error(msg);
        };

        try {
        if (format === 'ply') {
            const PLYLoaderClass = THREE.PLYLoader || window.PLYLoader;
            const geometry = await new Promise((resolve, reject) => {
                const loader = new PLYLoaderClass();
                loader.load(url + '?' + Date.now(), resolve, undefined, reject);
            });

            geometry.computeVertexNormals();
            const hasColors = geometry.attributes.color !== undefined;
            // Taille des points : revenir au comportement initial
            const pointSize = this.sceneInfo.radius * 0.005;

            const material = new THREE.PointsMaterial({
                size: pointSize,
                vertexColors: hasColors,
                color: hasColors ? 0xffffff : 0x6699ff,
                sizeAttenuation: true,
            });

            this.model = new THREE.Points(geometry, material);
            this._plyGeometry = geometry;          // ← stocker pour extraction minimap
            this.scene.add(this.model);
        }
            
            this.centerModel();
            console.log("✓ Modèle chargé");

            if (this._plyGeometry) {
            this.extractMinimapFromGeometry(this._plyGeometry);
            }
        } catch (error) {
            const msg = error?.message || error?.target?.statusText || 
                (typeof error === 'string' ? error : 'Échec du chargement (vérifiez la console)');
            console.error('Erreur chargement:', error);
            throw new Error(`Erreur chargement modèle: ${msg}`);
        }
    }

    centerModel() {
        if (!this.model) return;

        // 1. Appliquer la rotation EN PREMIER
        this.model.rotation.x = Math.PI;

        // 2. Puis calculer le centre APRÈS rotation et recentrer
        const box = new THREE.Box3().setFromObject(this.model);
        const center = box.getCenter(new THREE.Vector3());
        this.model.position.sub(center);
    }
    extractMinimapFromGeometry(geometry) {
        // Forcer la mise à jour de la matrice monde (après rotation + centrage)
        this.model.updateMatrixWorld(true);
        const matrix = this.model.matrixWorld;

        const positions = geometry.attributes.position;
        const total = positions.count;

        // Sous-échantillonnage : max 8000 points pour la minimap
        const step = Math.max(1, Math.floor(total / 8000));

        const verts = [];
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        const v = new THREE.Vector3();
        for (let i = 0; i < total; i += step) {
            v.fromBufferAttribute(positions, i);
            v.applyMatrix4(matrix);   // coordonnées monde réelles

            if (minX > v.x) minX = v.x;
            if (maxX < v.x) maxX = v.x;
            if (minY > v.y) minY = v.y;
            if (maxY < v.y) maxY = v.y;
            if (minZ > v.z) minZ = v.z;
            if (maxZ < v.z) maxZ = v.z;

            verts.push([v.x, v.y, v.z]);
        }

        // Construire les projections 2D dans les trois plans
        this.minimapData = {
            vertices_xz: verts.map(p => [p[0], p[2]]),
            vertices_xy: verts.map(p => [p[0], p[1]]),
            vertices_yz: verts.map(p => [p[1], p[2]]),
            bounds_xz: { min: [minX, minZ], max: [maxX, maxZ] },
            bounds_xy: { min: [minX, minY], max: [maxX, maxY] },
            bounds_yz: { min: [minY, minZ], max: [maxY, maxZ] },
        };

        // Mettre à jour les infos de scène avec les vraies valeurs
        this.sceneInfo.center = [
            (minX + maxX) / 2,
            (minY + maxY) / 2,
            (minZ + maxZ) / 2,
        ];
        this.sceneInfo.radius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2;

        console.log(`✓ Minimap extraite du PLY : ${verts.length} points, rayon réel = ${this.sceneInfo.radius.toFixed(3)}`);
    }

    initControls() {
        const TrackballControlsClass = THREE.TrackballControls || window.TrackballControls;
        if (!TrackballControlsClass) {
            throw new Error("TrackballControls non chargé — vérifiez le script dans index.html");
        }

        this.controls = new TrackballControlsClass(this.camera, this.renderer.domElement);

        this.controls.rotateSpeed = 2.0;
        this.controls.panSpeed = 0.8;
        this.controls.noZoom = true;
        this.controls.noPan = false;
        this.controls.staticMoving = false;
        this.controls.dynamicDampingFactor = 0.1;

        const zoomStep = this.sceneInfo.radius * 0.06;
        const minDistance = this.sceneInfo.radius * 0.05; // distance minimale à la cible

        this.renderer.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();

            const toTarget = new THREE.Vector3()
                .subVectors(this.controls.target, this.camera.position);
            const distance = toTarget.length();
            const direction = toTarget.normalize();

            const delta = e.deltaY > 0 ? -zoomStep : zoomStep;

            // Zoom avant : bloquer avant d'atteindre la cible
            if (delta > 0 && distance - delta < minDistance) return;

            // Zoom arrière : pas de limite
            this.camera.position.addScaledVector(direction, delta);
        }, { passive: false });
            }


    initUI() {
        console.log("🎨 Initialisation de l'interface...");
        
        // Afficher le nom du modèle
        document.getElementById('model-name').textContent = `Fichier: ${this.sceneInfo.filename}`;
        
        // Afficher les infos 3D
        document.getElementById('info-content').innerHTML = `
            <p><strong>Vertices:</strong> ${this.formatNumber(this.sceneInfo.radius * 10000)}</p>
            <p><strong>Centre:</strong> [${this.sceneInfo.center.map(x => x.toFixed(2)).join(', ')}]</p>
            <p><strong>Rayon:</strong> ${this.sceneInfo.radius.toFixed(2)}</p>
        `;
        
        // Boutons
        document.getElementById('btn-reset').addEventListener('click', () => this.resetView());
        document.getElementById('btn-screenshot').addEventListener('click', () => this.captureScreenshot());
        document.getElementById('btn-fullscreen').addEventListener('click', () => this.toggleFullscreen());
        document.getElementById('btn-help').addEventListener('click', () => this.showHelp());
        
        // Minimap toolbar
        document.getElementById('minimap-view-xz').addEventListener('click', () => this.setMinimapView('xz'));
        document.getElementById('minimap-view-xy').addEventListener('click', () => this.setMinimapView('xy'));
        document.getElementById('minimap-view-yz').addEventListener('click', () => this.setMinimapView('yz'));
        document.getElementById('minimap-reset').addEventListener('click', () => this.resetMinimapView());
        this.setMinimapView('xz');  // Vue par défaut
        
        // Minimap pan/zoom (style Google Maps)
        this.initMinimapControls();
        
        // Clavier
        window.addEventListener('keydown', (e) => this.handleKeyPress(e));
        
        // Afficher la minimap
        this.updateMinimap();
    }

    setMinimapView(mode) {
        this.minimapView = mode;
        ['minimap-view-xz', 'minimap-view-xy', 'minimap-view-yz'].forEach((id, i) => {
            const btn = document.getElementById(id);
            btn.classList.toggle('active', (['xz', 'xy', 'yz'][i] === mode));
        });
    }

    resetMinimapView() {
        this.minimapPan = { x: 0, y: 0 };
        this.minimapZoom = 1;
    }

    initMinimapControls() {
        const wrapper = document.getElementById('minimap-wrapper');
        const canvas = document.getElementById('minimap');
        
        // Zoom molette
        wrapper.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            this.minimapZoom = Math.max(0.5, Math.min(5, this.minimapZoom + delta));
        }, { passive: false });
        
        // Pan glisser
        wrapper.addEventListener('mousedown', (e) => {
            if (e.button === 0) {
                this.minimapDrag = {
                    active: true,
                    startX: e.clientX,
                    startY: e.clientY,
                    startPanX: this.minimapPan.x,
                    startPanY: this.minimapPan.y
                };
            }
        });
        window.addEventListener('mousemove', (e) => {
            if (this.minimapDrag.active) {
                this.minimapPan.x = this.minimapDrag.startPanX + (e.clientX - this.minimapDrag.startX);
                this.minimapPan.y = this.minimapDrag.startPanY + (e.clientY - this.minimapDrag.startY);
            }
        });
        window.addEventListener('mouseup', () => { this.minimapDrag.active = false; });
        // Coordonnées au survol
        canvas.addEventListener('mousemove', (e) => {
            if (!this._minimapBounds) return;

            const rect = canvas.getBoundingClientRect();
            const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
            const sy = (e.clientY - rect.top) * (canvas.height / rect.height);

            const W = canvas.width, H = canvas.height;
            const cx = W / 2, cy = H / 2;
            const { bounds, rangeX, rangeY } = this._minimapBounds;

            // Inverse exacte de worldToScreen
            const unpannedX = sx - this.minimapPan.x;
            const unpannedY = sy - this.minimapPan.y;
            const px = cx + (unpannedX - cx) / this.minimapZoom;
            const py = cy + (unpannedY - cy) / this.minimapZoom;
            const wx = (px / W) * rangeX + bounds.min[0];
            const wy = (1 - py / H) * rangeY + bounds.min[1];

            // Reconstituer les 3 axes selon la vue active
            const camPos = this.camera.position;
            let x, y, z;
            if (this.minimapView === 'xz')      { x = wx; y = camPos.y; z = wy; }
            else if (this.minimapView === 'xy') { x = wx; y = wy;       z = camPos.z; }
            else                                { x = camPos.x; y = wx; z = wy; }

            document.getElementById('minimap-coords').textContent =
                `X: ${x.toFixed(3)}   Y: ${y.toFixed(3)}   Z: ${z.toFixed(3)}`;
        });

        canvas.addEventListener('mouseleave', () => {
            document.getElementById('minimap-coords').textContent = '—';
        });
    }

    handleKeyPress(event) {
        switch(event.key.toLowerCase()) {
            case 'r':
                event.preventDefault();
                this.resetView();
                break;
            case 'm':
                event.preventDefault();
                this.toggleMinimap();
                break;
            case 'p':
                event.preventDefault();
                this.captureScreenshot();
                break;
            case 'f':
                event.preventDefault();
                this.toggleFullscreen();
                break;
            case 'h':
                event.preventDefault();
                this.showHelp();
                break;
        }
    }

    resetView() {
        console.log("↻ Réinitialisation de la vue...");
        const config = this.cameraConfig;
        
        this.camera.position.fromArray(config.position);
        this.camera.lookAt(...config.target);
        
        const target = new THREE.Vector3(...config.target);
        this.controls.target.copy(target);
        this.controls.update();
    }

    updateMinimap() {
        if (!this.minimapData) return;
        
        const canvas = document.getElementById('minimap');
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        
        // Choix vue et données
        const viewKey = this.minimapView;
        const bounds = this.minimapData[`bounds_${viewKey}`];
        const vertices = this.minimapData[`vertices_${viewKey}`] || [];
        
        // Position caméra en 2D selon la vue (axes: xz -> (x,z), xy -> (x,y), yz -> (y,z))
        const camPos = this.camera.position;
        let cam2d;
        if (viewKey === 'xz') cam2d = [camPos.x, camPos.z];
        else if (viewKey === 'xy') cam2d = [camPos.x, camPos.y];
        else cam2d = [camPos.y, camPos.z];
        
        if (!bounds || bounds.min[0] === bounds.max[0] || bounds.min[1] === bounds.max[1]) return;
        
        const rangeX = bounds.max[0] - bounds.min[0];
        const rangeY = bounds.max[1] - bounds.min[1];
        this._minimapBounds = { bounds, rangeX, rangeY };
        const cx = W / 2, cy = H / 2;
        
        // Conversion monde -> écran avec pan/zoom (style Google Maps)
        const worldToScreen = (wx, wy) => {
            const sx = (wx - bounds.min[0]) / rangeX;           // 0-1
            const sy = 1 - (wy - bounds.min[1]) / rangeY;       // 0-1 (Y inversé)
            const px = sx * W;
            const py = sy * H;
            const zoomedX = cx + (px - cx) * this.minimapZoom;
            const zoomedY = cy + (py - cy) * this.minimapZoom;
            return [zoomedX + this.minimapPan.x, zoomedY + this.minimapPan.y];
        };
        
        // Fond
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, W, H);
        
        // Grille
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const t = i / 5;
            const [gx1] = worldToScreen(bounds.min[0] + t * rangeX, bounds.min[1]);
            const [, gy1] = worldToScreen(bounds.min[0], bounds.min[1] + t * rangeY);
            const [gx2] = worldToScreen(bounds.min[0] + t * rangeX, bounds.max[1]);
            const [, gy2] = worldToScreen(bounds.max[0], bounds.min[1] + t * rangeY);
            ctx.beginPath();
            ctx.moveTo(gx1, 0);
            ctx.lineTo(gx1, H);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, gy1);
            ctx.lineTo(W, gy1);
            ctx.stroke();
        }
        
        // Plan au sol (vertices - forme de la salle)
        if (vertices.length > 0) {
            ctx.fillStyle = '#0066ff';
            for (let i = 0; i < vertices.length; i++) {
                const [px, py] = worldToScreen(vertices[i][0], vertices[i][1]);
                ctx.fillRect(px, py, 1.5, 1.5);
            }
        }
        
        // Point rouge = position de l'utilisateur
        const [camSx, camSy] = worldToScreen(cam2d[0], cam2d[1]);
        ctx.fillStyle = '#ff3333';
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(camSx, camSy, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }

    toggleMinimap() {
        this.minimapVisible = !this.minimapVisible;
        const minimap = document.getElementById('minimap-container');
        minimap.style.display = this.minimapVisible ? 'block' : 'none';
        console.log(this.minimapVisible ? "✓ Minimap affichée" : "✗ Minimap masquée");
    }

    captureScreenshot() {
        console.log("📷 Capture d'écran...");
        this.renderer.render(this.scene, this.camera);
        this.renderer.domElement.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `screenshot_${Date.now()}.png`;
            a.click();
            console.log("✓ Screenshot sauvegardé");
        });
    }

    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => console.error(err));
        } else {
            document.exitFullscreen();
        }
    }

    showHelp() {
        document.getElementById('help-modal').classList.remove('hidden');
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = `✗ Erreur: ${message}`;
        document.body.appendChild(errorDiv);
    }

    updateStats() {
        this.stats.frameCount++;
        const now = Date.now();
        const delta = now - this.stats.lastTime;
        
        if (delta >= 1000) {
            this.stats.fps = this.stats.frameCount;
            this.stats.frameCount = 0;
            this.stats.lastTime = now;
            
            document.getElementById('fps').textContent = this.stats.fps;
        }

        // Position X, Y, Z (mise à jour immédiate)
        const pos = this.camera.position;
        document.getElementById('pos-x').textContent = pos.x.toFixed(2);
        document.getElementById('pos-y').textContent = pos.y.toFixed(2);
        document.getElementById('pos-z').textContent = pos.z.toFixed(2);

        // Zoom
        const zoomLevel = this.camera.zoom || 1;
        document.getElementById('zoom').textContent = `${(1/zoomLevel).toFixed(2)}x`;
    }

    onWindowResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.controls) {
            this.controls.update(); // sans argument
        }

        this.updateStats();
        this.updateMinimap();
        this.renderer.render(this.scene, this.camera);
    }


    formatNumber(num) {
        return Math.floor(num).toLocaleString('fr-FR');
    }
}

// Initialiser au chargement du DOM
document.addEventListener('DOMContentLoaded', () => {
    console.log("🌐 DOM chargé, démarrage de la navigation 3D...");
    new SceneNavigator();
});
