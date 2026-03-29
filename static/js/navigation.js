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
        this.renderer.shadowMap.type = THREE.PCFShadowShadowMap;
        container.appendChild(this.renderer.domElement);

        // Lumières
        this.setupLighting();

        // Événements de redimensionnement
        window.addEventListener('resize', () => this.onWindowResize());
    }

    setupLighting() {
        console.log("💡 Configuration des lumières...");
        
        // Lumière ambiante
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Lumière directionnelle (soleil)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
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

        // Lumière point pour relief
        const pointLight = new THREE.PointLight(0x6699ff, 0.3);
        pointLight.position.set(-5, 5, -5);
        this.scene.add(pointLight);
    }

    async loadModel() {
        console.log("📦 Chargement du modèle GLB...");
        
        try {
            const loader = new THREE.GLTFLoader();
            const gltf = await new Promise((resolve, reject) => {
                loader.load('/model.glb', resolve, undefined, reject);
            });

            this.model = gltf.scene;
            this.model.castShadow = true;
            this.model.receiveShadow = true;
            
            // Traverser tous les enfants pour activer les ombres
            this.model.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    // Améliorer le matériau
                    if (child.material) {
                        child.material.side = THREE.FrontSide;
                    }
                }
            });

            this.scene.add(this.model);
            
            // Centrer le modèle (optionnel)
            this.centerModel();
            
            console.log("✓ Modèle chargé");
        } catch (error) {
            throw new Error(`Erreur chargement modèle: ${error.message}`);
        }
    }

    centerModel() {
        if (!this.model) return;
        
        const box = new THREE.Box3().setFromObject(this.model);
        const center = box.getCenter(new THREE.Vector3());
        
        // Déplacer le modèle pour que son centre soit à l'origine
        this.model.position.sub(center);
    }

    initControls() {
        console.log("🎮 Initialisation des contrôles...");
        
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enableZoom = true;
        this.controls.zoomSpeed = 1.5;
        this.controls.enablePan = true;
        this.controls.autoRotate = false;
        
        // Configuration initiale
        const target = new THREE.Vector3(...this.cameraConfig.target);
        this.controls.target.copy(target);
        this.controls.update();
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
        
        // Clavier
        window.addEventListener('keydown', (e) => this.handleKeyPress(e));
        
        // Afficher la minimap
        this.updateMinimap();
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
        
        // Fond
        ctx.fillStyle = '#222';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Grille
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const pos = (i / 4) * canvas.width;
            ctx.beginPath();
            ctx.moveTo(pos, 0);
            ctx.lineTo(pos, canvas.height);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(0, pos);
            ctx.lineTo(canvas.width, pos);
            ctx.stroke();
        }
        
        // Contour du modèle (vue XZ/top-down)
        const bounds = this.minimapData.bounds_xz;
        if (bounds) {
            const scaleX = canvas.width / (bounds.max[0] - bounds.min[0] || 1);
            const scaleZ = canvas.height / (bounds.max[1] - bounds.min[1] || 1);
            
            ctx.fillStyle = '#4a7c59';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = '#7cfc00';
            ctx.fillRect(10, 10, canvas.width - 20, canvas.height - 20);
        }
        
        // Caméra actuelle
        const camPos = this.camera.position;
        const camX = ((camPos.x - (this.minimapData.bounds_xz?.min[0] || 0)) / 
                     ((this.minimapData.bounds_xz?.max[0] || 1) - (this.minimapData.bounds_xz?.min[0] || 0))) * canvas.width;
        const camZ = ((camPos.z - (this.minimapData.bounds_xz?.min[1] || 0)) / 
                     ((this.minimapData.bounds_xz?.max[1] || 1) - (this.minimapData.bounds_xz?.min[1] || 0))) * canvas.height;
        
        ctx.fillStyle = '#ff6b6b';
        ctx.beginPath();
        ctx.arc(camX, camZ, 5, 0, Math.PI * 2);
        ctx.fill();
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

        // Position caméra
        const pos = this.camera.position;
        document.getElementById('position').textContent = 
            `${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)}`;

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
        
        this.controls.update();
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
