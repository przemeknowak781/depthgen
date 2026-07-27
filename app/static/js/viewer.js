import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const MATERIALS = {
  clay:     { color: 0xc8a48a, roughness: 0.95, metalness: 0.0 },
  plaster:  { color: 0xe8e6e0, roughness: 0.85, metalness: 0.0 },
  bronze:   { color: 0xb07d3a, roughness: 0.38, metalness: 0.95 },
  steel:    { color: 0xb9c2cc, roughness: 0.28, metalness: 1.0 },
};

export class Viewer {
  constructor(el) {
    this.el = el;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0e1116);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 20000);
    this.camera.position.set(0, -160, 130);
    this.camera.up.set(0, 0, 1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    el.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    // światło kluczowe pod kątem — kluczowe dla oceny płaskorzeźby
    this.key = new THREE.DirectionalLight(0xfff4e6, 3.0);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    this.key.shadow.bias = -0.0004;
    this.key.shadow.normalBias = 0.02;
    this.scene.add(this.key, this.key.target);
    this.fill = new THREE.DirectionalLight(0x9ec2ff, 0.35);
    this.fill.position.set(-120, 60, 60);
    this.scene.add(this.fill);
    this.rim = new THREE.DirectionalLight(0xffffff, 0.5);
    this.rim.position.set(0, 140, -60);
    this.scene.add(this.rim);
    this.scene.add(new THREE.HemisphereLight(0x4b5a6b, 0x0f1216, 0.35));

    this.grid = new THREE.GridHelper(400, 40, 0x2a323c, 0x1c2229);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.position.z = -0.01;
    this.scene.add(this.grid);

    this.material = new THREE.MeshStandardMaterial({ ...MATERIALS.clay, side: THREE.DoubleSide, flatShading: false });
    this.mesh = null;
    this.lightAngle = 45;
    this.spin = false;
    this.radius = 100;

    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  setLight(deg) {
    this.lightAngle = deg;
    const r = Math.max(this.radius * 2.2, 60), a = deg * Math.PI / 180;
    this.key.position.set(Math.cos(a) * r, Math.sin(a) * r, r * 0.6);  // światło ślizgowe
    const d = this.key.shadow.camera;
    d.left = -r; d.right = r; d.top = r; d.bottom = -r; d.near = 1; d.far = r * 5;
    d.updateProjectionMatrix();
  }

  setMaterial(kind) {
    const old = this.material;
    if (kind === 'normals') {
      this.material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
    } else {
      this.material = new THREE.MeshStandardMaterial({
        ...MATERIALS[kind] || MATERIALS.clay, side: THREE.DoubleSide,
      });
    }
    this.material.wireframe = old.wireframe;
    if (this.mesh) this.mesh.material = this.material;
    old.dispose();
  }

  setWireframe(on) {
    this.material.wireframe = on;
  }

  /** Wczytuje pakiet binarny: [u32 nv, u32 nf][pos f32][nrm f32][idx u32] */
  load(buffer) {
    const head = new Uint32Array(buffer, 0, 2);
    const nv = head[0], nf = head[1];
    let off = 8;
    const pos = new Float32Array(buffer, off, nv * 3); off += nv * 12;
    const nrm = new Float32Array(buffer, off, nv * 3); off += nv * 12;
    const idx = new Uint32Array(buffer, off, nf * 3);

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeBoundingSphere();
    g.computeBoundingBox();

    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.mesh.geometry = g;
    } else {
      this.mesh = new THREE.Mesh(g, this.material);
      this.mesh.castShadow = true;
      this.mesh.receiveShadow = true;
      this.scene.add(this.mesh);
    }
    const bb = g.boundingBox;
    this.radius = Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, 1) * 0.5;
    this.grid.scale.setScalar(Math.max(this.radius / 100, 0.05) * 2);
    this.setLight(this.lightAngle);
    return { nv, nf };
  }

  fit(front = false) {
    if (!this.mesh) return;
    const bb = this.mesh.geometry.boundingBox;
    const c = bb.getCenter(new THREE.Vector3());
    const s = bb.getSize(new THREE.Vector3());
    const d = Math.max(s.x, s.y, s.z) * 1.5;
    this.controls.target.copy(c);
    if (front) this.camera.position.set(c.x, c.y - 0.02, c.z + d);
    else this.camera.position.set(c.x + d * 0.05, c.y - d * 0.85, c.z + d * 0.62);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  tick() {
    if (this.spin && this.mesh) this.mesh.rotation.z += 0.004;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
