import * as THREE from 'three';

// Define geometries once for reuse
const ARROW_BODY = new THREE.CylinderGeometry(1, 1, 1, 12)
    .rotateX(Math.PI / 2)
    .translate(0, 0, 0.5);

const ARROW_HEAD = new THREE.ConeGeometry(1, 1, 12)
    .rotateX(Math.PI / 2)
    .translate(0, 0, -0.5);

export function Arrow(fx, fy, fz, ix, iy, iz, thickness, color, alpha = 1.0) {
    var material = new THREE.MeshBasicMaterial({
        color: color,
        transparent: alpha < 1.0,
        opacity: alpha
    });

    var length = Math.sqrt((ix - fx) ** 2 + (iy - fy) ** 2 + (iz - fz) ** 2);

    var body = new THREE.Mesh(ARROW_BODY, material);
    body.scale.set(thickness, thickness, length - 10 * thickness);

    var head = new THREE.Mesh(ARROW_HEAD, material);
    head.position.set(0, 0, length);
    head.scale.set(3 * thickness, 3 * thickness, 10 * thickness);

    var arrow = new THREE.Group();
    arrow.position.set(ix, iy, iz);
    arrow.lookAt(fx, fy, fz);
    arrow.add(body, head);

    return arrow;
}