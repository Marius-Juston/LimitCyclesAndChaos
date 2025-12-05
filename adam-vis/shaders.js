// Create a file called 'shaders.js'
export const shaders = {

  vertex: `
      attribute vec3 aColor;
      varying vec3 vColor;
      uniform float size;
      void main() {
        vColor = aColor;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,

  fragment: `
      varying vec3 vColor;
      uniform float alpha;
      void main() {
        vec2 p = gl_PointCoord - vec2(0.5);
        float dist = length(p);
        float pixelSize = fwidth(dist);
        float edge = 0.475;
        float multiplier = smoothstep(edge + pixelSize, edge - pixelSize, dist);
        gl_FragColor = vec4(vColor, alpha * multiplier);
      }
    `
};
