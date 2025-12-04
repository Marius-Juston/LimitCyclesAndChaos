// Create a file called 'shaders.js'
export const shaders = {
  vertex: `
      uniform float size;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,

  fragment: `
      uniform vec3 pointColor;
      uniform float alpha;
      void main() {
        vec2 p = gl_PointCoord - vec2(0.5);
        float dist = length(p);
        float pixelSize = fwidth(dist);
        float edge = 0.475;
        float multiplier = smoothstep(edge + pixelSize, edge - pixelSize, dist);
        gl_FragColor = vec4(1.0, 1.0, 1.0, alpha * multiplier);
      }
    `
};