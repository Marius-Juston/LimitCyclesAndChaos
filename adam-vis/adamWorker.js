onmessage = function (e) {
    const { params, steps, warmup } = e.data;
    const vertexBuffer = new Float32Array(steps * 3);

    let optimizer = params.optimizer;

    const enable_qr = params.enable_qr;
    const enable_clv = params.enable_clv;

    const clvBuffer = new Float32Array(enable_clv? steps * 3 * 3: 0);

    const fun = (optimizer === "adam")? stepAdam: stepAdamW ;
    const funD = (optimizer === "adam")? stepAdamJ: stepAdamWJ ;

    // Initialize variables
    let x = 1.0, m = 0.0, v = 1.0;

    let q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

    let le = [0, 0, 0];

    let historyQ = [];
    let historyR = [];

    // Warmup iterations
    for (let i = 0; i < warmup; i++) {
        [x, m, v] = fun(x, m, v, params);

        if(enable_qr){
            // Used to align the q matrix beforehand
            q = LEStep(x, m, v, q, le, funD, params).Q;
        }
    }

    le = [0, 0, 0];

    // Compute visualization points
    for (let i = 0; i < steps; i++) {
        [x, m, v] = fun(x, m, v, params);

        if(enable_qr){
            const res =  LEStep(x, m, v, q, le, funD, params);

            q = res.Q;

            if(enable_clv){
                historyQ.push(res.Q); 
                historyR.push(res.R);
            }
        }

        vertexBuffer[i * 3] = x;
        vertexBuffer[i * 3 + 1] = m;
        vertexBuffer[i * 3 + 2] = v;
    }

    if(enable_qr){
        for(let i = 0; i < le.length; i++){                
            le[i] /= (steps);
        }

        if(enable_clv) {
            computeCLVs(historyQ, historyR, clvBuffer, steps);
        }
    }

    // Normalize the vertices (same normalization logic as before)
    normalizeVertices(vertexBuffer);

    // Send the computed vertices back to the main thread,
    // transferring the underlying buffer to improve performance.
    postMessage({ vertices: vertexBuffer, clv: clvBuffer, le:le }, [vertexBuffer.buffer, clvBuffer.buffer]);
};

function LEStep(x, m, v, q, le, funcD, params){
    const D = funcD(x, m, v, params);

    const qtilde = matMul3(D, q);

    const res = qr3(qtilde);

    for(let i = 0; i < le.length; i++){                
        le[i] += Math.log(Math.abs(res.R[i][i]));
    }

    return res;
}


// Adam
function stepAdam(x, m, v, params) {
    if(params.weight_decay !== 0.0){
        x = x + params.weight_decay * x;
    }

    m = params.b1 * m + (1 - params.b1) * x;
    v = params.b2 * v + (1 - params.b2) * (x * x);
    x = x - params.lr * m / (Math.sqrt(v) + params.epsilon);
    return [x, m, v];
}

// AdamW
function stepAdamW(x, m, v, params) {
    x = x - params.weight_decay * params.lr * x;

    m = params.b1 * m + (1 - params.b1) * x;
    v = params.b2 * v + (1 - params.b2) * (x * x);
    x = x - params.lr * m / (Math.sqrt(v) + params.epsilon);
    return [x, m, v];
}

function normalizeVertices(vertices) {
    let max = [-Infinity, -Infinity, -Infinity];
    let min = [Infinity, Infinity, Infinity];
    let avg_z = 0;
    const len = vertices.length / 3;
    for (let i = 0; i < len; i++) {
        min[0] = Math.min(min[0], vertices[i * 3]);
        min[1] = Math.min(min[1], vertices[i * 3 + 1]);
        min[2] = Math.min(min[2], vertices[i * 3 + 2]);
        max[0] = Math.max(max[0], vertices[i * 3]);
        max[1] = Math.max(max[1], vertices[i * 3 + 1]);
        max[2] = Math.max(max[2], vertices[i * 3 + 2]);
        avg_z += vertices[i * 3 + 2] / len;
    }
    for (let i = 0; i < len; i++) {
        vertices[i * 3] /= max[0] - min[0];
        vertices[i * 3 + 1] /= max[1] - min[1];
        vertices[i * 3 + 2] /= 2 * avg_z;
    }
}

// [params.weight_decay, params.lr, params.b1, params.b2, x , m, v, params.epsilon]
function stepAdamJ(x, m, v, params) {
    const b2_c = -1  + params.b2;
    const b1_c = -1 +  params.b1;

    const gamma = (1 + params.weight_decay);
    const gamma_square = gamma * gamma;

    const x_gamma = x * gamma;
    const x_gamma_sq = x_gamma * x_gamma;

    const x_g_b1 = x_gamma + (m -x_gamma) * params.b1;

    const int_eps = params.epsilon + Math.sqrt(-x_gamma_sq * b2_c + v * params.b2);
    const int_eps_sq = int_eps * int_eps;

    const top1_a = x * params.lr * gamma_square * x_g_b1 * b2_c;
    const bottom1_a = Math.sqrt(-x_gamma_sq * b2_c + v * params.b2) * int_eps_sq;

    const top1_b = b1_c * gamma
    const bottom1_b = int_eps;

    const top3 = params.lr * x_g_b1 * params.b2;
    const bottom3 = 2 * bottom1_a;

    const D11 = 1 - (top1_a/bottom1_a) + (top1_b/bottom1_b);
    const D12 = -params.lr * params.b1 / int_eps;
    const D13 = top3 / bottom3;

    const D21 = -gamma*b1_c;
    const D22 = params.b1;
    const D23 = 0;

    const D31 = -2 * x * gamma_square * b2_c;
    const D32 = 0;
    const D33 = params.b2;

    const D =  [[D11, D12, D13],
                [D21, D22, D23],
                [D31, D32, D33]];

    return D;
}


function stepAdamWJ(x, m, v, params) {
    const b2_c = -1 + params.b2;
    const b1_c = -1 + params.b1;

    const x_sq = x * x;

    const x_g_b1 = x + (m -x) * params.b1;

    const int_eps = params.epsilon + Math.sqrt(x_sq + (v - x_sq) * params.b2);
    const int_eps_sq = int_eps * int_eps;

    const top1_a = -x * params.lr  * x_g_b1 * b2_c;
    const bottom1_a = Math.sqrt(x_sq + (v - x_sq) * params.b2) * int_eps_sq;

    const top1_b = b1_c * params.lr;
    const bottom1_b = int_eps;

    const top3 = params.lr * x_g_b1 * params.b2;
    const bottom3 = 2 * bottom1_a;

    const D11 = 1 - params.lr * params.weight_decay + (top1_a/bottom1_a) + (top1_b/bottom1_b);
    const D12 = -params.lr * params.b1 / int_eps;
    const D13 = top3 / bottom3;

    const D21 = 1-params.b1;
    const D22 = params.b1;
    const D23 = 0;

    const D31 = -2 * x * b2_c;
    const D32 = 0;
    const D33 = params.b2;

    const D =  [[D11, D12, D13],
                [D21, D22, D23],
                [D31, D32, D33]];

    return D;
}


function computeCLVs(histQ, histR, buffer, steps) {
    // Initialize Coefficient matrix C as Identity
    // This represents the vectors in the basis of Q.
    let C = [[1,0,0], [0,1,0], [0,0,1]];

    // Iterate backwards from the last step
    for (let i = steps - 1; i >= 0; i--) {
        const R = histR[i];
        const Q = histQ[i];

        // 1. Invert R (Upper Triangular)
        const R_inv = invertUpperTriangular3(R);

        // 2. Back-propagate C: C_new = R_inv * C
        C = matMul3(R_inv, C);

        // 3. Normalize columns of C to prevent numerical overflow/underflow
        // The direction is what matters, not the magnitude here.
        normalizeColumns3(C);

        // 4. Compute actual CLV in phase space: V = Q * C
        const V = matMul3(Q, C);

        // 5. Write to buffer
        // Layout: Step i -> [Vec1_x, Vec1_y, Vec1_z, Vec2_x, ... Vec3_z]
        const offset = i * 9;
        
        // Vector 1 (Most unstable)
        buffer[offset + 0] = V[0][0];
        buffer[offset + 1] = V[1][0];
        buffer[offset + 2] = V[2][0];

        // Vector 2
        buffer[offset + 3] = V[0][1];
        buffer[offset + 4] = V[1][1];
        buffer[offset + 5] = V[2][1];

        // Vector 3 (Least unstable / Most stable)
        buffer[offset + 6] = V[0][2];
        buffer[offset + 7] = V[1][2];
        buffer[offset + 8] = V[2][2];
    }
}

function invertUpperTriangular3(R) {
    const S = [[0,0,0], [0,0,0], [0,0,0]];
    
    // Diagonal elements: 1 / R_ii
    const a = R[0][0], d = R[1][1], f = R[2][2];
    S[0][0] = 1.0 / a;
    S[1][1] = 1.0 / d;
    S[2][2] = 1.0 / f;

    // Off-diagonals (explicit back-substitution for 3x3)
    const b = R[0][1], c = R[0][2], e = R[1][2];

    // S[0][1]
    S[0][1] = -b * S[0][0] * S[1][1]; 
    
    // S[1][2]
    S[1][2] = -e * S[1][1] * S[2][2];

    // S[0][2]
    S[0][2] = -(b * S[1][2] + c * S[2][2]) * S[0][0];

    return S;
}

function qr3(A) {
  // Validate shape
  if (!Array.isArray(A) || A.length !== 3 || A.some(r => !Array.isArray(r) || r.length !== 3)) {
    throw new Error("Input must be a 3x3 array");
  }

  // Copy A into R (Float64)
  const R = [
    [ +A[0][0], +A[0][1], +A[0][2] ],
    [ +A[1][0], +A[1][1], +A[1][2] ],
    [ +A[2][0], +A[2][1], +A[2][2] ]
  ];

  // Initialize Q as identity
  const Q = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
  ];

  // Helper: apply Householder defined by unit vector v of length m-k
  // to submatrix rows k..2 of R and to Q (left-multiplying)
  function applyHouseholderToSubmatrix(v, k) {
    // v length = 3-k, indexes 0..(2-k) correspond to rows k..2
    const len = 3 - k;
    // Apply to R: for each column j = k..2, compute dot = v^T * R[k..2][j]
    for (let j = k; j < 3; ++j) {
      let dot = 0.0;
      for (let t = 0; t < len; ++t) dot += v[t] * R[k + t][j];
      if (dot !== 0.0) {
        const scale = 2 * dot;
        for (let t = 0; t < len; ++t) R[k + t][j] -= scale * v[t];
      }
    }
    // Apply to Q: Q := H * Q where H = I - 2 v v^T on rows k..2
    for (let j = 0; j < 3; ++j) {
      let dot = 0.0;
      for (let t = 0; t < len; ++t) dot += v[t] * Q[k + t][j];
      if (dot !== 0.0) {
        const scale = 2 * dot;
        for (let t = 0; t < len; ++t) Q[k + t][j] -= scale * v[t];
      }
    }
  }

  // Householder for column 0 (zero R[1][0], R[2][0])
  {
    // x = [R[0][0], R[1][0], R[2][0]]^T
    const x0 = R[0][0], x1 = R[1][0], x2 = R[2][0];
    const normx = Math.hypot(x0, x1, x2); // stable sqrt(x0^2 + x1^2 + x2^2)
    if (normx > 0) {
      const sign = x0 >= 0 ? 1 : -1;
      // v = x + sign*normx * e1  (length 3)
      let v0 = x0 + sign * normx, v1 = x1, v2 = x2;
      // normalize v
      const vnorm = Math.hypot(v0, v1, v2);
      if (vnorm > 0) {
        v0 /= vnorm; v1 /= vnorm; v2 /= vnorm;
        applyHouseholderToSubmatrix([v0, v1, v2], 0);
      }
    }
  }

  // Householder for column 1 (zero R[2][1]); only rows 1..2 involved
  {
    // x = [R[1][1], R[2][1]]^T
    const x0 = R[1][1], x1 = R[2][1];
    const normx = Math.hypot(x0, x1);
    if (normx > 0) {
      const sign = x0 >= 0 ? 1 : -1;
      let v0 = x0 + sign * normx, v1 = x1;
      const vnorm = Math.hypot(v0, v1);
      if (vnorm > 0) {
        v0 /= vnorm; v1 /= vnorm;
        applyHouseholderToSubmatrix([v0, v1], 1);
      }
    }
  }

  // Now R should be upper triangular (numerical tiny values possible)
  // Force tiny entries below diagonal to zero for clean output
  const eps = 1e-15;
  if (Math.abs(R[1][0]) < eps) R[1][0] = 0;
  if (Math.abs(R[2][0]) < eps) R[2][0] = 0;
  if (Math.abs(R[2][1]) < eps) R[2][1] = 0;

  // Return Q and R. Note: The algorithm produces R such that A = Q * R
  return { Q: Q, R: R };
}

function matMul3(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i=0;i<3;++i) for (let j=0;j<3;++j) {
    let s = 0;
    for (let k=0;k<3;++k) s += A[i][k]*B[k][j];
    C[i][j]=s;
  }
  return C;
}

function normalizeColumns3(M) {
    for (let col = 0; col < 3; col++) {
        let sumSq = 0;
        for (let row = 0; row < 3; row++) {
            sumSq += M[row][col] * M[row][col];
        }
        const mag = Math.sqrt(sumSq);
        if (mag > 1e-15) {
            for (let row = 0; row < 3; row++) {
                M[row][col] /= mag;
            }
        }
    }
}