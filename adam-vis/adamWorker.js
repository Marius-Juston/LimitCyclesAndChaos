onmessage = function (e) {
    const { params, steps, warmup } = e.data;
    const vertexBuffer = new Float32Array(steps * 3);
    let optimizer = params.optimizer;

    let fun = stepAdam;

    if(optimizer == "adam"){
        fun = stepAdam;
    }else if(optimizer == "adamw"){
        fun = stepAdamW;
    }

    // Initialize variables
    let x = 1.0, m = 0.0, v = 1.0;

    // Warmup iterations
    for (let i = 0; i < warmup; i++) {
        [x, m, v] = fun(x, m, v, params);
    }

    // Compute visualization points
    for (let i = 0; i < steps; i++) {
        [x, m, v] = fun(x, m, v, params);
        vertexBuffer[i * 3] = x;
        vertexBuffer[i * 3 + 1] = m;
        vertexBuffer[i * 3 + 2] = v;
    }

    // Normalize the vertices (same normalization logic as before)
    normalizeVertices(vertexBuffer);

    // Send the computed vertices back to the main thread,
    // transferring the underlying buffer to improve performance.
    postMessage({ vertices: vertexBuffer }, [vertexBuffer.buffer]);
};


// Adam
function stepAdam(x, m, v, params) {
    if(params.weight_decay != 0.0){
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