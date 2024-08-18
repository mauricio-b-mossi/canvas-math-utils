export function drawGrid(ctx, canvas, squareSize, lineColor = "#fff", backgroundColor = "#000") {

    ctx.fillStyle = backgroundColor
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    ctx.strokeStyle = lineColor

    // Drawing vertical line.
    ctx.beginPath()
    ctx.moveTo(canvas.width / 2, 0)
    ctx.lineTo(canvas.width / 2, canvas.height)
    ctx.stroke()

    // Drawing horizontal line.
    ctx.beginPath()
    ctx.moveTo(0, canvas.height / 2)
    ctx.lineTo(canvas.width, canvas.height / 2)
    ctx.stroke()

    ctx.lineWidth = 0.10

    let sy = squareSize

    while (canvas.height / 2 + sy < canvas.height) {

        ctx.beginPath()
        ctx.moveTo(0, canvas.height / 2 + sy)
        ctx.lineTo(canvas.width, canvas.height / 2 + sy)
        ctx.stroke()

        ctx.beginPath()
        ctx.moveTo(0, canvas.height / 2 - sy)
        ctx.lineTo(canvas.width, canvas.height / 2 - sy)
        ctx.stroke()

        sy += squareSize
    }

    let sx = squareSize

    while (canvas.width / 2 + sx < canvas.width) {

        ctx.beginPath()
        ctx.moveTo(canvas.width / 2 + sx, 0)
        ctx.lineTo(canvas.width / 2 + sx, canvas.height)
        ctx.stroke()

        ctx.beginPath()
        ctx.moveTo(canvas.width / 2 - sx, 0)
        ctx.lineTo(canvas.width / 2 - sx, canvas.height)
        ctx.stroke()

        sx += squareSize
    }

    ctx.strokeStyle = "#000"
    ctx.lineWidth = 1
}

export function drawVector2d(ctx, canvas, squareSize, vec, offset = { x: canvas.width / 2, y: canvas.height / 2 }, lineWidth = 1, arrowWidth = 5, vecColor = "#fff") {

    ctx.strokeStyle = vecColor
    ctx.fillStyle = vecColor
    ctx.lineWidth = lineWidth

    // Drawing Line
    ctx.beginPath()
    ctx.moveTo(offset.x, offset.y)
    ctx.lineTo(offset.x + vec.x * squareSize, offset.y - vec.y * squareSize) // - y since top (0,0) bottom (xmax, ymax).
    ctx.stroke()

    const lenV = len(vec)

    // Drawing Arrow
    const norm = { x: vec.y / lenV, y: -vec.x / lenV }
    const unit = { x: vec.x / lenV, y: vec.y / lenV }
    const scale = lenV * squareSize - arrowWidth // Scale factor in pixels.

    ctx.beginPath()
    ctx.moveTo((offset.x + unit.x * scale), (offset.y - unit.y * scale))
    ctx.lineTo((offset.x + unit.x * scale) + norm.x * arrowWidth, (offset.y - unit.y * scale) - norm.y * arrowWidth)
    ctx.lineTo(offset.x + vec.x * squareSize, offset.y - vec.y * squareSize)
    ctx.lineTo((offset.x + unit.x * scale) - norm.x * arrowWidth, (offset.y - unit.y * scale) + norm.y * arrowWidth)
    ctx.closePath()
    ctx.fill()

    ctx.strokeStyle = "#000"
    ctx.fillStyle = "#000"
    ctx.lineWidth = 1

}

/**
 * Calculates the Euclidean length (magnitude) of a vector.
 * 
 * The vector can be represented as any object where the only properties represent the components of the vector. 
 * This means that an array would work, since each index represents a component, as well as an object like `{x: ..., y: ...}`. 
 * However, more complex objects, such as those with methods, will not work correctly.
 *
 * @param {Object|Array<number>} vec - The vector for which to calculate the length.
 * @returns {number} The Euclidean length of the vector.
 *
 * @example
 * const vector1 = [3, 4];
 * const length1 = len(vector1);
 * console.log(length1);
 * // Output: 5
 * 
 * const vector2 = {x: 3, y: 4};
 * const length2 = len(vector2);
 * console.log(length2);
 * // Output: 5
 */
export function len(vec) {
    let sum = 0;
    for (const comp in vec) {
        sum += vec[comp] * vec[comp];
    }
    return Math.sqrt(sum);
} export function len(vec) {
    let sum = 0
    for (const comp in vec) {
        sum += vec[comp] * vec[comp]
    }
    return Math.sqrt(sum)
}

/**
 * Transposes a matrix.
 * 
 * @param {number[][]} A - The matrix to be transposed, in the form A[row][col].
 * @returns {number[][]} The transposed matrix, where rows are converted to columns.
 * 
 * @example
 * const matrix = [
 *   [1, 2, 3],
 *   [4, 5, 6]
 * ];
 * const transposedMatrix = transpose(matrix);
 * console.log(transposedMatrix);
 * // Output:
 * // [
 * //   [1, 4],
 * //   [2, 5],
 * //   [3, 6]
 * // ]
 */
export function transpose(A) {

    const at = Array(A[0].length)

    for (let i = 0; i < at.length; i++) {
        at[i] = Array(A.length).fill(0);
    }

    for (let i = 0; i < A[0].length; i++) {
        for (let j = 0; j < A.length; j++) {
            at[i][j] = A[j][i]
        }
    }

    return at
}

/**
 * Multiplies two matrices.
 * 
 * @param {number[][]} A - The first matrix, in the form A[row][col].
 * @param {number[][]} B - The second matrix, in the form B[row][col].
 * @returns {number[][]} The product of the matrices A and B.
 * 
 * @throws {Error} If the number of columns in A is not equal to the number of rows in B.
 * 
 * @example
 * const matrixA = [
 *   [1, 2],
 *   [3, 4]
 * ];
 * const matrixB = [
 *   [5, 6],
 *   [7, 8]
 * ];
 * const productMatrix = matMult(matrixA, matrixB);
 * console.log(productMatrix);
 * // Output:
 * // [
 * //   [19, 22],
 * //   [43, 50]
 * // ]
 */
export function matMult(A, B) {
    if (A[0].length !== B.length)
        throw new Error("Dimensions of the matrix do not match. Number of Cols of A (A[0].length) is not equal to number of Rows in B (B.length). Matrix should be in form A[row][col].")

    const c = Array(A.length)

    for (let i = 0; i < A.length; i++) {
        c[i] = Array(B[0].length).fill(0)
    }

    for (let i = 0; i < c.length; i++) {
        for (let j = 0; j < c[0].length; j++) {
            let sum = 0
            for (let k = 0; k < A[0].length; k++) {
                sum += A[i][k] * B[k][j]
            }
            c[i][j] = sum
        }
    }

    return c
}

/**
 * Scales a vector by a scalar.
 * 
 * @param {number[]} vec - The vector to be scaled.
 * @param {number} s - The scalar by which to scale the vector.
 * @returns {number[]} A new vector that is the result of scaling the input vector by the scalar.
 * 
 * @example
 * const vector = [1, 2, 3];
 * const scaledVector = scale(vector, 2);
 * console.log(scaledVector);
 * // Output:
 * // [2, 4, 6]
 */
export function scale(vec, s) {
    const nvec = Array(vec.length)
    for (let i = 0; i < vec.length; i++) {
        nvec[i] = vec[i] * s
    }
    return nvec
}

/**
 * Subtracts one vector from another.
 * 
 * @param {number[]} vec1 - The vector from which to subtract.
 * @param {number[]} vec2 - The vector to subtract from vec1.
 * @returns {number[]} A new vector that is the result of the subtraction vec1 - vec2.
 * 
 * @throws {Error} If the vectors are not of the same length.
 * 
 * @example
 * const vector1 = [4, 5, 6];
 * const vector2 = [1, 2, 3];
 * const difference = subtract(vector1, vector2);
 * console.log(difference);
 * // Output:
 * // [3, 3, 3]
 */
export function subtract(vec1, vec2) {
    if (vec1.length !== vec2.length)
        throw new Error(`vec1 and vec2 have different lengths.\n- vec1: ${vec1}.\n- vec2: ${vec2}`)

    const nvec = Array(vec1.length)
    for (let i = 0; i < vec1.length; i++) {
        nvec[i] = vec1[i] - vec2[i]
    }
    return nvec
}

/**
 * Computes the Row-Reduced Echelon Form (RREF) of a matrix. 
 * This function does not mutate the input matrix.
 *
 * @param {number[][]} A - The input matrix to be row-reduced.
 * @returns {number[][]} A new matrix in Row-Reduced Echelon Form (RREF).
 *
 * @example
 * const matrix = [
 *   [2, 4, 6],
 *   [1, 3, 5],
 *   [7, 8, 9]
 * ];
 * const rrefMatrix = rref(matrix);
 * console.log(rrefMatrix);
 * // Output: 
 * // [
 * //   [1, 0, -1],
 * //   [0, 1, 2],
 * //   [0, 0, 0]
 * // ]
 */
export function rref(A) {
    const cA = JSON.parse(JSON.stringify(A));
    // There are as many pivots as the smallest dimension.
    for (let i = 0; i < (cA.length > cA[0].length ? cA[0].length : cA.length); i++) {

        // Finding pivot for ith col, if cA[i][i] = 0.
        if (cA[i][i] === 0) {
            // Find row s.t. cA[row][i] != 0.
            let row = null

            // Cannot swap with anything above pivot.
            for (let n = i + 1; n < cA.length; n++) {
                if (cA[n][i] !== 0) {
                    row = n
                    break
                }
            }

            if (row === null)
                // Matrix is singular.
                return normalizePivots(cA)


            // Changing rows.
            const temp = cA[i]
            cA[i] = cA[row]
            cA[row] = temp

        }

        if (cA[i][i] !== 1) {
            // Scaling the row s.t. pivot = 1.
            scale(cA[i], 1 / cA[i][i])
        }

        // Row Reduction step. 
        for (let k = 0; k < cA.length; k++) {
            if (k === i || cA[k][i] === 0)
                continue
            cA[k] = subtract(cA[i], scale(cA[k], cA[i][i] / cA[k][i]))
        }

    }

    return normalizePivots(cA)
}


/**
 * Normalizes the pivot rows of a matrix to have a pivot of 1.
 *
 * @param {number[][]} A - The input matrix to normalize.
 * @returns {number[][]} A new matrix where the pivot elements of the rows are 1.
 *
 * @example
 * const matrix = [
 *   [2, 4, 6],
 *   [1, 3, 5],
 *   [7, 8, 9]
 * ];
 * const normalizedMatrix = normalizePivots(matrix);
 * console.log(normalizedMatrix);
 * // Output: 
 * // [
 * //   [1, 2, 3],
 * //   [1, 3, 5],
 * //   [7, 8, 9]
 * // ]
 */
export function normalizePivots(A) {
    const cA = JSON.parse(JSON.stringify(A));
    for (let i = 0; i < (A.length > A[0].length ? A[0].length : A.length); i++) {
        if (cA[i][i] !== 1 && cA[i][i] !== 0) {
            cA[i] = scale(cA[i], 1 / A[i][i])
        }
    }
    return cA
}


/**
 * Extends matrix A by appending matrix B to its columns.
 * 
 * @param {number[][]} A - The original matrix to extend.
 * @param {number[][]} B - The matrix to append to A. Must have the same number of rows as A.
 * @returns {number[][]} A new matrix with B appended to A.
 * @throws {Error} If the number of rows in A and B do not match.
 */
export function extend(A, B) {
    const cA = JSON.parse(JSON.stringify(A));
    if (cA.length !== B.length)
        throw new Error(`Number of rows in B must match number of rows in A.\n- A: ${cA}\n- B: ${B}`)
    for (let i = 0; i < cA.length; i++) {
        for (let k = 0; k < B[0].length; k++)
            cA[i].push(B[i][k])
    }
    return cA
}

/**
 * Performs the Gauss-Jordan elimination algorithm on matrix A.
 * 
 * @param {number[][]} A - The matrix to perform Gauss-Jordan elimination on. Must be a square matrix.
 * @returns {number[][]} The matrix in reduced row echelon form (RREF).
 * @throws {Error} If A is not a square matrix.
 */
export function gaussJordan(A) {
    if (A.length !== A[0].length)
        throw new Error(`Dimensions (m*n) of matrix A must be equal.`)
    const cA = extend(A, eye(A.length))
    return rref(cA)
}

/**
 * Calculates the inverse of matrix A using the Gauss-Jordan elimination method.
 * 
 * @param {number[][]} A - The square matrix to invert.
 * @returns {number[][]} The inverse of matrix A.
 * @throws {Error} If A is singular (non-invertible).
 */
export function inv(A) {
    const cA = gaussJordan(A)
    for (let i = 0; i < A.length; i++) {
        if (cA[i][i] === 0)
            throw new Error(`Matrix is singular.`)
    }
    return matSlice(cA, [0, A.length], [A.length - 1, 2 * A.length - 1])
}

/**
 * Creates an identity matrix of size n x n.
 * 
 * @param {number} n - The size of the identity matrix.
 * @returns {number[][]} The n x n identity matrix.
 */
export function eye(n) {
    const cA = []
    for (let i = 0; i < n; i++) {
        cA.push([])
        for (let k = 0; k < n; k++) {
            if (k === i)
                cA[i][k] = 1
            else
                cA[i][k] = 0
        }
    }
    return cA
}

/**
 * Extracts a submatrix from matrix A using the specified start and end indices.
 * 
 * @param {number[][]} A - The matrix to slice.
 * @param {number[]} start - The starting index [row, column] for the slice.
 * @param {number[]} end - The ending index [row, column] for the slice.
 * @returns {number[][]} The sliced submatrix.
 * @throws {Error} If the slice indices are invalid.
 */
export function matSlice(A, start = [0, 0], end = [A.length - 1, A[0].length - 1]) {
    if (start.length != 2 || end.length != 2 || start[0] > end[0] || start[1] > end[1] || start[0] < 0 || start[0] > A.length - 1 || start[1] < 0 || start[1] > A[0].length - 1 || end[0] > A.length - 1 || end[1] > A[0].length - 1)
        throw new Error(`Invalid slice. Slices must have two components [i, y], be non-negative, within the range of indices of A, and start >= end.`)

    const cA = []
    for (let i = 0; i < (end[0] - start[0] + 1); i++) {
        cA.push([])
        for (let k = 0; k < (end[1] - start[1] + 1); k++) {
            cA[i].push(A[start[0] + i][start[1] + k])
        }
    }
    return cA
}

/**
 * Builds a square matrix from a list of arguments.
 * 
 * @param {...number} args - The elements to populate the matrix, in row-major order.
 * @returns {number[][]} The resulting square matrix.
 * @throws {Error} If the number of arguments is not a perfect square.
 */
export function squareMatrixBuilder(...args) {
    const cA = []
    const sqrt = Math.sqrt(args.length)
    if (!Number.isInteger(sqrt)) {
        throw new Error("Number of arguments must be a perfect square.")
    }
    for (let i = 0; i < sqrt; i++) {
        cA.push([])
        for (let k = 0; k < sqrt; k++) {
            cA[i].push(args[sqrt * i + k])
        }
    }
    return cA
}
