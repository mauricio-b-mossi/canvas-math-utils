import {drawGrid} from "./utils.js"

window.onload = function() {
    const canvas = document.getElementById("canvas1")
    const ctx = canvas.getContext("2d")

    canvas.width = window.innerWidth
    canvas.height = window.innerHeight

    drawGrid(ctx, canvas, 10, "white", "black")
}
