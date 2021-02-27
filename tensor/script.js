import * as tf from "@tensorflow/tfjs"
// 基本概念，基础值乘以起对应的权重
const input = [1,2,3,4]
const w = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]; // 权重
const output = [0,0,0,0]

// 传统for循环
for(let i=0;i<w.length;i++){
    for(let j=0;j<input.length;j++){
        output[i] += input[j] * w[i][j];
    }
}
console.log(output)

// tensor方法  dot点乘运算
tf.tensor(w).dot(tf.tensor(input)).print()