import * as tf from "@tensorflow/tfjs"
// 使用包去机器学习，注意使用parcel,这样能在js中直接使用es6的语法
var a = tf.tensor([1,2,[1,2,3]])

a.print()

console.log(a)