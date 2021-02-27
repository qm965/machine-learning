// 单个神经元线性回归训练

import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

window.onload = async () => {
    const xs = [1,2,3,4];
    const ys = [1,3,5,7];

    tfvis.render.scatterplot(
        { name: '线性回归训练集' },
        { values: xs.map((x,i)=>({x,y:ys[i]})) },
        { xAxisDomain: [0,5], yAxisDomain: [0,8] }
    );
    
    const model = tf.sequential();  // 上一个神经元的输入时当前神经元的输出,一系列
    // 加一个层，units是神经元个数，inputShape是形状[1,2]代表一个二维矩阵,数字代表元素个数，如[[1,3]]
    model.add(tf.layers.dense({units: 1, inputShape: [1]}))
    // 损失函数，此处均方误差。，，优化器降低损失：：0.1学习率
    model.compile({loss: tf.losses.meanSquaredError,optimizer: tf.train.sgd(0.1)})
    
    const inputs = tf.tensor(xs);
    const labels = tf.tensor(ys);

    // await 等待模型训练完成，之后就可以做预测
    await model.fit(inputs,labels,{
        // 批量时的个数
        batchSize: 4,
        // 迭代次数
        epochs: 100,
        // 展示训练过程
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练过程'},
            ['loss']
        )
    })
    // 预测
    const output = model.predict(tf.tensor([5]))
    // 转化输出为js类型值
    console.log(output.dataSync())
}