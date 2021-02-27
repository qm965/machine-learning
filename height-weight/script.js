// 身高体重训练，归一化训练

import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
window.onload= async()=>{
    const heights = [150,160,170];
    const weights = [40,50,60];

    tfvis.render.scatterplot(
        {name: '身高体重训练数据'},
        { values: heights.map((x,i) => ({x,y: weights[i]})) },
        {
            xAxisDomain: [140,180],
            yAxisDomain: [30,70]
        }
    )

    // 归一化，将范围压缩到0，1之间。。全部减去初始值，并除以长度
    const inputs= tf.tensor(heights).sub(150).div(20);
    // inputs.print();
    const labels = tf.tensor(weights).sub(40).div(20)
    // labels.print();

    const model = tf.sequential();  // 上一个神经元的输入时当前神经元的输出,一系列
    // 加一个层，units是神经元个数，inputShape是形状[1,2]代表一个二维矩阵,数字代表元素个数，如[[1,3]]
    model.add(tf.layers.dense({units: 1, inputShape: [1]}))
    // 损失函数，此处均方误差。，，优化器降低损失：：0.1学习率
    model.compile({loss: tf.losses.meanSquaredError,optimizer: tf.train.sgd(0.1)})
    // await 等待模型训练完成，之后就可以做预测
    await model.fit(inputs,labels,{
        // 批量时的个数
        batchSize: 3,
        // 迭代次数
        epochs: 200,
        // 展示训练过程
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练过程'},
            ['loss']
        )
    })
    const h = 165;
    // 预测
    const output = model.predict(tf.tensor([h]).sub(150).div(20)) // 预测同样归一化
    // 转化输出为js类型值
    console.log(`预测如果身高为${h}cm,那么体重为：：`,output.mul(20).add(40).dataSync()[0],'千克')  // 反归一化
}