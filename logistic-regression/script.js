import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from '@tensorflow/tfjs';
import { getData } from "./data";

window.onload = async ()=>{
    const data = getData(400)
    tfvis.render.scatterplot(
        {name: '逻辑回归训练数据'},
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0)
            ]
        }
    )

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [2],
        activation: 'sigmoid' // 这个激活函数作用就是把输出值压缩到0，1之间
    }))

    // 损失函数和优化器
    model.compile({loss: tf.losses.logLoss,optimizer: tf.train.adam(0.1)})

    const inputs  = tf.tensor(data.map(p=>[p.x,p.y]));
    const labels = tf.tensor(data.map(p=>p.label));

    await model.fit(inputs,labels,{
        batchSize: 40,
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            {name: '训练过程'},
            ['loss']
        )
    })

    window.predict = (form) => {
        const value = model.predict(tf.tensor([[form.x.value*1,form.y.value*1]]))
        alert(`预测为${value.dataSync()[0]}`)
    }
}