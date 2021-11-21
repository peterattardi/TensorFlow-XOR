let model;
let in_array = [[1, 1], [0, 0], [0, 1], [1, 0]];
let out_array = [[0], [0], [1], [1]];
const tr_in = tf.tensor2d(in_array);
const tr_out = tf.tensor2d(out_array);

let resolution = 10;
function setup() {
    let canvas = createCanvas(501, 501);
    canvas.parent("canvas");
       
    doTheMLStuff();
    setTimeout(train, 10);
}



function draw() {
    background(51);
    wr = width / resolution;
    hr = height / resolution;
    for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
            tf.tidy(() => { y = predict([[i * wr / width, j * hr / height]])[0]; });
            y = nf(y, 1, 2);
            stroke(0);
            fill(y * 255);
            rect(i * wr, j * hr, wr, hr);
            noStroke();
            fill(255 - y * 255);
            textAlign(CENTER,CENTER);
            text(y, i * wr + wr/2 , j * hr + hr/2 );
        }

    }

}

function predict(input) {
    tensor = tf.tensor2d(input);
    return model.predict(tensor).dataSync();
}


function doTheMLStuff() {
    model = tf.sequential();

    const hidden = tf.layers.dense({
        units: 2,
        inputShape: [2],
        activation: 'sigmoid'
    });


    const output = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    });

    model.add(hidden);
    model.add(output);

    model.compile({
        optimizer: tf.train.adam(0.1),
        loss: tf.losses.meanSquaredError
    });

}


function train() {

    trainModel().then(result => {
        let loss = result.history.loss[0];
        console.log(loss);
        let msg1 = "Loss after last epoche = " + loss;
        let msg2 = "Current tensors = " + tf.memory().numTensors;
        document.querySelector("#msg1").textContent = msg1;
        document.querySelector("#msg2").textContent = msg2;
        setTimeout(train, 10);
    });


}

function trainModel() {
    return model.fit(tr_in, tr_out, {
        shuffle: true,
        epochs: 10
    });
}

