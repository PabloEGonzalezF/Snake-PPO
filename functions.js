console.log("Main script cargado correctamente")

const grid_square = 20;
const Canvas_Size = 280;
const Rows = 14;
const Cols = 14;

const empty = 0;
const body = 1;
const head = 2;
const food = 3;

const STATE_SIZE = Rows * Cols;
const ACTION_SIZE = 4;

const FPS = 1000; //Para setInterval

const NUM_ENVS = 4;
const envs = [];
const states = [];

const GAMMA = 0.99;
const LAMBDA = 0.95;
const EPOCHS = 3;
const CLIP_RATIO = 0.2;

let episodeRewards = new Array(NUM_ENVS).fill(0);
let episodeSteps = new Array(NUM_ENVS).fill(0);
let totalEpisodes = 0;

const grid = document.getElementById("agents");

class SnakeGame{
    constructor(canvas){
        this.canvas = canvas;
        this.ctx = this.canvas.getContext("2d");
        this.map = new Uint8Array(Rows * Cols);
        this.reset();
    }

    reset(){
        this.playerX = Math.floor(Cols / 2);
        this.playerY = Math.floor(Rows / 2);
        this.length = 3;
        this.score = 0;
        this.snake = [];
        this.map.fill(empty);
        this.VectorX = 0;
        this.VectorY = 0;
        this.foodX = 0;
        this.foodY = 0;
        this.step = 0;
        this.maxSteps = 200;
        this.generateFood();

        this.distprev = Math.abs(this.playerX - this.foodX) + Math.abs(this.playerY - this.foodY);
        
        return this.getState()
    }

    generateFood(){
        let pos;

        do{
            pos = Math.floor(Math.random() * this.map.length)
        }while(this.map[pos] == body || this.map[pos] == head)

        this.foodX = pos % Cols;
        this.foodY = Math.floor(pos / Cols);
    }

    getState(){
        return new Uint8Array(this.map);
    }

    Step(action){

        if(action == 0 && this.VectorY !== 1){
            //console.log("W");
            this.VectorY = -1; 
            this.VectorX = 0;
        }
        else if(action == 1 && this.VectorY !== -1) {
            //console.log("S")
            this.VectorY = 1;
            this.VectorX = 0;
        }
        else if(action == 2 && this.VectorX !== 1) {
            //console.log("A")
            this.VectorX = -1;
            this.VectorY = 0;
        }
        else if(action == 3 && this.VectorX !== -1) {
            //console.log("D")
            this.VectorX = 1;
            this.VectorY = 0;
        }

        let nextX = this.playerX + this.VectorX;
        let nextY = this.playerY + this.VectorY;
        this.step++;
        let reward = -0.01;
        let bool = false;

        if(nextX < 0 || nextX >= Cols || nextY < 0 || nextY >= Rows || this.map[nextY * Cols + nextX] == body){
            reward = -0.5;
            bool = true;
            return {state: this.reset(), reward, bool}
        }

        if(this.step >= this.maxSteps){
            reward = -0.5;
            bool = true
            return {state: this.reset(), reward, bool}
        }

        this.playerX = nextX;
        this.playerY = nextY;

        this.snake.push({x:this.playerX, y:this.playerY});

        while(this.snake.length > this.length){
            this.snake.shift();
        }

        this.map.fill(empty);

        this.ctx.fillStyle = "black";
        this.ctx.fillRect(0,0, Canvas_Size, Canvas_Size);

        for(let i = 0; i < this.snake.length; i++){
            this.map[this.snake[i].y * Cols + this.snake[i].x] = body;
            if(i == this.snake.length - 1) this.map[this.snake[i].y * Cols + this.snake[i].x] = head;
        }

        this.map[this.foodY * Cols + this.foodX] = food;

        const currentDist = Math.abs(this.playerX - this.foodX) + Math.abs(this.playerY - this.foodY);

        if(currentDist < this.distprev){
            reward += 0.1;
        }
        else if(currentDist > this.distprev){
            reward -= 0.05;
        }

        if(this.foodX == this.playerX && this.foodY == this.playerY){
            this.length++;
            this.score = this.length - 3;
            reward += 0.5;
            this.maxSteps = this.step + 200;
            this.generateFood();
        }

        this.distprev = currentDist;

        return {state: this.getState(), reward, bool}
    }

    draw(){
        this.ctx.fillStyle = "black";
        this.ctx.fillRect(0, 0, Canvas_Size, Canvas_Size);
        for(let i = 0; i < this.map.length; i++){
            let x = i % Cols;
            let y = Math.floor(i / Cols);

            if(this.map[i] == head){
                this.ctx.fillStyle = "lime";
                this.ctx.fillRect(x * grid_square, y * grid_square, grid_square, grid_square);
            }
            else if(this.map[i] == body){
                this.ctx.fillStyle = "green";
                this.ctx.fillRect(x * grid_square, y * grid_square, grid_square, grid_square);
            }
            else if(this.map[i] == food){
                this.ctx.fillStyle = "red";
                this.ctx.fillRect(x * grid_square, y * grid_square, grid_square, grid_square);
            }
        }
    }
}

function createModel(){
    const input = tf.input({shape: [STATE_SIZE]});
    const d1 = tf.layers.dense({units:128, activation: "relu"}).apply(input);
    const d2 = tf.layers.dense({units:128, activation: "relu"}).apply(d1);

    const logits = tf.layers.dense({units: ACTION_SIZE}).apply(d2);
    const value = tf.layers.dense({units: 1}).apply(d2)

    return tf.model({inputs: input, outputs: [logits, value]});
}

const model = createModel();
const optimizer = tf.train.adam(0.0003);

function sample(logits){
    const probabilidad = tf.softmax(logits);
    const action = tf.multinomial(probabilidad, 1).dataSync()[0];
    probabilidad.dispose();
    return action;
}

function createEnvs(){
    for(let i = 0; i < NUM_ENVS; i++){
        const canvas = document.createElement("canvas");
        canvas.width = Canvas_Size;
        canvas.height = Canvas_Size;
        grid.appendChild(canvas);

        const env = new SnakeGame(canvas);
        envs.push(env);
        states.push(env.reset());
    }
}

function normalizeState(state){
    const out = new Float32Array(state.length);
    for(let i = 0; i < state.length; i++){
        out[i] = state[i] / 3.0;
    }
    return out;
}

function trainStep(){

    tf.tidy(()=>{

        const stateBatch=[];
        const actionBatch=[];
        const rewardBatch=[];
        const doneBatch=[];
        const valueBatch=[];
        const nextValueBatch=[];
        const logProbBatch=[];

        for(let i=0;i<NUM_ENVS;i++){

            const normState=normalizeState(states[i]);

            const stateTensor=tf.tensor2d([normState]);

            const [logits,value]=model.predict(stateTensor);

            const action=sample(logits);

            const probs=tf.softmax(logits);

            const prob=probs.dataSync()[action];

            const logProb=Math.log(prob+1e-10);

            const valueScalar=value.dataSync()[0];

            const result=envs[i].Step(action);

            envs[i].draw();

            let nextValue=0;

            if(!result.done){
                const nextNorm=normalizeState(result.state);

                const nextTensor=tf.tensor2d([nextNorm]);

                const [,nextValueTensor]=model.predict(nextTensor);

                nextValue=nextValueTensor.dataSync()[0];

                nextTensor.dispose();
                nextValueTensor.dispose();
            }

            stateBatch.push(normState);
            actionBatch.push(action);
            rewardBatch.push(result.reward);
            doneBatch.push(result.done?1:0);
            valueBatch.push(valueScalar);
            nextValueBatch.push(nextValue);
            logProbBatch.push(logProb);

            episodeRewards[i]+=result.reward;
            episodeSteps[i]++;

            if(result.done){
                episodeRewards[i]=0;
                episodeSteps[i]=0;
                totalEpisodes++;
            }

            states[i]=result.state;

            stateTensor.dispose();
            logits.dispose();
            value.dispose();
            probs.dispose();

        }

        const statesTensor=tf.tensor2d(stateBatch);
        const actionsTensor=tf.tensor1d(actionBatch,"int32");
        const rewardsTensor=tf.tensor1d(rewardBatch);
        const valuesTensor=tf.tensor1d(valueBatch);
        const nextValuesTensor=tf.tensor1d(nextValueBatch);
        const donesTensor=tf.tensor1d(doneBatch);
        const oldLogTensor=tf.tensor1d(logProbBatch);

        const returns=rewardsTensor.add(nextValuesTensor.mul(GAMMA).mul(tf.scalar(1).sub(donesTensor)));

        const advantages=returns.sub(valuesTensor);

        const advMean=advantages.mean();
        const advStd=tf.moments(advantages).variance.sqrt().add(1e-8);

        const normAdvantages=advantages.sub(advMean).div(advStd);

        optimizer.minimize(()=>{

            const [logits,values]=model.predict(statesTensor);

            const probs=tf.softmax(logits);

            const actionOneHot=tf.oneHot(actionsTensor,ACTION_SIZE);

            const selected=tf.sum(probs.mul(actionOneHot),1);

            const logProbs=tf.log(selected.add(1e-10));

            const ratio=tf.exp(logProbs.sub(oldLogTensor));

            const clipped=tf.clipByValue(ratio,1-CLIP_RATIO,1+CLIP_RATIO);

            const surrogate1=ratio.mul(normAdvantages);
            const surrogate2=clipped.mul(normAdvantages);

            const policyLoss=tf.neg(tf.mean(tf.minimum(surrogate1,surrogate2)));

            const valueLoss=tf.mean(
                tf.square(returns.sub(tf.squeeze(values)))
            );

            const entropy=tf.neg(
                tf.mean(tf.sum(probs.mul(tf.log(probs.add(1e-10))),1))
            );

            const totalLoss= policyLoss.add(valueLoss.mul(0.5)).add(entropy.mul(-0.01)); 
            return totalLoss;

        });

        statesTensor.dispose();
        actionsTensor.dispose();
        rewardsTensor.dispose();
        valuesTensor.dispose();
        nextValuesTensor.dispose();
        donesTensor.dispose();
        oldLogTensor.dispose();
        returns.dispose();
        advantages.dispose();
        normAdvantages.dispose();

    });

}

createEnvs();

let speed=60;

const slider=document.getElementById("speedSlider");
const speedValue=document.getElementById("speedValue");

slider.oninput=()=>{
    speed=parseInt(slider.value);
    speedValue.innerText=speed;
};

function loop(){
    for(let i=0;i<speed;i++){
        trainStep();
    }

    requestAnimationFrame(loop);
}

loop();