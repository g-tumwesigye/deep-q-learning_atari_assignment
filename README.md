# Deep Q-Learning with Atari: Breakout-v4

This project demonstrates Deep Q-Learning (DQN) using **Stable Baselines3** to train an agent to play **Breakout-v4**, an Atari environment from Gymnasium.

We trained and compared two policy networks:
- `CnnPolicy` – uses convolutional layers
- `MlpPolicy` – uses fully connected layers 

---

##  Project Structure

```
deep-q-learning_atari_assignment/
├── README.md 
├── train.py 
├── train_mlp.py 
├── play.py 
├── evaluation.log 
├── dqn_tensorboard/
│ └── DQN_1/
│ └── events.out.tfevents.*
├── models/ 
│ ├── README.md
│ ├── dqn_model_cnn.zip 
│ └── dqn_model_mlp.zip             
````

---

## Key Details
- **Environment:** `Breakout-v4`
- **Library:** Gymnasium + Stable Baselines3
- **Framework:** PyTorch (via SB3)
- **Libraries Used**:
  - `stable-baselines3`
  - `gymnasium`
  - `PyTorch`
  - `imageio` (for recording gameplay)

- **Training Policies**:
  - `CNNPolicy`: Uses convolutional layers for pixel-based state representation
  - `MLPPolicy`: Uses dense layers (less optimal for image-based input)

---

## DQN agent summary

We implemented and trained a DQN agent using two policy types:

| Policy      | Performance Notes                                           |
|-------------|-------------------------------------------------------------|
| `CnnPolicy` | Performed significantly better, learns spatial patterns     |
| `MlpPolicy` | Poor performance, lacks ability to process image inputs     |

The CNN-based model (`dqn_model_cnn.zip`) is used in `play.py`.

![Your Alt Text](models/Model%20Comparison.png)
---

## Summary of Findings

- **CNNPolicy outperformed MLPPolicy** significantly in both average reward and episode length.
- The CNN-based model will be used for final gameplay evaluation.
- MLP-based training was included for comparison and learning purposes.

---

### Hyperparameter Tuning Table

| Hyperparameter Set | Learning Rate | Gamma | Batch Size | Epsilon (start → end, decay) | Observed Behavior                                |
|--------------------|---------------|--------|------------|-------------------------------|--------------------------------------------------|
| Set 1              | 0.0001        | 0.99   | 32         | 1.0 → 0.1, 100000 steps        | Stable reward increase, slow learning            |
| Set 2              | 0.0005        | 0.98   | 64         | 1.0 → 0.05, 50000 steps        | Improved exploration, faster convergence         |
| Set 3              | 0.001         | 0.95   | 32         | 0.9 → 0.1, 20000 steps         | Learning too aggressive, less stability          |
| Set 4 (Peter)      | 0.0003        | 0.97   | 64         | 1.0 → 0.05, 30000 steps        | Moderate improvement, smooth reward progression  |

We've compiled this table to illustrate how adjusting various settings impacts our model. While you might notice that "Set 1" led to "slow learning," we prefer it because it gave us a "stable reward increase." This stability, which we achieved by using a very careful learning rate, a high gamma (meaning we're thinking long-term), and a slow, steady decay for our exploration strategy (epsilon), all these point to a really solid and robust learning process. For us, that consistent and reliable long-term performance is far more important than just getting a quick, but potentially unstable, result.

---

## How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/g-tumwesigye/deep-q-learning_atari_assignment.git
cd deep-q-learning_atari_assignment
````

### 2. Install dependencies
pip install stable-baselines3[extra] gymnasium[atari] torch imageio

### 3. Download the Pretrained Models

Due to GitHub’s file size limits, the trained `.zip` models are not included in the repo.

> Go to [`models/README.md`](models/README.md) for download links.

Please manually download and place them in models/

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If no `requirements.txt` exists, install manually:

```bash
pip install stable-baselines3[extra] gymnasium[atari] imageio
```

### 4. Run Training 

To retrain from scratch:

```bash
python train.py       
python train_mlp.py   
```

### 5. Run and Record Gameplay

Once `play.py` is implemented, you’ll be able to run:

```bash
python play.py
```

It will generate a `.gif` or `.mp4` of the gameplay in the root directory.

---

## Team Responsibilities

| Name                | Role                                                       |
| ------------------- | ---------------------------------------------------------- |
| **Geofrey**         | Model training (CNN & MLP), repo setup                       |
| **Justice**         | Implemented `play.py`, test & record gameplay                |
| **Steven**          | Analyzed & compared CNN vs MLP performance                   |
| **Peter**           | Recorded the hyperparameter configs, fine tuned Set 4 & prepared the README.md |


---

## models/ Directory

Please refer to [`models/README.md`](models/README.md) for:

* Download links to `.zip` models
* Notes on CNN vs MLP performance

---
---

## Video Demo

Watch the agent playing Breakout-v4 (loaded from `play.py`):

🔗 [Click to Watch the Gameplay Video](https://youtu.be/7NKadKI9rng)

---
