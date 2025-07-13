# Deep Q-Learning With Atari: Breakout-v4

This project implements Deep Q-Learning (DQN) agents using both **CNN (Convolutional Neural Network)** and **MLP (Multi-Layer Perceptron)** policies to play the Atari game **Breakout-v4**. The primary objective is to train reinforcement learning agents that learn to play the game effectively using different policy architectures and compare their performance.

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

- **Environment**: `Breakout-v4` (via `gymnasium`)
- **Libraries Used**:
  - `stable-baselines3`
  - `gymnasium`
  - `PyTorch`
  - `imageio` (for recording gameplay)

- **Training Policies**:
  - `CNNPolicy`: Uses convolutional layers for pixel-based state representation
  - `MLPPolicy`: Uses dense layers (less optimal for image-based input)

---

## Summary of Findings

- **CNNPolicy outperformed MLPPolicy** significantly in both average reward and episode length.
- The CNN-based model will be used for final gameplay evaluation.
- MLP-based training was included for comparison and learning purposes.

---

## How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/g-tumwesigye/deep-q-learning_atari_assignment.git
cd deep-q-learning_atari_assignment
````

### 2. Download the Pretrained Models

Due to GitHub’s file size limits, the trained `.zip` models are not included in the repo.

> Go to [`models/README.md`](models/README.md) for download links.

Place both `.zip` files into the `models/` folder.

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
| **Geofrey**         | Model training (CNN & MLP), repo setup, evaluation summary |
| **Justice**         | Implemented `play.py`, test & record gameplay                |
| **Steven**          | Analyzed & compared CNN vs MLP performance                   |
| **Peter**           | Recorded the hyperparameter configs & prepared the README.md           |


---

## models/ Directory

Please refer to [`models/README.md`](models/README.md) for:

* Download links to `.zip` models
* Notes on CNN vs MLP performance

---

## Final Output

* Trained DQN models
* Gameplay recording 
* Codebase for reproducibility
* Clean documentation
* Presentation-ready material

---


```
