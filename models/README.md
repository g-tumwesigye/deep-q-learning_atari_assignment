#  Models Directory

This folder contains the trained Deep Q-Learning models for the Atari Breakout game.

## Contents

- `dqn_model_cnn.zip`  
  → Trained model using **CNNPolicy**.  
  → This is the **main model** used for testing in `play.py`.

- `dqn_model_mlp.zip`  
  → Trained model using **MLPPolicy**.  
  → Included for **comparison purposes only**.

## Notes

- These files are **not tracked in GitHub** due to large file size limitations.
- To run `play.py`, make sure the `dqn_model_cnn.zip` file is present in this folder.

## How to Use

1. Download the models from the shared Google Drive:
   - [ CNN Policy Model](https://drive.google.com/file/d/136qtgyfMiIOv9GIMJEGDNrdh9oM3TYqm/view?usp=drive_link)
   - [ MLP Policy Model](https://drive.google.com/file/d/1jREqaz0vK7EBRpv-faOmCbjdTAszH_nN/view?usp=drive_link)

2. Place both `.zip` files into this `models/` directory.

3. Run:

   ```bash
   python play.py



- The CNN model outperformed the MLP model and is the recommended one for evaluation and demonstration.



