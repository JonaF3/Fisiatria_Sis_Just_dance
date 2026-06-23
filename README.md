# physical-rehabilitation-pose-app

Sparsh Gupta, Chang Jun Park, and Akshat Jain

## Project Description

This project has been adapted into a physical rehabilitation exercise app.

The objective is to follow a reference movement on screen and receive feedback from posture matching.
The system is intended for exercise sessions such as abdominal flexion and extension.
The app uses pose-estimation techniques to detect the patient's movement and calculate a session score.
We also utilize OpenCV and Tensorflow for computations on the video feed obtained from the user's webcam.

The app uses a reference video and webcam pose tracking to support guided movement practice.

## Dependencies

| Package    | Uses                    |
|------------|-------------------------|
| NumPy      | Array Operations        |
| OpenCV     | Computer Vision         |
| Playsound  | Plays sound file        |
| TensorFlow | Machine Learning        |
| Mutagen    | Audio Length Extraction |
| Pytest     | Testing functions       |


 The dependencies are present in `requirements.txt` and can be installed using the following in terminal/command prompt (make sure to have your present working directory as this repo):
 
 ```
 pip install -r requirements.txt
 ```


## Computational Requirements

To have the best performance for this project, your machine must meet the following minimum requirements:

- **RAM:** At least 16GB of RAM is required to ensure smooth performance without lagging of video frames. We recommend using a machine with 32GB of RAM.

- **Camera:** A camera of at least 1080p resolution is recommended, although not required.

- **Processor:** We recommend using a multi-core processor with a clock speed of at least 2GHz to ensure fast computation times.

- **Graphics Card:** A dedicated graphics card with at least 2GB of memory is highly recommended to accelerate rendering and visualization tasks.


## Code Execution

To run the code in this repo, please clone this repo to your local machine and run `just_dance_gui.py` in either a Python-compatible IDE or if using a terminal/command prompt (make sure to have your present working directory as this repo):

```
python3 just_dance_gui.py
```

Executing `just_dance_gui.py` using python will automatically open the GUI window to start the rehabilitation session.

If you want to exit the app anytime during its execution, press ESC from the lobby.


## Exercise Media Credits

**Disclaimer**: We hereby declare that we do not own the rights to any media used in this project.
All rights belong to the owner.
No copyright infringement intended.

Current media is used as a movement reference for rehabilitation exercises.

## Unit Tests

We test the functionality of our code by running pytest test cases. 
We mainly test whether the score calculated lies between 0-100, the top 5 leaderboard scores obtained are sorted in descending order, and the angles for the video and user's joints lies between 0-180.
The test files are present in the `test` directory. You can test the code by running the following in terminal/command prompt (make sure to have your present working directory as this repo):

```
pytest test_just_dance.py
```
