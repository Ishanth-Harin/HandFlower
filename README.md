# 🌸 Hand Flower

A real-time webcam effect made with Python, OpenCV, and MediaPipe.

Open your **left palm** to make a flower appear, then move your **right index finger upward** to grow the stem and bloom.

## Features

- Live webcam hand tracking
- Left-palm detection
- Right-hand growth control
- Smooth stem and flower animation
- Stylized glow, petals, and leaves
- Beginner-friendly Python code

## Requirements

- macOS, Windows, or Linux
- Python 3.10–3.12 recommended
- A webcam

## Run on macOS

Open Terminal in this project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Press **Q** or **Esc** to close the camera window.

## macOS camera permission

If the camera does not open, go to:

**System Settings → Privacy & Security → Camera**

Allow the application you are using to access the camera. If you run the program from Terminal, you may need to allow Terminal.

## How the effect works

1. Open your left palm.
2. The program finds your palm position and places the flower there.
3. Move your right index finger upward.
4. The program converts the finger's height into a growth value from 0 to 1.
5. The stem grows, leaves appear, and the flower blooms.

## Project structure

```text
hand-flower/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes

This is a procedural animation: the flower is drawn by Python instead of using a pre-made flower image. This keeps the project self-contained and easy to modify.
