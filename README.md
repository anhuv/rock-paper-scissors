# rock-paper-scissors


An AI to play the Rock Paper Scissors game

## Requirements
- Python 3
- Keras
- Tensorflow
- OpenCV

## Set up instructions
1. Clone the repo.
```sh
$ git clone https://github.com/ungvietanh20172394/rock-paper-scissors.git
$ cd rock-paper-scissors
```

2. Install the dependencies
```sh
$ pip install -r requirements.txt
``` 
cài bản mới nhất, không cài theo requirements.txt

3. Gather Images for each gesture (rock, paper and scissors and None):
In this example, we gather 200 images for the "rock" gesture
```sh
$ python gather_images.py rock 200
```

4. Train the model
```sh
$ python train.py
```

5. Test the model on some images
```sh
$ python test.py <path_to_test_image>
```
không cần làm bước 5

6. Play the game with your computer!
```sh
$ python play.py
```
