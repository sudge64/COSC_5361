from tkinter import *
from utils.PCA9685 import PCA9685
from utils.RobotHand import RobotHand
from utils.async_hand import AsyncHand

def _hand_factory() -> RobotHand:
    pwm = PCA9685(0x40, debug=False)
    pwm.setPWMFreq(50)
    fingers = {
        "thumb": 1,
        "index": 2,
        "middle": 3,
        "ring": 4,
        "pinky": 5,
    }
    low, high = 600, 2300
    sthumb, shigh = 2000, 1300
    
    return RobotHand(pwm, fingers, low, high, sthumb, shigh)


def create_hand():
    hand = AsyncHand(_hand_factory, delay=0.5)
    
    gestures = {
        "rock" : hand.rock,
        "paper" : hand.paper,
        "scissors" : hand.scissors,
        "one" : hand.numeric_one,
        "two" : hand.numeric_two,
        "three" : hand.numeric_three,
        "four" : hand.numeric_four,
        "five" : hand.numeric_five,
        "victory" : hand.victory,
        "spiderman" : hand.spiderman,
        "tea_time" : hand.tea_time,
        "hang_time" : hand.hang_time,
        "boy_scout" : hand.boy_scout,
        "thumbs_up" : hand.thumbs_up,
        "grab" : hand.grab,
        "reset" : hand.reset
    }
    
    return hand, gestures

def main():
    
    hand, gestures = create_hand()
    
    root = Tk()
    
    root.title("Robot Hand Control")
    
    frame = Frame(root)
    frame.grid(row=0, column=0, sticky="news")
        
    for i, (key, value) in enumerate(gestures.items()):
        btn = Button(frame, text=key, command=value)
        btn.grid(row=i // 4, column=i % 4, sticky="news")
    
    #slider = Scale(
            #root,
            #from_ = 2300,
            #to = 600,
            #orient = 'vertical',
            #command = lambda value: RobotHand.set_finger('index', value)
            #)
    #slider.grid(row = 5, column = 0)
    
    #for i in range(1, 6):
    root.mainloop()

def show_values(value, additional_param):
    print(f"Slider Value: {value}, Additional Param: {additional_param}")

if __name__=='__main__':
    main()

