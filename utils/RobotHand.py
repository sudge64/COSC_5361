from utils.PCA9685 import PCA9685

class RobotHand:
    def __init__(self, pwm, fingers, low=600, high=2300, sthumb=2000, shigh=1300):
        self.pwm = pwm
        self.fingers = fingers
        self.low = low
        self.high = high
        self.sthumb = sthumb
        self.shigh = shigh
        
    def set_finger(self, name, value):
        self.pwm.setServoPulse(self.fingers[name], value)
        
    def rock(self):
        self.set_finger("thumb", self.high)
        self.set_finger("index", self.low)
        self.set_finger("middle", self.low)
        self.set_finger("ring", self.low)
        self.set_finger("pinky", self.low)
    
    def paper(self):
        self.set_finger("thumb", self.low)
        self.set_finger("index", self.high)
        self.set_finger("middle", self.high)
        self.set_finger("ring", self.high)
        self.set_finger("pinky", self.high)

    def scissors(self):
        self.set_finger("thumb", self.high)
        self.set_finger("index", self.high)
        self.set_finger("middle", self.high)
        self.set_finger("ring", self.low)
        self.set_finger("pinky", self.low)

    def numeric_one(self):
        self.set_finger("thumb", self.high)
        self.set_finger("index", self.high)
        self.set_finger("middle", self.low)
        self.set_finger("ring", self.low)
        self.set_finger("pinky", self.low)
        
    def numeric_two(self):
        self.set_finger("thumb", self.high)
        self.set_finger("index", self.high)
        self.set_finger("middle", self.high)
        self.set_finger("ring", self.low)
        self.set_finger("pinky", self.low)

    def numeric_three(self):
        self.set_finger("thumb", self.high)
        self.set_finger("index", self.high)
        self.set_finger("middle", self.high)
        self.set_finger("ring", self.high)
        self.set_finger("pinky", self.low)

    def numeric_four(self):
        self.set_finger("thumb", self.high)
        self.set_finger("index", self.high)
        self.set_finger("middle", self.high)
        self.set_finger("ring", self.high)
        self.set_finger("pinky", self.high)
        
    def spiderman(self):
        self.set_finger("thumb", self.low)
        self.set_finger("index", self.high)
        self.set_finger("middle", self.low)
        self.set_finger("ring", self.low)
        self.set_finger("pinky", self.high)
        
    def tea_time(self):
        self.set_finger("thumb", self.high)
        self.set_finger("index", self.low)
        self.set_finger("middle", self.low)
        self.set_finger("ring", self.low)
        self.set_finger("pinky", self.high)
        
    def hang_time(self):
        self.set_finger("thumb", self.low)
        self.set_finger("index", self.low)
        self.set_finger("middle", self.low)
        self.set_finger("ring", self.low)
        self.set_finger("pinky", self.high)
        
    def boy_scout(self):
        self.set_finger("thumb", self.high)
        self.set_finger("index", self.high)
        self.set_finger("middle", self.high)
        self.set_finger("ring", self.high)
        self.set_finger("pinky", self.low)
    
    def thumbs_up(self):
        self.set_finger("thumb", self.low)
        self.set_finger("index", self.low)
        self.set_finger("middle", self.low)
        self.set_finger("ring", self.low)
        self.set_finger("pinky", self.low)
        
    def grab(self):
        self.set_finger("thumb", self.sthumb)
        self.set_finger("index", self.shigh)
        self.set_finger("middle", self.shigh)
        self.set_finger("ring", self.shigh)
        self.set_finger("pinky", self.shigh)
        
    # Call paper for five
    def numeric_five(self):
        self.paper()

    # Call scissors for victory
    def victory(self):
        self.scissors()
        
    # Call rock for reset
    def reset(self):
        self.rock()
