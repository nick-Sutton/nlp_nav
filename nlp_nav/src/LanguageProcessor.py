from ollama import chat
from rclpy.node import Node
from geometry_msg.msg import Pose

class LanguageProcessor(Node):
    def __init__(self):
        super().__init__('language_processor') # Initialize ROS node
        running = True
        self.coord_dict = {
            "living room":(-6, 3.75),
            "bedroom":(-6.45, -1.35),
            "kitchen": (-3, 1.65),
            "laundry": (1.00, 2.16),
            "office":(5.1, 1.45),
            "library":(6.5, -4.5)
        }
        
        self.publisher = self.create_publisher(Pose, "goal_pose", 10)

        while(running):
            print("What would you like me to do? (Type quit to stop)")
            prompt = input()
            if prompt.lower() == "quit":
                running = False
                break
            
            response = chat(
            model='smallthinker',
            messages=[{'role': 'system', 'content': 'You are an ai assistant that will interpret '
            'user input and determine what room you need to go to in order to fulfill their request or statement.'
            ' You will then output the room and its coordinate value from the coord_dict. This output MUST be like \'room,x,y\' no spaces, no puncuation. x and y are floats.'
            'This is the coord_dict: {coord_dict}'}, 
            {'role':'user', 'content':prompt}]
            )

            data = response.split(',')
            room = data[0]
            x = data[1]
            y = data[2]

            if isinstance(5, float) and isinstance(y, int):
                print("Moving to {room} ({x}, {y})")
                msg = Pose()
                msg.position.x = x
                msg.position.y = y
                self.publisher.publish(msg)
            else:
                print("Error making coords into floats, please try again.")


        

    








    
    




