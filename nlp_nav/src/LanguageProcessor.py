from ollama import chat
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class LanguageProcessor(Node):

    def __init__(self):
        super().__init__('language_processor')
        self.publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)   
        


    def main(self):
        
        #super().__init__('language_processor') # Initialize ROS node
        running = True
        coord_dict = {
            "living room":(-6, 3.75),
            "bedroom":(-6.45, -1.35),
            "kitchen": (-3, 1.65),
            "laundry": (1.00, 2.16),
            "office":(5.1, 1.45),
            "library":(6.5, -4.5)
        }
        
        #self.publisher = self.create_publisher(Pose, "goal_pose", 10)

        while(running):
            print("What would you like me to do? (Type quit to stop)")
            prompt = input()
            if prompt.lower() == "quit":
                running = False
                break
            
            print(f"Got input {prompt}")
            
            printed = False
            for i in range(5):
                response = chat(
                model='qwen2.5:3b',
                messages=[{'role': 'system', 'content': 'You are an ai assistant that will interpret '
                'user input and determine what room is most related to the statement '
                f'This is the list of options: {coord_dict.keys()}' 
                'output only one word'},
                {'role':'user', 'content':prompt}], 
                
                )
                room = response['message']['content']
                if room in coord_dict.keys():
                    printed = True
                    break

            if not printed:
                print("I'm sorry, please try again.")
                continue
               
            

            

            


            
            print(room)
           
            

            x, y = coord_dict[room]
            

            print(f"Moving to {room} ({x}, {y})")
            msg = PoseStamped()
            msg.position.x = x
            msg.position.y = y
            self.publisher.publish(msg)
       

    if __name__ == '__main__':
        main()


    








    
    




