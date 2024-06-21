from typing import Callable


from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def convert_images_to_arrays(*encodings: str):
    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        cv_bridge = CvBridge()
        
        def wrapper(self, *image_msgs: Image):
            if len(encodings) != len(image_msgs):
                raise ValueError("Number of encodings must match the number of image messages")
            
            images = [cv_bridge.imgmsg_to_cv2(msg, encoding) for msg, encoding in zip(image_msgs, encodings)]
            return func(self, *images)

        return wrapper

    return decorator
