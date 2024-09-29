import requests

def send_image_and_get_depth():
        response = requests.post('http://150.136.71.52:8775/api/image-dimensions', files = {'image': open('room.jpeg', 'rb')})
        print(response.text)


if __name__ == "__main__":
    
    send_image_and_get_depth()

