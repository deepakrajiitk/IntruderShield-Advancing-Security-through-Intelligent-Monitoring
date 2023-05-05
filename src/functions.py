import os
from collections import OrderedDict
# from twilio.rest import Client


def sendSMS(name, roomnumber, msg):
    account_sid = 'ACc7611854961d275d9f71aea2a5540781'
    auth_token = 'b406726163d4a445335a2e7ffeb31a0e'
    client = Client(account_sid, auth_token)

    # Send a message using the Twilio API
    message = client.messages.create(
        to='+919643225192',  # recipient's phone number
        from_='+15074172693',  # your Twilio phone number
        body=msg
    )

    # Print the message ID
    print('Message sent with ID:', message.sid)

# clear those names from dictionary whose probablity has become 0


def update_ordered_dict(prev_objects: OrderedDict, decrement: float) -> OrderedDict:
    for key in list(prev_objects.keys()):
        centroid, prob = prev_objects[key]
        prob = prob - decrement
        if prob > 0:
            prev_objects[key] = (centroid, prob)
        else:
            prev_objects.pop(key)
    return prev_objects


# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))


def create_directory(username):
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(parent_dir, '..', 'dataset')
    authorized_dir = os.path.join(dataset_dir, 'authorized')
    user_dir = os.path.join(authorized_dir, username)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        return True
    else:
        return False


def upload_photos(username, files, text_input):
    # Get the user's directory path
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(parent_dir, '..', 'dataset')
    user_dir = os.path.join(dataset_dir, 'authorized', username)
    # Get the files uploaded by the user
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    for file in files:
        # Check that the file is an image file
        if file.filename.split('.')[-1].lower() in allowed_extensions:
            # Save the file in the user's directory
            file.save(os.path.join(user_dir, file.filename))
        else:
            return "Error: only image files (PNG, JPG, JPEG) are allowed!"

    # Save the text input to a text file in the user's directory
    with open(os.path.join(user_dir, 'Authorized_Rooms.txt'), 'w') as f:
        f.write(text_input)

    return f"{len(files)} photos uploaded successfully with text input: {text_input}!"
