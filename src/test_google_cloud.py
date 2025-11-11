from google.cloud import vision

client = vision.ImageAnnotatorClient()

image = vision.Image()
image.source.image_uri = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"

response = client.web_detection(image=image, max_results=10)
annotations = response.web_detection