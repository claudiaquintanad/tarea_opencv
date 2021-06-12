import cv2
import sys

# Get user supplied values
imagePath = "pantallazo.png" #pasamos el nombre de la imagen con su correcta ruta
cascPath = "haarcascade_frontalface_default.xml" #aquí se encuentra la data que detecta caras

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath) #se crea este cascade y se inicializa con faceCascade. Eso carga la face cascade en la memoria para que esté lista para usar.

# Read the image
image = cv2.imread(imagePath) #para leer la imagen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convierte la imagen a escala de grises (greyscale).

# Detect faces in the image
faces = faceCascade.detectMultiScale( 
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
#esta función detecta la cara actual y es la parte clave de nuestro código:
    #1. La función detectMultiScale es una función general que detecta objetos. Como llamamos faceCascade, detectará eso, ed decir, caras.
    #2.En la línea 17 vemos la opción de escala de grises
    #3.En la línea 18 está scaleFactor. Como algunas caras están más cercas de la cámara que otras, pueden parecer más grandes que las que están más alejadas, es por eso que scaleFactor compensa esto.
    #4.El algoritmo de detección usa una ventana en movimiento que detecta en objetos. minNeighbors define cuantos objetos son detectados cerca del mismo antes de declararlo como una cara encontradaa. Por otro lado, el minSize da el tamaño de cada ventana.
#la función retorna una lista de rectángulos donde se cree que hay una cara. Luego hace un loop donde pieza que encontró algo.
print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) #esta función retorna 4 valores: la ubicación de x e y del rectángulo, y el ancho y largo del rectángulo (w, h) (dibuja el rectángulo)
    cv2.putText(image, 'Cara', (x,y-10), 2, 0.7,(255,0,0),2,cv2.LINE_AA) #esta función hace que aparezca "cara" encima del rectángulo
cv2.imshow("Faces found", image) 
cv2.waitKey(0)