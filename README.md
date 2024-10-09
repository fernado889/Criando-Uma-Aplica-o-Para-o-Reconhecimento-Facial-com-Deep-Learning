# Criando-Uma-Aplica-o-Para-o-Reconhecimento-Facial-com-Deep-Learning
Etapas do Projeto

Coleta de Dados
A coleta de dados é a primeira etapa do projeto. É necessário coletar imagens de rostos para treinar o modelo de reconhecimento facial. As imagens devem ser coletadas em diferentes condições de iluminação, ângulos e expressões faciais.

Pré-Processamento
O pré-processamento é a segunda etapa do projeto. É necessário pré-processar as imagens coletadas para que elas estejam em um formato adequado para o treinamento do modelo. Isso inclui a conversão das imagens para uma escala de cinza, a detecção de rostos e a extração de recursos.

Treinamento do Modelo
O treinamento do modelo é a terceira etapa do projeto. É necessário treinar um modelo de deep learning para reconhecimento facial utilizando as imagens pré-processadas. O modelo mais comum utilizado para isso é o modelo YOLO.

Implementação da Aplicação
A implementação da aplicação é a quarta etapa do projeto. É necessário implementar a aplicação de reconhecimento facial utilizando o modelo treinado. Isso inclui a criação de uma interface de usuário para capturar imagens da webcam e a implementação da lógica de reconhecimento facial.
import cv2
import numpy as np

# Carregar o modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Carregar as classes do modelo YOLO
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Carregar a imagem da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de rostos
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)), 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Loop through each detection
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                # Draw a bounding box around the face
                x, y, x_end, y_end = detection[0:4] * np.array([w, h, w, h])
                cv2.rectangle(frame, (int(x), int(y)), (int(x_end), int(y_end)), (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
