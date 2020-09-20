## Как запустить


### 1. Поставить докер
https://docs.docker.com/get-docker/

### 2. Запустить докер процесс

### 3. Скачать образ 

```
docker pull fogugy/dobro-ml
```

### 4. Спулить ml-проект
```
git clone https://github.com/fogugy/dobro_ml.git
```

### 5. Дать права докеру на шаринг файлов проекта
settings/ resources/ file sharing

### 6. Перейти в директорию с проектом

### 7. Запустить оттуда 
```
bash runml.sh
```

Сервер запускается в докер-контейнере на 4444 порту.

/test - get метод для проверки

/project_type - post метод для определения класса проекта. Возвращает json 
{"type": "personal"/"common"}

/msg_score - post метод для определения вероятности неприятного сообщения. Возвращает json 
{"score": float[0-1]}
