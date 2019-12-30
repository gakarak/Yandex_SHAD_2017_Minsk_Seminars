# Запуск inference для детектирования зернышек


### берем docker-образ 
```
docker pull alxkalinovsky/yshad-kaggle-counter:latest

```

### запускаем inference

```
docker run -it -v /path/to/data:/data -v /path/to/model:/model alxkalinovsky/yshad-kaggle-counter:latest bash -c "python3 example-counter-predict.py -i=/data/idx.txt -m=/model"
```
