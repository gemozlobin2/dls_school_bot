# Style transfer telegramm bot

Итоговый проект по курсу [dls school](https://www.dlschool.org/base-track). 

Бот умеет переносить стиль с одной картинки на другую а также стилизовать изображения под анимацию.

## Используемые технологии
Для переноса стилей используется [Universal Style Transfer via Feature Transforms](https://arxiv.org/pdf/1705.08086.pdf) алгоритм. Веса модели из оригинальной работы для lua torch можно взять [здесь](https://drive.google.com/file/d/0B8_MZ8a8aoSeWm9HSTdXNE9Eejg/view)

Для стилизации используется алгоритм [CartoonGAN: Generative Adversarial Networks for Photo Cartoonization](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) реализация модели и веса взяты из [репозитория](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch)

## Установка
* Склонировать проект
* Скачать веса по ссылкам выше
* Установить пакеты из requirements.txt  
* Переименовать settings.ini.dist в settings.ini
* Отредактировать настройки в settings.ini
* Получить токен бота и установить его в настройках
