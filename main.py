import telebot
import io
import gc
import os
import configparser
from PIL import Image
from state import StateEnum, StateManager
from handlers import style_transfer_handler, cycle_gan_handler

config = configparser.ConfigParser()
config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), "settings.ini"))

state_manager = StateManager()
bot = telebot.TeleBot(config["bot"]["token"])
logger = telebot.logger
telebot.logger.setLevel(config["bot"]["log_level"])

messages = {
	'help': """
		Бот понимает следующие команды
		/start Начало работы, сбрасывает состояние бота
		/transfer Перенести стиль одной картинки на другую
		/style Стилизация картинки под мультик
		/help Вывести это сообщение
	""",
	'send_next_photo': "Отправьте картинку стиля который нужно перенести",
	'style': "Отправьте изображение, оно будет стилизовано",
	'transfer': "Отправьте два изображения, бот перенесет стиль второго на первое",
	'transfer_stage1': "Отправьте второе изображение",
	'unclosed_action': "Завершите предыдущую операцию",
	'start_process': "Начинаю обработку, это может занять пару минут",
	'unknown_action': """
		Сначала выберите тип трансформации
		/transfer Перенести стиль одной картинки на другую
		/style Стилизация картинки под мультик
	"""
}


@bot.message_handler(commands=['start'])
def send_welcome(message):
	bot.send_message(message.chat.id, messages['help'])
	state_manager.set_state(message.chat.id, StateEnum.BASE_STATE)


@bot.message_handler(commands=['help'])
def send_welcome(message):
	bot.send_message(message.chat.id, messages['help'])


@bot.message_handler(commands=['transfer'])
def send_welcome(message):
	if state_manager.get_state(message.chat.id) in [StateEnum.BASE_STATE, StateEnum.STYLE_TRANSFER_STAGE_0]:
		bot.send_message(message.chat.id, messages['transfer'])
		state_manager.set_state(message.chat.id, StateEnum.STYLE_TRANSFER_STAGE_0)
	else:
		bot.send_message(message.chat.id, messages['unclosed_action'])


@bot.message_handler(commands=['style'])
def send_welcome(message):
	if state_manager.get_state(message.chat.id) in [StateEnum.BASE_STATE, StateEnum.CYCLE_GAN]:
		bot.send_message(message.chat.id, messages['style'])
		state_manager.set_state(message.chat.id, StateEnum.CYCLE_GAN)
	else:
		bot.send_message(message.chat.id, messages['unclosed_action'])


@bot.message_handler(func=lambda message: True, content_types=['photo'])
def image_handler(message):
	user_id = message.chat.id
	state = state_manager.get_state(message.chat.id)
	file_id = message.photo[-1].file_id
	if state == StateEnum.STYLE_TRANSFER_STAGE_0:
		state_manager.set_attr(user_id, 'img', file_id)
		state_manager.set_state(user_id, StateEnum.STYLE_TRANSFER_STAGE_1)
		bot.send_message(user_id, messages['send_next_photo'])
	elif state == StateEnum.STYLE_TRANSFER_STAGE_1:
		content_img = state_manager.get_attr(user_id, 'img')
		bot.send_message(user_id, messages['start_process'])
		logger.info('Start style transfer')
		photo = run_style_transfer(content_img, file_id)
		bot.send_photo(user_id, photo=photo)
		del photo
		logger.info('Finish style transfer')
		state_manager.set_state(user_id, StateEnum.BASE_STATE)
	elif state == StateEnum.CYCLE_GAN:
		bot.send_message(user_id, messages['start_process'])
		logger.info('Start cycle GAN')
		photo = run_cycle_gan(file_id)
		bot.send_photo(user_id, photo=photo)
		del photo
		logger.info('Finish cycle GAN')
		state_manager.set_state(user_id, StateEnum.BASE_STATE)
	else:
		bot.send_message(user_id, messages['unknown_action'])
	gc.collect()


def convert_pil_to_bytestream(img_data: Image):
	bio = io.BytesIO()
	bio.name = 'image.jpeg'
	img_data.save(bio, 'JPEG')
	bio.seek(0)
	return bio


def get_tg_file_as_pil(file_id):
	file_info = bot.get_file(file_id)
	image_data = bot.download_file(file_info.file_path)
	return Image.open(io.BytesIO(image_data))


def run_style_transfer(content_img_id, style_img_id):
	content = get_tg_file_as_pil(content_img_id)
	style = get_tg_file_as_pil(style_img_id)
	return convert_pil_to_bytestream(style_transfer_handler(config, content, style))


def run_cycle_gan(img_id):
	image = get_tg_file_as_pil(img_id)
	return convert_pil_to_bytestream(cycle_gan_handler(config, image))


bot.polling()