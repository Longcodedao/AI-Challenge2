import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config.set('Date', 'past_date', 'Nov 21, 2023')

with open('config.ini', 'w') as config_file:
    config.write(config_file)
