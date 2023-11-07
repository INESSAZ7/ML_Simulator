import re
import yaml


def dic_to_env(yml_dict, env, str_key):
    """ """
    env=''
    for key, value in yml_dict.items():
        if isinstance(value, dict):
            str_key = str_key + key + '.'
            env = env + dic_to_env(value, env, str_key)
            str_key = str_key.replace(key+'.', '')
        else:
            env= env + str_key + key + '=' + str(value) + '\n'      
    return env

def yaml_to_env(config_file: str) -> str:
    """ """
    yaml_dict = yaml.safe_load(config_file)
    env=''
    env = dic_to_env(yaml_dict, env, '')
    return env

def represents_int(s):
    """ """
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

def from_str_to_type(value: str):
    """ """
    if represents_int(value):
        return int(value)
    if re.findall(r'[\d]*[.][\d]+', value):
        return float(value)
    if value == "True":
        return True
    elif value == "False":
        return False
    return value

def _env_to_yaml(env_str: str, config: dict):
    """ """
    env_dict = {}
    key, value = env_str.split('=')
    value = from_str_to_type(value)
    if '.' in key:
        key_0 = key.split('.')[0]
        new_str = '.'.join(env_str.split('.')[1:])
        if key_0 in config:
            config[key_0].update(_env_to_yaml(new_str, config[key_0]))
        else:
            env_dict[key_0] = _env_to_yaml(new_str, config)

    else:
        env_dict[key]= value
    return env_dict

def env_to_yaml(env_list: str):
    """ """
    env_dict = {}
    env_str_list = env_list.split('\n')[:-1]
    for env_str in env_str_list:
        d = _env_to_yaml(env_str, env_dict)
        env_dict.update(d)
    return yaml.dump(env_dict)
